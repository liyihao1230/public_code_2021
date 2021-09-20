import sys
import os
import glob
import time
import argparse

import re
import json
import random
import logging

import cv2
from PIL import Image
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from imgaug import augmenters as iaa
from torch.autograd import Variable
# from pytorchtools import EarlyStopping
from sklearn import preprocessing

from tools.Resnet_Family import ResNet, ResNetv2, ResNeXt, SeResNet, SeResNeXt

from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"  # 0,
torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23496', rank=0, world_size=1)
from torch.utils.data.distributed import DistributedSampler

local_rank = torch.distributed.get_rank()

from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


# 定义dataset
class RetrivalDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, used_data, transform=None):  # labels, imgs 为 list
        self.used_data = used_data
        self.len = len(used_data)
        self.transform = transform

    def __getitem__(self, index):
        item = self.used_data[index]
        anchor = item['anchor']
        positives = item['positive']
        negatives = item['negative']

        anchor = cv2.imread(anchor)
        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
        anchor = cv2.resize(anchor, (224, 224))
        anchor = np.transpose(anchor, (2, 0, 1))
        anchor = torch.FloatTensor(anchor)

        positive = []
        for p in positives:
            temp = cv2.imread(p)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp = cv2.resize(temp, (224, 224))
            temp = np.transpose(temp, (2, 0, 1))
            temp = torch.FloatTensor(temp)
            positive.append(temp)
        positive = torch.stack(positive)

        negative = []
        for n in negatives:
            temp = cv2.imread(n)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp = cv2.resize(temp, (224, 224))
            temp = np.transpose(temp, (2, 0, 1))
            temp = torch.FloatTensor(temp)
            negative.append(temp)
        negative = torch.stack(negative)

        return anchor, positive, negative

    def __len__(self):
        return self.len


class TriNet(nn.Module):
    def __init__(self, pre_model):
        super(TriNet, self).__init__()
        self.feat_layer = nn.Sequential(*list(pre_model.children())[:-1])
        self.clf_layer = nn.Sequential(list(pre_model.children())[-1])
        self.fc = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 512))
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_anc, input_poses, input_neges):
        out_anc = self.feat_layer(input_anc)
        out_anc = out_anc.view(out_anc.size(0), -1)
        out_anc = self.fc(out_anc)
        # out_anc = self.sigmoid(out_anc)
        out_poses = []
        out_neges = []
        for i in range(input_poses.shape[0]):
            out_pos = self.feat_layer(input_poses[i])
            out_pos = out_pos.view(out_pos.size(0), -1)
            out_pos = self.fc(out_pos)
            # out_pos = self.sigmoid(out_pos)
            out_poses.append(out_pos)
        for i in range(input_neges.shape[0]):
            out_neg = self.feat_layer(input_neges[i])
            out_neg = out_neg.view(out_neg.size(0), -1)
            out_neg = self.fc(out_neg)
            # out_neg = self.sigmoid(out_neg)
            out_neges.append(out_neg)
        out_poses = torch.stack(out_poses)
        out_neges = torch.stack(out_neges)

        return out_anc, out_poses, out_neges


# 自定义 cosine distance triplet loss
class cosine_triplet_loss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anc, pos, neg):
        loss = 0
        d_p = 1 - torch.cosine_similarity(anc, pos)
        d_n = 1 - torch.cosine_similarity(anc, neg)
        div = d_p - d_n + self.margin
        new_div = div.reshape(div.shape[0], -1)
        loss = torch.max(torch.cat((new_div, torch.zeros_like(new_div)), dim=1), dim=1)[0]
        return torch.mean(loss)


# 自定义arc cosine distance triplet loss
class arccosine_triplet_loss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anc, pos, neg):
        loss = 0
        d_p = torch.acos(torch.cosine_similarity(anc, pos))
        d_n = torch.acos(torch.cosine_similarity(anc, neg))
        div = d_p - d_n + self.margin
        new_div = div.reshape(div.shape[0], -1)
        loss = torch.max(torch.cat((new_div, torch.zeros_like(new_div)), dim=1), dim=1)[0]
        return torch.mean(loss)


# 新 pos 和 neg 都为 list
class multi_cosine_triplet_loss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anc, poses, neges):
        loss = 0
        d_p = 1 - torch.min(torch.cosine_similarity(anc.unsqueeze(1), poses, dim=2), dim=1).values
        d_n = 1 - torch.max(torch.cosine_similarity(anc.unsqueeze(1), poses, dim=2), dim=1).values
        div = d_p - d_n + self.margin
        new_div = div.reshape(div.shape[0], -1)
        loss = torch.max(torch.cat((new_div, torch.zeros_like(new_div)), dim=1), dim=1)[0]
        return torch.mean(loss)


# 新 pos 和 neg 都为 list
class multi_arccosine_triplet_loss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anc, poses, neges):
        loss = 0
        # d_p = torch.acos(torch.cosine_similarity(anc,pos[pos_pointer]))
        c_p = torch.cosine_similarity(anc.unsqueeze(1), poses, dim=2)
        c_p = torch.where(c_p > 1.0, torch.ones_like(c_p), c_p)
        c_p = torch.where(c_p < -1.0, -torch.ones_like(c_p), c_p)
        d_p = torch.max(torch.acos(c_p), dim=1).values
        # d_n = torch.acos(torch.cosine_similarity(anc,neg))
        c_n = torch.cosine_similarity(anc.unsqueeze(1), neges, dim=2)
        c_n = torch.where(c_n > 1.0, torch.ones_like(c_n), c_n)
        c_n = torch.where(c_n < -1.0, -torch.ones_like(c_n), c_n)
        d_n = torch.min(torch.acos(c_n), dim=1).values
        div = d_p - d_n + self.margin
        new_div = div.reshape(div.shape[0], -1)
        loss = torch.max(torch.cat((new_div, torch.zeros_like(new_div)), dim=1), dim=1)[0]
        return torch.mean(loss)


# add soft_margin
class multi_arccosine_triplet_loss_soft_margin_easy(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anc, poses, neges):
        loss = 0
        # d_p = torch.acos(torch.cosine_similarity(anc,pos[pos_pointer]))
        c_p = torch.cosine_similarity(anc.unsqueeze(1), poses, dim=2)
        c_p = torch.where(c_p > 1.0, torch.ones_like(c_p), c_p)
        c_p = torch.where(c_p < -1.0, -torch.ones_like(c_p), c_p)
        # easy
        d_p = torch.min(torch.acos(c_p), dim=1).values
        # semi-hard
        # d_p = torch.median(torch.acos(c_p), dim=1).values
        # hard
        # d_p = torch.max(torch.acos(c_p), dim=1).values
        # d_n = torch.acos(torch.cosine_similarity(anc,neg))
        c_n = torch.cosine_similarity(anc.unsqueeze(1), neges, dim=2)
        c_n = torch.where(c_n > 1.0, torch.ones_like(c_n), c_n)
        c_n = torch.where(c_n < -1.0, -torch.ones_like(c_n), c_n)
        # easy
        d_n = torch.max(torch.acos(c_n), dim=1).values
        # semi-hard
        # d_n = torch.median(torch.acos(c_n), dim=1).values
        # hard
        # d_n = torch.min(torch.acos(c_n), dim=1).values
        div = d_p - d_n + self.margin
        new_div = div.reshape(div.shape[0], -1)
        loss = torch.max(torch.cat((new_div, torch.zeros_like(new_div)), dim=1), dim=1)[0]
        loss = torch.log(1 + torch.exp(loss))
        return torch.mean(loss)

class multi_arccosine_triplet_loss_soft_margin(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anc, poses, neges):
        loss = 0
        # d_p = torch.acos(torch.cosine_similarity(anc,pos[pos_pointer]))
        c_p = torch.cosine_similarity(anc.unsqueeze(1), poses, dim=2)
        c_p = torch.where(c_p > 1.0, torch.ones_like(c_p), c_p)
        c_p = torch.where(c_p < -1.0, -torch.ones_like(c_p), c_p)
        # semi-hard
        d_p = torch.median(torch.acos(c_p), dim=1).values
        # hard
        # d_p = torch.max(torch.acos(c_p), dim=1).values
        # d_n = torch.acos(torch.cosine_similarity(anc,neg))
        c_n = torch.cosine_similarity(anc.unsqueeze(1), neges, dim=2)
        c_n = torch.where(c_n > 1.0, torch.ones_like(c_n), c_n)
        c_n = torch.where(c_n < -1.0, -torch.ones_like(c_n), c_n)
        # semi-hard
        d_n = torch.median(torch.acos(c_n), dim=1).values
        # hard
        # d_n = torch.min(torch.acos(c_n), dim=1).values
        div = d_p - d_n + self.margin
        new_div = div.reshape(div.shape[0], -1)
        loss = torch.max(torch.cat((new_div, torch.zeros_like(new_div)), dim=1), dim=1)[0]
        loss = torch.log(1 + torch.exp(loss))
        return torch.mean(loss)

class multi_arccosine_triplet_loss_soft_margin_hard(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anc, poses, neges):
        loss = 0
        # d_p = torch.acos(torch.cosine_similarity(anc,pos[pos_pointer]))
        c_p = torch.cosine_similarity(anc.unsqueeze(1), poses, dim=2)
        c_p = torch.where(c_p > 1.0, torch.ones_like(c_p), c_p)
        c_p = torch.where(c_p < -1.0, -torch.ones_like(c_p), c_p)
        # semi-hard
        # d_p = torch.median(torch.acos(c_p), dim=1).values
        # hard
        d_p = torch.max(torch.acos(c_p), dim=1).values
        # d_n = torch.acos(torch.cosine_similarity(anc,neg))
        c_n = torch.cosine_similarity(anc.unsqueeze(1), neges, dim=2)
        c_n = torch.where(c_n > 1.0, torch.ones_like(c_n), c_n)
        c_n = torch.where(c_n < -1.0, -torch.ones_like(c_n), c_n)
        # semi-hard
        # d_n = torch.median(torch.acos(c_n), dim=1).values
        # hard
        d_n = torch.min(torch.acos(c_n), dim=1).values
        div = d_p - d_n + self.margin
        new_div = div.reshape(div.shape[0], -1)
        loss = torch.max(torch.cat((new_div, torch.zeros_like(new_div)), dim=1), dim=1)[0]
        loss = torch.log(1 + torch.exp(loss))
        return torch.mean(loss)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, trained = False,delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.trained = trained
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if trained:
            torch.save(model.state_dict(), 'models/hard_retrive_ckpt.pt')
        else:
            torch.save(model.state_dict(), 'models/retrive_ckpt.pt')  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


if __name__ == '__main__':

    # 设置脚本参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", action="store_true", help="train with semi or retrain with hard")
    args = parser.parse_args()
    trained = args.trained
    logger.info('trained: {}'.format(trained))

    # trained = True

    with open('data/labels_count.pkl', 'rb') as fp:
        labels_count = pickle.load(fp)
    with open('data/label_mapping.pkl', 'rb') as fp:
        label_mapping = pickle.load(fp)
    pre_model = SeResNet([3, 4, 6, 3], sum(labels_count), 4)
    checkpoint = torch.load(
        '/home/administrator/lyh/transfer_learning/MyDataset/data/1_pro_big_type_n_pro_small_type_n_colsystem_ckpt.pt')
    pre_model.load_state_dict({k.replace('module.', ''): v for k, v in dict(checkpoint).items()})

    # 固定特征提取参数不更新
    for p in pre_model.parameters():
        p.requires_grad = False

    model = TriNet(pre_model)

    # model = nn.DataParallel(model)
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    with open('data/train_data.json', 'r') as f:
        train_data = json.load(f)
    with open('data/valid_data.json', 'r') as f:
        valid_data = json.load(f)
    with open('data/test_data.json', 'r') as f:
        test_data = json.load(f)

    train_data = RetrivalDataset(train_data, transform=None)
    BATCH_SIZE = 16
    # train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,sampler=DistributedSampler(train_data))
    train_loader = DataLoaderX(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    valid_data = RetrivalDataset(valid_data, transform=None)
    BATCH_SIZE = 8
    # valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    # valid_loader=DataLoader(valid_data,batch_size=BATCH_SIZE,sampler=DistributedSampler(valid_data))
    valid_loader = DataLoaderX(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    test_data = RetrivalDataset(test_data, transform=None)
    BATCH_SIZE = 8
    # test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,sampler=DistributedSampler(test_data))
    test_loader = DataLoaderX(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 非首次训练
    if trained:
        print('hard stage')
        model.load_state_dict(torch.load('models/retrive_ckpt.pt'))
        criterion = multi_arccosine_triplet_loss_soft_margin_hard(margin=0.3)
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    else:
        # criterion = cosine_triplet_loss(margin=0.3)
        print('easy stage')
        criterion = multi_arccosine_triplet_loss_soft_margin_easy(margin=0.3)
        # criterion = multi_arccosine_triplet_loss_soft_margin(margin=0.3)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    patience = 10
    early_stopping = EarlyStopping(patience, verbose=True, trained=trained)
    epochs = 50

    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0.

        for step, (anchor, positive, negative) in enumerate(tqdm(train_loader)):
            anchor, positive, negative = Variable(anchor).cuda(), \
                                         Variable(positive).cuda(), Variable(negative).cuda()
            anc, pos, neg = model(anchor, positive, negative)
            loss = criterion(anc, pos, neg)
            train_loss += loss.item() * 16  # * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        eval_loss = 0.

        for step, (anchor, positive, negative) in enumerate(tqdm(valid_loader)):
            anchor, positive, negative = Variable(anchor).cuda(), \
                                         Variable(positive).cuda(), Variable(negative).cuda()
            anc, pos, neg = model(anchor, positive, negative)
            loss = criterion(anc, pos, neg)
            eval_loss += loss.item() * 8

        tqdm.write('epoch: {}'.format(epoch))
        tqdm.write('Train Loss: {:.6f}'.format(train_loss / (len(train_data))))
        tqdm.write('Valid Loss: {:.6f}'.format(eval_loss / (len(valid_data))))

        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            tqdm.write("Early stopping")
            # 结束模型训练
            break
        if epoch == 20:
            if trained:
                model.load_state_dict(torch.load('models/hard_retrive_ckpt.pt'))
            else:
                model.load_state_dict(torch.load('models/retrive_ckpt.pt'))
            optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
        # break

    # 获得 early stopping 时的模型参数
    if trained:
        model.load_state_dict(torch.load('model/hard_retrive_ckpt.pt'))
    else:
        model.load_state_dict(torch.load('models/retrive_ckpt.pt'))

    # Test
    model.eval()
    eval_loss = 0.

    for step, (anchor, positive, negative) in enumerate(tqdm(test_loader)):
        anchor, positive, negative = Variable(anchor).cuda(), \
                                     Variable(positive).cuda(), Variable(negative).cuda()
        anc, pos, neg = model(anchor, positive, negative)
        loss = criterion(anc, pos, neg)
        eval_loss += loss.item() * 8

    tqdm.write('Test Loss: {:.6f}'.format(eval_loss / (len(test_data))))

    # semi-hard
    if trained:
        exit(0)
    print('semi-hard stage')
    criterion = multi_arccosine_triplet_loss_soft_margin(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    patience = 10
    early_stopping = EarlyStopping(patience, verbose=True, trained=trained)
    epochs = 50

    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0.

        for step, (anchor, positive, negative) in enumerate(tqdm(train_loader)):
            anchor, positive, negative = Variable(anchor).cuda(), \
                                         Variable(positive).cuda(), Variable(negative).cuda()
            anc, pos, neg = model(anchor, positive, negative)
            loss = criterion(anc, pos, neg)
            train_loss += loss.item() * 16  # * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        eval_loss = 0.

        for step, (anchor, positive, negative) in enumerate(tqdm(valid_loader)):
            anchor, positive, negative = Variable(anchor).cuda(), \
                                         Variable(positive).cuda(), Variable(negative).cuda()
            anc, pos, neg = model(anchor, positive, negative)
            loss = criterion(anc, pos, neg)
            eval_loss += loss.item() * 8

        tqdm.write('epoch: {}'.format(epoch))
        tqdm.write('Train Loss: {:.6f}'.format(train_loss / (len(train_data))))
        tqdm.write('Valid Loss: {:.6f}'.format(eval_loss / (len(valid_data))))

        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            tqdm.write("Early stopping")
            # 结束模型训练
            break
        if epoch == 20:
            if trained:
                model.load_state_dict(torch.load('models/hard_retrive_ckpt.pt'))
            else:
                model.load_state_dict(torch.load('models/retrive_ckpt.pt'))
            optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
        # break

    # 获得 early stopping 时的模型参数
    if trained:
        model.load_state_dict(torch.load('model/hard_retrive_ckpt.pt'))
    else:
        model.load_state_dict(torch.load('models/retrive_ckpt.pt'))

    # Test
    model.eval()
    eval_loss = 0.

    for step, (anchor, positive, negative) in enumerate(tqdm(test_loader)):
        anchor, positive, negative = Variable(anchor).cuda(), \
                                     Variable(positive).cuda(), Variable(negative).cuda()
        anc, pos, neg = model(anchor, positive, negative)
        loss = criterion(anc, pos, neg)
        eval_loss += loss.item() * 8

    tqdm.write('Test Loss: {:.6f}'.format(eval_loss / (len(test_data))))



