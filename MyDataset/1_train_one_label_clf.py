import sys
import os
import glob
import time
import re
import cv2
import pickle
import json
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
# from pytorchtools import EarlyStopping
from sklearn import preprocessing

from tools.Resnet_Family import ResNet, ResNetv2, ResNeXt, SeResNet, SeResNeXt

from tqdm import tqdm, trange
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)



from torch.utils.data.distributed import DistributedSampler

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23455', rank=0, world_size=1)
from torch.utils.data.distributed import DistributedSampler

local_rank = torch.distributed.get_rank()

# img_path = '/home/administrator/Dataset/SKC/'
# img_path = '/home/administrator/Dataset/Resize_SKC/'
# img_path = '/home/administrator/Dataset/NoPad_Resize_SKC/'


class MyDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, labels, imgs, transform=None):  # labels, imgs 为 list
        self.labels = labels
        self.imgs = imgs
        #         self.x1_list = x1_list
        #         self.y1_list = y1_list
        #         self.x2_list = x2_list
        #         self.y2_list = y2_list
        self.len = len(labels)
        self.transform = transform

    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img[self.y1_list[index]:self.y2_list[index],self.x1_list[index]:self.x2_list[index]]
        try:
            img = cv2.resize(img, (224, 224))
        except:
            print(self.imgs[index])
        img = np.transpose(img, (2, 0, 1))
        img = torch.FloatTensor(img)
        label = self.labels[index]
        label = np.array(label)
        # label = onehot.transform(np.array(label).reshape(-1,1)).toarray()[0]
        label = torch.from_numpy(label)
        return img, label

    def __len__(self):
        return self.len


from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, label_name, patience=7, verbose=False, delta=0):
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
        self.label_name = label_name
        self.verbose = verbose
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
        torch.save(model.state_dict(), 'data/1_'+self.label_name+'_ckpt.pt')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

if __name__ == '__main__':
    # 参数定义
    # 标签名称 (与attr表中商品标签一致)
    label_name = 'pro_big_type'
    model_name = 'SeResNet'
    if_pretrain = False
    pretrain_label_name = 'pro_big_type'

    # skc单品图
    skc_img_path = '/home/administrator/Dataset/NoPad_Resize_SKC/'
    with open('data/skc_train.json', 'r') as json_file:
        skc_train = json.load(json_file)
    with open('data/skc_valid.json', 'r') as json_file:
        skc_valid = json.load(json_file)
    with open('data/skc_test.json', 'r') as json_file:
        skc_test = json.load(json_file)
    attr = pd.read_parquet('data/used_attr.parquet')
    # 标签类别数
    label_count = attr[label_name].nunique()
    # 类别索引cate2index
    label_dict = dict(zip(list(attr[label_name].unique()), [i for i in range(label_count)]))
    logger.info('label count: {}'.format(label_count))
    logger.info('cate2index: {}'.format(label_dict))
    # 缺失值默认填充
    if attr[label_name].dtype == 'O':
        attr[label_name] = attr[label_name].fillna('unknown')
    elif attr[label_name].dtype == int:
        attr[label_name] = attr[label_name].fillna(9999)
    elif attr[label_name].dtype == float:
        attr[label_name] = attr[label_name].fillna(0.0)
    # item2brand = dict(zip(attr['skc_file'], attr['pro_brand']))
    item2label = dict(zip(attr['skc_file'], attr[label_name]))

    train_img_list = list(skc_train.values())
    for i,train_img in enumerate(train_img_list):
        train_img_list[i] = skc_img_path + train_img
    train_label_list = list(map(lambda x: label_dict[item2label[x]], list(skc_train.keys())))
    valid_img_list = list(skc_valid.values())
    for i,valid_img in enumerate(valid_img_list):
        valid_img_list[i] = skc_img_path + valid_img
    valid_label_list = list(map(lambda x: label_dict[item2label[x]], list(skc_valid.keys())))
    test_img_list = list(skc_test.values())
    for i,test_img in enumerate(test_img_list):
        test_img_list[i] = skc_img_path + test_img
    test_label_list = list(map(lambda x: label_dict[item2label[x]], list(skc_test.keys())))

    # order_skc订货会skc图
    order_skc_img_path = '/home/administrator/Dataset/ORDER_SKC/nopad_resize_images/'
    with open('data/order_skc_train.json', 'r') as json_file:
        order_skc_train = json.load(json_file)
    with open('data/order_skc_valid.json', 'r') as json_file:
        order_skc_valid = json.load(json_file)
    with open('data/order_skc_test.json', 'r') as json_file:
        order_skc_test = json.load(json_file)
    order_skc_df = pd.read_parquet('data/used_order_skc_df.parquet')
    # 构造order_item2label
    order_item2label = {}
    skc2skc_file = dict(zip(attr['skc'],attr['skc_file']))
    # item2label为 skc_file -> label
    for key in order_skc_train.keys():
        order_item2label[key] = item2label[skc2skc_file[key.split('_')[0]]]
    for key in order_skc_valid.keys():
        order_item2label[key] = item2label[skc2skc_file[key.split('_')[0]]]
    for key in order_skc_test.keys():
        order_item2label[key] = item2label[skc2skc_file[key.split('_')[0]]]
    order_train_img_list = list(order_skc_train.values())
    for i, order_train_img in enumerate(order_train_img_list):
        order_train_img_list[i] = order_skc_img_path + order_train_img
    order_train_label_list = list(map(lambda x: label_dict[order_item2label[x]], list(order_skc_train.keys())))
    order_valid_img_list = list(order_skc_valid.values())
    for i, order_valid_img in enumerate(order_valid_img_list):
        order_valid_img_list[i] = order_skc_img_path + order_valid_img
    order_valid_label_list = list(map(lambda x: label_dict[order_item2label[x]], list(order_skc_valid.keys())))
    order_test_img_list = list(order_skc_test.values())
    for i, order_test_img in enumerate(order_test_img_list):
        order_test_img_list[i] = order_skc_img_path + order_test_img
    order_test_label_list = list(map(lambda x: label_dict[order_item2label[x]], list(order_skc_test.keys())))

    # skc 和 order_skc 拼接在一起
    train_img_list = train_img_list + order_train_img_list
    train_label_list = train_label_list + order_train_label_list
    valid_img_list = valid_img_list + order_valid_img_list
    valid_label_list = valid_label_list + order_valid_label_list
    test_img_list = test_img_list + order_test_img_list
    test_label_list = test_label_list + order_test_label_list


    # wx_spu图
    wx_spu_img_path = '/home/administrator/Dataset/WX_SPU/'
    with open('data/wx_spu_train.json', 'r') as json_file:
        wx_spu_train = json.load(json_file)
    with open('data/wx_spu_valid.json', 'r') as json_file:
        wx_spu_valid = json.load(json_file)
    with open('data/wx_spu_test.json', 'r') as json_file:
        wx_spu_test = json.load(json_file)
    wx_spu_df = pd.read_parquet('data/used_wx_spu_df.parquet')
    # 构造wx_spu_item2label
    wx_spu_item2label = {}
    wx_spu2skc_file = dict(zip(wx_spu_df['wx_spu_file'], wx_spu_df['skc_file']))
    # item2label为 skc_file -> label
    for key in wx_spu_train.keys():
        wx_spu_item2label[key] = item2label[wx_spu2skc_file[key]]
    for key in wx_spu_valid.keys():
        wx_spu_item2label[key] = item2label[wx_spu2skc_file[key]]
    for key in wx_spu_test.keys():
        wx_spu_item2label[key] = item2label[wx_spu2skc_file[key]]
    wx_spu_train_img_list = list(wx_spu_train.values())
    for i, wx_spu_train_img in enumerate(wx_spu_train_img_list):
        wx_spu_train_img_list[i] = wx_spu_img_path + wx_spu_train_img
    wx_spu_train_label_list = list(map(lambda x: label_dict[wx_spu_item2label[x]], list(wx_spu_train.keys())))
    wx_spu_valid_img_list = list(wx_spu_valid.values())
    for i, wx_spu_valid_img in enumerate(wx_spu_valid_img_list):
        wx_spu_valid_img_list[i] = wx_spu_img_path + wx_spu_valid_img
    wx_spu_valid_label_list = list(map(lambda x: label_dict[wx_spu_item2label[x]], list(wx_spu_valid.keys())))
    wx_spu_test_img_list = list(wx_spu_test.values())
    for i, wx_spu_test_img in enumerate(wx_spu_test_img_list):
        wx_spu_test_img_list[i] = wx_spu_img_path + wx_spu_test_img
    wx_spu_test_label_list = list(map(lambda x: label_dict[wx_spu_item2label[x]], list(wx_spu_test.keys())))

    # skc 和 wx_spu 拼接在一起
    train_img_list = train_img_list + wx_spu_train_img_list
    train_label_list = train_label_list + wx_spu_train_label_list
    valid_img_list = valid_img_list + wx_spu_valid_img_list
    valid_label_list = valid_label_list + wx_spu_valid_label_list
    test_img_list = test_img_list + wx_spu_test_img_list
    test_label_list = test_label_list + wx_spu_test_label_list


    # 构造数据集
    train_data = MyDataset(train_label_list, train_img_list, transform=None)
    train_batch_size = 64
    # train_loader=DataLoader(train_data,batch_size=valid_batch_size,shuffle=True)
    # train_loader=DataLoader(train_data,batch_size=valid_batch_size,sampler=DistributedSampler(train_data))
    train_loader = DataLoaderX(train_data, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    valid_data = MyDataset(valid_label_list, valid_img_list, transform=None)
    valid_batch_size = 16
    # valid_loader=DataLoader(valid_data,batch_size=valid_batch_size,shuffle=True)
    # valid_loader=DataLoader(valid_data,batch_size=valid_batch_size,sampler=DistributedSampler(valid_data))
    valid_loader = DataLoaderX(valid_data, batch_size=valid_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_data = MyDataset(test_label_list, test_img_list, transform=None)
    test_batch_size = 16
    # test_loader=DataLoader(test_data,batch_size=test_batch_size,shuffle=True)
    # test_loader=DataLoader(test_data,batch_size=test_batch_size,sampler=DistributedSampler(test_data))
    test_loader = DataLoaderX(test_data, batch_size=test_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if model_name == 'SeResNet':
        model = SeResNet([3, 4, 6, 3], label_count, 4)
    # elif model_name == '':

    # 是否加载预训练
    if if_pretrain:
        checkpoint = torch.load('data/1_' + pretrain_label_name + '_ckpt.pt')
        model_state = model.state_dict()
        load_state = {k.replace('module.', ''): v for k, v in list(dict(checkpoint).items())[:-2]}
        model_state.update(load_state)
        model.load_state_dict(model_state)

    # GPU+加速器训练
    # model = nn.DataParallel(model)
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)
    # optimizer=torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9,weight_decay = 0.0001)

    epoch_total = 200
    patience = 20
    early_stopping = EarlyStopping(label_name, patience, verbose=True)

    for epoch in range(epoch_total):

        # training
        train_loss = 0.
        train_acc = 0.
        for step, (image, label) in enumerate(tqdm(train_loader)):
            image, label = Variable(image).cuda(), Variable(label).cuda()
            out = model(image)
            loss = criterion(out, label)
            train_loss += loss.item() * train_batch_size
            pred = torch.max(out, 1)[1]
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # evaluation
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for step, (image, label) in enumerate(tqdm(valid_loader)):
            image, label = Variable(image).cuda(), Variable(label).cuda()
            out = model(image)
            loss = criterion(out, label)

            eval_loss += loss.item() * valid_batch_size
            pred = torch.max(out, 1)[1]
            eval_correct = (pred == label).sum()
            eval_acc += eval_correct.item()
        tqdm.write('epoch: {}'.format(epoch))
        tqdm.write(
            'Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))
        tqdm.write(
            'Valid Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(valid_data)), eval_acc / (len(valid_data))))
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            tqdm.write("Early stopping")
            # 结束模型训练
            break
        if epoch == 10:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0001)
            model.load_state_dict(torch.load('data/1_' + label_name + '_ckpt.pt'))
        # if epoch == 6:
        #     break

    # 获得 early stopping 时的模型参数
    model.load_state_dict(torch.load('data/1_' + label_name + '_ckpt.pt'))

    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for step, (image, label) in enumerate(tqdm(test_loader)):
        image, label = Variable(image).cuda(), Variable(label).cuda()
        out = model(image)
        loss = criterion(out, label)

        eval_loss += loss.item() * test_batch_size
        pred = torch.max(out, 1)[1]
        eval_correct = (pred == label).sum()
        eval_acc += eval_correct.item()
    tqdm.write('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))

    pass