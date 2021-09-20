import sys
import os
import glob
import time
import re
import logging
import cv2
import pickle
import joblib
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
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23457', rank=0, world_size=1)
from torch.utils.data.distributed import DistributedSampler

local_rank = torch.distributed.get_rank()

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# img_path = '/home/administrator/Dataset/SKC/'
# img_path = '/home/administrator/Dataset/Resize_SKC/'
img_path = '/home/administrator/Dataset/NoPad_Resize_SKC/'

# 生成检索使用特征向量网络
class GenNet(nn.Module):
    def __init__(self,pre_model):
        super(GenNet, self).__init__()
        self.feat_layer = nn.Sequential(*list(pre_model.children())[:-1])
        # self.predict_layer = nn.Sequential(list(pre_model.children())[-1])
    def forward(self,input1):
        out1 = self.feat_layer(input1)
        out1 = out1.view(out1.size(0), -1)
        # out1 = self.predict_layer(out1)
        return out1

if __name__ == '__main__':

    # 参数定义
    # 标签名称 (与attr表中商品标签一致)
    label_names = ['pro_big_type', 'pro_small_type','colsystem']
    model_name = 'SeResNet'
    if_pretrain = False
    pretrain_label_name = 'pro_big_type'

    attr = pd.read_parquet('data/used_attr.parquet')
    item2labels = {}
    labels_count = []
    labels_total = 0
    labels_dict = {}
    for label_name in label_names:
        # 标签类别数
        label_count = attr[label_name].nunique()
        labels_count.append(label_count)
        labels_total += label_count
        # 类别索引cate2index
        label_dict = dict(zip(list(attr[label_name].unique()), [i for i in range(label_count)]))
        labels_dict[label_name] = label_dict.copy()
        logger.info('label count: {}'.format(label_count))
        logger.info('cate2index: {}'.format(label_dict))
        # 缺失值默认填充
        if attr[label_name].dtype == 'O':
            attr[label_name] = attr[label_name].fillna('unknown')
        elif attr[label_name].dtype == int:
            attr[label_name] = attr[label_name].fillna(9999)
        elif attr[label_name].dtype == float:
            attr[label_name] = attr[label_name].fillna(0.0)
        item2label = dict(zip(attr['skc_file'], attr[label_name]))
        item2labels[label_name] = item2label.copy()

    if model_name == 'SeResNet':
        pre_model = SeResNet([3, 4, 6, 3], labels_total, 4)
    # elif model_name == '':
    else:
        pre_model = SeResNet([3, 4, 6, 3], labels_total, 4)
    checkpoint = torch.load('data/1_' + '_n_'.join(label_names) + '_ckpt.pt')
    pre_model.load_state_dict({k.replace('module.', ''): v for k, v in dict(checkpoint).items()})

    gen_model = GenNet(pre_model)
    gen_model.cuda()
    gen_model.eval()

    time_start = time.time()

    for brand in ['JNBY', 'LESS', 'CROQUIS', 'tjnby']:
        img_feat_dict = {}
        logger.info('start calculate {} img_feat_dict'.format(brand))
        for name in tqdm(glob.glob('/home/administrator/Dataset/NoPad_Resize_SKC/' + brand + '/' + '*.jpg')):
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.reshape((1, 224, 224, 3))
            img = np.transpose(img, (0, 3, 1, 2))
            img = torch.FloatTensor(img)

            name = name.split('/')[-1]
            name = name.split('.')[0]

            # predict
            img_feat = gen_model(img.cuda())
            # save feat into dict
            img_feat_dict[name] = img_feat.data.cpu().numpy()[0].tolist()

        # joblib替换pickle
        with open('./data/' + brand + '_skc_img_feat_dict.pkl', 'wb') as fo:
            joblib.dump(img_feat_dict, fo)
        # with open('./data/'+brand+'_'+label_name+'_'+'img_feat_dict.pkl', 'rb') as fo:
        #     img_feat_dict = joblib.load(fo)

        logger.info('finished {} img_feat_dict'.format(brand))

        pro_list = list(img_feat_dict.keys())
        logger.info('start calculate {} sim_dict'.format(brand))
        # 计算相似度用list保存
        sim_matrix = cosine_similarity(np.asarray(list(img_feat_dict.values()))).tolist()
        sim_dict = {}
        for i in trange(len(pro_list)):
            temp_dict = {}
            for j in range(len(pro_list)):
                temp_dict[pro_list[j]] = sim_matrix[i][j]
            sim_dict[pro_list[i]] = temp_dict

        logger.info('finished {} img_sim_dict'.format(brand))

        # joblib替换pickle
        with open('./data/' + brand + '_skc_img_sim_dict.pkl', 'wb') as fo:
            joblib.dump(sim_dict, fo)
        # with open('./data/' + brand + '_' + label_name + '_' + 'img_sim_dict.pkl', 'rb') as fo:
        #     sim_dict = joblib.load(fo)

    time_end = time.time()
    logger.info('finished get image feature dict and sim dict:time cost', time_end - time_start, 's')

    pass

