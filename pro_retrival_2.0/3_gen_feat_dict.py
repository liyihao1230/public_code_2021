import sys
import os
import glob
import time

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

import argparse
import faiss
from faiss import normalize_L2

from tools.Resnet_Family import ResNet, ResNetv2, ResNeXt, SeResNet, SeResNeXt
from unet_segmentation.prediction.display import get_masks

from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3" # 0,
torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23497', rank=0, world_size=1)
from torch.utils.data.distributed import DistributedSampler

local_rank = torch.distributed.get_rank()

from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

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

# 生成检索使用特征向量网络
class GenNet(nn.Module):
    def __init__(self,pre_model):
        super(GenNet, self).__init__()
        self.feat_layer = nn.Sequential(*list(pre_model.children())[:-2])
        self.clf_layer = nn.Sequential(list(pre_model.children())[-2])
        self.predict_layer = nn.Sequential(list(pre_model.children())[-1])
    def forward(self,input1):
        out1 = self.feat_layer(input1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.predict_layer(out1)
        return out1

brand_list = ['JNBY','LESS','CROQUIS','tjnby']
# cate_list = ['dress','outwear','shorts','skirt','top','trousers']
cate_list = ['衬衣', '裤子', 'T恤', '夹克', '连衣裙', '腰裙', '毛衫', '羽绒服', '大衣', '风衣', '卫衣',
            '连体衣', '背心', '西服', '马甲', '皮衣皮草', '棉衣']
cate_list = ['可内可外', '下装', '外套', '通身']
# 标签名称 (与attr表中商品标签一致)
label_names = ['pro_big_type', 'pro_small_type', 'colsystem']

# 加载商品属性表
attr = pd.read_parquet('deploy_data/used_attr.parquet')
order_skc_df = pd.read_parquet('deploy_data/used_order_skc_df.parquet')
# pos_id == 2 为细节图
order_skc_df = order_skc_df[order_skc_df['pos_id'] == 1]
wx_spu_df = pd.read_parquet('deploy_data/used_wx_spu_df.parquet')

# 搞结果dict
seg_order2skc_file = dict(zip(order_skc_df['order_skc_file'],order_skc_df['skc_file']))
order_skc2brand = dict(zip(order_skc_df['order_skc_file'],order_skc_df['pro_brand']))
seg_wx2skc_file = dict(zip(wx_spu_df['wx_spu_file'],wx_spu_df['skc_file']))
wx_spu2brand = dict(zip(wx_spu_df['wx_spu_file'],wx_spu_df['pro_brand']))
skc_file2spu = dict(zip(attr['skc_file'],attr['m_product_id'].astype(str)))
skc_file2brand = dict(zip(attr['skc_file'],attr['pro_brand']))

if __name__ == '__main__':

    # 设置脚本参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--ishard", action="store_true", help="use semi-hard or hard")
    args = parser.parse_args()
    ishard = args.ishard
    logger.info('ishard: {}'.format(ishard))

    with open('data/labels_count.pkl','rb') as fp:
        labels_count = pickle.load(fp)
    with open('data/label_mapping.pkl','rb') as fp:
        label_mapping = pickle.load(fp)
    # 加载模型
    pre_model = SeResNet([3, 4, 6, 3], sum(labels_count), 4)
    checkpoint = torch.load(
        '/home/administrator/lyh/transfer_learning/MyDataset/data/1_pro_big_type_n_pro_small_type_n_colsystem_ckpt.pt')
    pre_model.load_state_dict({k.replace('module.', ''): v for k, v in dict(checkpoint).items()})
    # 固定特征提取参数不更新
    for p in pre_model.parameters():
        p.requires_grad = False
    pre_model = TriNet(pre_model)
    if ishard:
        checkpoint = torch.load('models/hard_retrive_ckpt.pt')
    else:
        checkpoint = torch.load('models/retrive_ckpt.pt')
    pre_model.load_state_dict({k.replace('module.', ''): v for k, v in dict(checkpoint).items()})

    # 加载gennet
    gen_model = GenNet(pre_model)
    gen_model.cuda()
    gen_model.eval()

    # 创建存放目录
    for brand in brand_list + ['all']:
        if not os.path.exists(os.getcwd()+'/deploy_data/'+brand+'/'):
            os.mkdir(os.getcwd()+'/deploy_data/'+brand+'/')
        for c in cate_list:
            if not os.path.exists(os.getcwd()+'/deploy_data/'+brand+'/'+c+'/'):
                os.mkdir(os.getcwd()+'/deploy_data/'+brand+'/'+c+'/')

    for brand in brand_list + ['all']:
        for pro_class in cate_list:
            feat_dict = {}
            for name in tqdm(glob.glob('/home/administrator/lyh/unet/pro_retrival_2.0/data/' \
                                       + brand + '/' + pro_class + '/' + '*.jpg')):
                pid = name.split('/')[-1]
                pid = pid.split('.')[0]
                if pid.split('_')[1] == '2':
                    continue
                # if pid.split('_')[1] == 'wx':
                #     continue
                # predict
                img = cv2.imread(name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.reshape((1, 224, 224, 3))
                img = np.transpose(img, (0, 3, 1, 2))
                img = torch.FloatTensor(img)
                img_feat = gen_model(img.cuda())
                # save feat into dict
                feat_dict[pid] = img_feat.data.cpu().numpy()[0].tolist()
            # 构建索引查找商品dict
            pids = list(feat_dict.keys())
            pro_mapping_dict = dict(zip([i for i in range(len(pids))], pids))
            # faiss part
            if len(feat_dict) == 0:
                logger.info('empty faiss at {}/{}/'.format(brand,pro_class))
                continue
            elif len(feat_dict) > 1248:  # with IVF
                d = len(list(feat_dict.values())[0])  # dimension
                nb = len(feat_dict)  # dataset size
                # quantizer = faiss.IndexFlatL2(d)   # build the index
                quantizer = faiss.IndexFlatIP(d)  # build the index
                nlist = 32  # 聚类中心的个数
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                xb = np.asarray(list(feat_dict.values())).astype('float32')  # index 向量
                normalize_L2(xb)
                assert not index.is_trained
                index.train(xb)
                index.add(xb)  # add may be a bit slower as well
                assert index.is_trained
            elif len(feat_dict) > 0:
                d = len(list(feat_dict.values())[0])
                nb = len(feat_dict)
                index = faiss.IndexFlatIP(d)
                xb =np.asarray(list(feat_dict.values())).astype('float32')
                normalize_L2(xb)
                index.add(xb)
            # 实际上不会执行
            else:  # with IVFPQ
                d = len(list(feat_dict.values())[0])  # dimension
                nb = len(feat_dict)  # dataset size
                # quantizer = faiss.IndexFlatL2(d)   # build the index
                quantizer = faiss.IndexFlatIP(d)  # build the index
                nlist = 32  # 聚类中心的个数
                m = 8  # number of subvector
                index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8,
                                         faiss.METRIC_INNER_PRODUCT)  # 8 specifies that each sub-vector is encoded as 8 bits
                xb = np.asarray(list(feat_dict.values())).astype('float32')  # index 向量
                normalize_L2(xb)
                assert not index.is_trained
                index.train(xb)
                index.add(xb)  # add may be a bit slower as well
                assert index.is_trained
            # 保存index
            faiss.write_index(index, './deploy_data/' + brand + '/' + pro_class + '/' + 'retrival_index.index')
            # 保存 pro_mapping_dict
            with open('deploy_data/' + brand + '/' + pro_class + '/' + 'pro_mapping_dict.json',
                      'w') as json_file:
                json.dump(pro_mapping_dict, json_file, ensure_ascii=False)

    pass