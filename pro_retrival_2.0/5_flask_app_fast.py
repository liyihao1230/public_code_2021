import sys
import os
import glob
import time
import datetime

import base64
from io import BytesIO
import re
import json
import random
import logging
import pickle

import cv2
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
# from imgaug import augmenters as iaa
from torch.autograd import Variable
# from pytorchtools import EarlyStopping
from sklearn import preprocessing

import argparse
import faiss
from faiss import normalize_L2

from flask import Flask, request
from flask_apscheduler import APScheduler
from unet_segmentation.prediction.display import get_masks

from tools.Resnet_Family import ResNet, ResNetv2, ResNeXt, SeResNet, SeResNeXt

from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler

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
class SearchNet(nn.Module):
    def __init__(self,pre_model,baseline):
        super(SearchNet, self).__init__()
        self.feat_layer = nn.Sequential(*list(pre_model.children())[:-2])
        self.clf_layer = nn.Sequential(list(pre_model.children())[-2])
        self.predict_layer = nn.Sequential(list(pre_model.children())[-1])
        self.baseline = baseline
    def forward(self,input1):
        out1 = self.feat_layer(input1)
        out1 = out1.view(out1.size(0), -1)
        out_clf = self.clf_layer(out1)
        if baseline:
            out_pred = out1
        else:
            out_pred = self.predict_layer(out1)
        return out_pred, out_clf

def seg_img_by_unet(mask,img):
    pro_class = mask.class_name
    pro_mask = mask.binary_mask
    y,x = pro_mask.shape
    # 水平映射垂直映射找到box
    # y_start, y_end
    y_start = None
    y_end = None
    for i in range(y):
        if pro_mask[i,:].sum() > 0:
            y_start = i
            break
    for i in range(y-1,-1,-1):
        if pro_mask[i,:].sum() > 0:
            y_end = i
            break
    # x_start, x_end
    x_start = None
    x_end = None
    for i in range(x):
        if pro_mask[:,i].sum() > 0:
            x_start = i
            break
    for i in range(x-1,-1,-1):
        if pro_mask[:,i].sum() > 0:
            x_end = i+1
            break
    sub_img = img[y_start:y_end,x_start:x_end]
    sub_img = cv2.resize(sub_img,(224,224))
    return sub_img, pro_class

def seg_img_by_unet_pad(mask,img,pad_rate=0.1):
    pro_class = mask.class_name
    pro_mask = mask.binary_mask
    y,x = pro_mask.shape
    # 水平映射垂直映射找到box
    # y_start, y_end
    y_start = None
    y_end = None
    for i in range(y):
        if pro_mask[i,:].sum() > 0:
            y_start = i
            break
    for i in range(y-1,-1,-1):
        if pro_mask[i,:].sum() > 0:
            y_end = i
            break
    # x_start, x_end
    x_start = None
    x_end = None
    for i in range(x):
        if pro_mask[:,i].sum() > 0:
            x_start = i
            break
    for i in range(x-1,-1,-1):
        if pro_mask[:,i].sum() > 0:
            x_end = i+1
            break
    # 保留边缘
    # seg_ratio = (y_end-y_start)/(x_end - x_start)
    # x_expand = int((x_end - x_start)*pad_rate/2)
    # y_expand = int(x_expand * seg_ratio)
    # 保留边缘 v2
    seg_ratio = (x_end - x_start)/(y_end - y_start)
    if seg_ratio < 1:
        y_expand = int((y_end - y_start)*pad_rate/2)
        x_expand = int((y_end - y_start + y_expand * 2 - x_end + x_start)/2)
    else:
        x_expand = int((x_end - x_start)*pad_rate/2)
        y_expand = int((x_end - x_start + x_expand * 2 - y_end + y_start)/2)
    new_y_start = y_start - y_expand
    new_y_end = y_end + y_expand
    new_x_start = x_start - x_expand
    new_x_end = x_end + x_expand
    if new_y_start < 0:
        new_y_start = 0
    if new_x_start < 0:
        new_x_start = 0
    if new_y_end > y:
        new_y_end = y
    if new_x_end > x:
        new_x_end = x
    # logger.info(new_y_start,new_y_end,y)
    # logger.info(new_x_start,new_x_end,x)
    sub_img = img[new_y_start:new_y_end,new_x_start:new_x_end]
    sub_img = cv2.resize(sub_img,(224,224))
    return sub_img, pro_class

brand_list = ['JNBY','LESS','CROQUIS','tjnby']
# cate_list = ['dress','outwear','shorts','skirt','top','trousers']
cate_list = ['衬衣', '裤子', 'T恤', '夹克', '连衣裙', '腰裙', '毛衫', '羽绒服', '大衣', '风衣', '卫衣',
            '连体衣', '背心', '西服', '马甲', '皮衣皮草', '棉衣']
cate_list = ['可内可外', '下装', '外套', '通身']
# 标签名称 (与attr表中商品标签一致)
label_names = ['pro_big_type', 'pro_small_type', 'colsystem']
label_name = 'pro_big_type'
label_index = label_names.index(label_name)
# -- whether use baseine --
baseline = False
if baseline:
    post_fix = '_baseline'
else:
    post_fix = ''
# -- whether use ishard --
ishard = True

# 加载商品属性表
attr = pd.read_parquet('deploy_data/used_attr.parquet')
order_skc_df = pd.read_parquet('deploy_data/used_order_skc_df.parquet')
wx_spu_df = pd.read_parquet('deploy_data/used_wx_spu_df.parquet')

# 搞结果dict
seg_order2skc_file = dict(zip(order_skc_df['order_skc_file'],order_skc_df['skc_file']))
order_skc2brand = dict(zip(order_skc_df['order_skc_file'],order_skc_df['pro_brand']))
seg_wx2skc_file = dict(zip(wx_spu_df['wx_spu_file'],wx_spu_df['skc_file']))
wx_spu2brand = dict(zip(wx_spu_df['wx_spu_file'],wx_spu_df['pro_brand']))
skc_file2spu = dict(zip(attr['skc_file'],attr['m_product_id'].astype(str)))
skc_file2brand = dict(zip(attr['skc_file'],attr['pro_brand']))

with open('deploy_data/labels_count.pkl', 'rb') as fp:
    labels_count = pickle.load(fp)
with open('deploy_data/label_mapping.pkl', 'rb') as fp:
    label_mapping = pickle.load(fp)

# 加载分割模型
seg_model = torch.load('models/unet_iter_1300000.pt') # , map_location='cpu'

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

# 加载searchnet
search_model = SearchNet(pre_model,baseline)
search_model.cuda()
search_model.eval()


index_n_mapping_dict = {}
for b in brand_list + ['all']:
    inner_dict = {}
    for p in cate_list:
        if not os.path.exists('./deploy_data/' + b + '/' + p + '/' + 'retrival_index'+post_fix+'.index'):
            logger.info('faiss {}_{} not exists'.format(b, p))
            continue
        if not os.path.exists('./deploy_data/' + b + '/' + p + '/' + 'pro_mapping_dict'+post_fix+'.json'):
            logger.info('map dict {}_{} not exists'.format(b, p))
            continue
        # 读取faiss index
        index = faiss.read_index('./deploy_data/'+b+'/'+p+'/'+'retrival_index'+post_fix+'.index')
        # 读取 pro_mapping_dict
        with open('./deploy_data/'+b+'/'+p+'/'+'pro_mapping_dict'+post_fix+'.json', 'rb') as json_file:
            pro_mapping_dict = json.load(json_file)
        inner_dict[p] = [index,pro_mapping_dict]
    index_n_mapping_dict[b] = inner_dict

def get_index_n_mapping(brand,pro_class):
    if brand not in index_n_mapping_dict:
        return {}, {}
    if pro_class not in index_n_mapping_dict[brand]:
        return {}, {}
    return index_n_mapping_dict[brand][pro_class]

def reload_data():
    logger.info('reload data start')
    global brand_dict

    # 预存数据
    test_date = datetime.datetime.now()
    test_date = (test_date - datetime.timedelta(days=1))
    test_date = test_date.strftime("%Y%m%d")

    # 加载商品属性表
    attr = pd.read_parquet('deploy_data/used_attr.parquet')
    order_skc_df = pd.read_parquet('deploy_data/used_order_skc_df.parquet')
    wx_spu_df = pd.read_parquet('deploy_data/used_wx_spu_df.parquet')

    # 搞结果dict
    seg_order2skc_file = dict(zip(order_skc_df['order_skc_file'], order_skc_df['skc_file']))
    order_skc2brand = dict(zip(order_skc_df['order_skc_file'], order_skc_df['pro_brand']))
    seg_wx2skc_file = dict(zip(wx_spu_df['wx_spu_file'], wx_spu_df['skc_file']))
    wx_spu2brand = dict(zip(wx_spu_df['wx_spu_file'], wx_spu_df['pro_brand']))
    skc_file2spu = dict(zip(attr['skc_file'], attr['m_product_id'].astype(str)))
    skc_file2brand = dict(zip(attr['skc_file'], attr['pro_brand']))

    with open('deploy_data/labels_count.pkl', 'rb') as fp:
        labels_count = pickle.load(fp)
    with open('deploy_data/label_mapping.pkl', 'rb') as fp:
        label_mapping = pickle.load(fp)

    # 加载分割模型
    seg_model = torch.load('models/unet_iter_1300000.pt')  # , map_location='cpu'

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

    # 加载searchnet
    search_model = SearchNet(pre_model,baseline)
    search_model.cuda()
    search_model.eval()

    index_n_mapping_dict = {}
    for b in brand_list + ['all']:
        inner_dict = {}
        for p in cate_list:
            if not os.path.exists('./deploy_data/' + b + '/' + p + '/' + 'retrival_index'+post_fix+'.index'):
                logger.info('faiss {}_{} not exists'.format(b, p))
                continue
            if not os.path.exists('./deploy_data/' + b + '/' + p + '/' + 'pro_mapping_dict'+post_fix+'.json'):
                logger.info('map dict {}_{} not exists'.format(b, p))
                continue
            # 读取faiss index
            index = faiss.read_index('./deploy_data/' + b + '/' + p + '/' + 'retrival_index'+post_fix+'.index')
            # 读取 pro_mapping_dict
            with open('./deploy_data/' + b + '/' + p + '/' + 'pro_mapping_dict'+post_fix+'.json', 'rb') as json_file:
                pro_mapping_dict = json.load(json_file)
            inner_dict[p] = [index, pro_mapping_dict]
        index_n_mapping_dict[b] = inner_dict

    logger.info('Reload data at {}'.format(test_date))

    return

def find_brand(x):
    if len(x.split('_')[1]) == 3:
        return skc_file2brand[x.split('_')[0]+'_'+x.split('_')[1]]
    elif len(x.split('_')[1]) == 2:
        return wx_spu2brand[x.split('_')[0]+'_'+x.split('_')[1]+'_'+x.split('_')[2]]
    elif len(x.split('_')[1]) == 1:
        return order_skc2brand[x.split('_')[0]+'_'+x.split('_')[1]]
    else:
        return 'all'
    return 'all'
# 根据mapping到的item找到skc_file
def find_skc_file(x):
    if len(x.split('_')[1]) == 3:
        return x.split('_')[0]+'_'+x.split('_')[1]
    elif len(x.split('_')[1]) == 2:
        return seg_wx2skc_file[x.split('_')[0]+'_'+x.split('_')[1]+'_'+x.split('_')[2]]
    elif len(x.split('_')[1]) == 1:
        return seg_order2skc_file[x.split('_')[0]+'_'+x.split('_')[1]]
    else:
        return 'invalid'
    return 'invalid'

def execute_retrival(brand,img,seg_model,search_model):
    # 分割模型提取masks
    masks = get_masks(seg_model, img)
    res_list = []
    for pt, mask in enumerate(masks):
        res_dict = {}
        sub_img, _ = seg_img_by_unet_pad(mask, img, pad_rate = 0.1)

        base64_str = cv2.imencode('.jpg', cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR))[1][:, -1].tostring()
        base64_str = base64.b64encode(base64_str)
        base64_str = base64_str.decode('ascii')
        res_dict['img'] = base64_str
        res_dict['pt'] = pt

        sub_img = sub_img.reshape((1, 224, 224, 3))
        sub_img = np.transpose(sub_img, (0, 3, 1, 2))
        sub_img = torch.FloatTensor(sub_img)
        # predict
        out_pred, out_clf = search_model(sub_img.cuda())
        # predict clf
        out_clf = out_clf.data.cpu()
        ones = torch.ones_like(out_clf)
        zeros = torch.zeros_like(out_clf)
        start = 0
        end = 0
        index_min = 0
        index_max = labels_count[0]
        for index, count in enumerate(labels_count):
            end += count
            if index == label_index:
                index_min = start
                index_max = end
            out_clf[:, start:end] = torch.softmax(out_clf[:, start:end], dim=1)
            start += count
        out_clf = out_clf.numpy()
        # 得到类名
        pro_index = np.argmax(out_clf[:, index_min:index_max])
        pro_class = label_mapping[pro_index + index_min]
        res_dict['pro_class'] = pro_class
        logger.info('search in {}'.format(pro_class))
        # save feat into dict
        out_pred = out_pred.data.cpu().numpy()
        normalize_L2(out_pred)

        index,pro_mapping_dict = get_index_n_mapping(brand,pro_class)
        if len(pro_mapping_dict) == 0:
            logger.info('empty at {}_{}'.format(brand,pro_class))
            temp_list = []
        else:
            k = 10  # top K
            index.nprobe = 100  # make comparable with experiment above max = nlist == violence search
            D, I = index.search(out_pred, k * 5)
            result = D[0]
            pids = list(I[0])
            res_df = pd.DataFrame(columns=['item', 'score'])
            res_df['item'] = pids
            res_df['score'] = result
            res_df['item_dir'] = res_df['item'].map(lambda x: pro_mapping_dict[str(x)])
            res_df['item'] = res_df['item_dir'].map(lambda x: find_skc_file(x))
            res_df['pro_class'] = pro_class
            res_df['pt'] = pt
            temp_dict = dict(zip(res_df.head(k * 5)['item'], res_df.head(k * 5)['score']))
            temp_dict_ori = dict(zip(res_df.head(k * 5)['item'], res_df.head(k * 5)['item_dir']))
            temp_list = []
            pids = []
            for pid, score in temp_dict.items():
                if pid not in pids:
                    try:
                        spu = skc_file2spu[pid]
                        pids.append(spu)
                        temp_list.append({'m_product_id': spu, 'score': score, \
                                      'item': temp_dict_ori[pid],'pro_class': pro_class})
                    except:
                        logger.info('lost one')
                        continue
                if len(pids) == k:
                    break
        res_dict['list'] = temp_list
        res_list.append(res_dict)
    return res_list

app = Flask(__name__)
app.debug = True
# app.debug = False

@app.route('/retriver/',methods=['get','post'],strict_slashes=False)
def cloth_retriver():
    if request.method == 'GET':
        result = {}
        result['code'] = 200
        result['info'] = 'please use post method'
        result['data'] = [{'img':'base64 image decode by ascii','list':[{'m_product_id':'111','score':'0.99'}]}]
        logger.info(json.dumps(result))
        return json.dumps(result)
    try:
        # 接收文件
        # request.values.get('')获取普通参数 request.files.get('')获取文件参数
        # bytes_stream = request.files.get('file_name').read()
        # bytes_stream = BytesIO(bytes_stream)
        # capture_img = Image.open(bytes_stream)
        # capture_img = np.asarray(capture_img,dtype='uint8')
        # 接收url
        # bytes_stream = request.get_data()
        # bytes_stream = json.loads(bytes_stream)
        # bytes_stream = bytes_stream['img']
        # capture_img = get_image_from_url(bytes_stream)
        # 接收str
        pointer = 0
        bytes_stream = request.get_data()
        bytes_stream = json.loads(bytes_stream)
        if 'brand' in bytes_stream:
            brand = bytes_stream['brand']
            if brand not in ['all','JNBY','CROQUIS','LESS','tjnby']:
                pointer += 1
                raise ValueError("wrong brand name")
        else:
            brand = 'all'
        bytes_stream = bytes_stream['img']
        bytes_stream = bytes_stream.encode('ascii')  # ascii编码
        bytes_stream = base64.b64decode(bytes_stream)  # base64解码
        bytes_stream = BytesIO(bytes_stream)
        capture_img = Image.open(bytes_stream)
        capture_img = np.asarray(capture_img,dtype='uint8')
        # 接收后处理
        if len(capture_img.shape) !=3:
            pointer += 2
            raise ValueError("wrong channel")
        if capture_img.shape[2] < 3 or capture_img.shape[2] > 4:
            pointer += 3
            raise ValueError("wrong channel nums")
        elif capture_img.shape[2] == 4:
            capture_img = cv2.cvtColor(capture_img,cv2.COLOR_RGBA2RGB)
        # capture_img = cv2.cvtColor(capture_img,cv2.COLOR_BGR2RGB)
        result = {}
        res_list = execute_retrival(brand,capture_img,seg_model,search_model)
        result['code'] = 200
        result['data'] = res_list
    except Exception as e:
        # logger.debug(e)
        # logger.debug('图片post请求出现异常')
        result = {}
        result['code'] = 400
        result['info'] = 'Bad request: Wrong post format' + str(pointer)
    logger.info(json.dumps(result))
    return json.dumps(result)

if __name__ != '__main__':
    # 实例化调度器
    scheduler = APScheduler()
    scheduler.add_job(id='job1', func=reload_data, trigger='cron', day='*', hour='6', minute='0', second='0')
    scheduler.init_app(app)
    scheduler.start()
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == '__main__':
    # 实例化调度器
    scheduler = APScheduler()
    scheduler.add_job(id='job1', func=reload_data, trigger='cron', day='*', hour='6', minute='0', second='0')
    scheduler.init_app(app)
    scheduler.start()
    app.run(host='0.0.0.0', port=8081)