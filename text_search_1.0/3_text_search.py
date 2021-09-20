# 搜索流程
# if 精确款号 --> 得到一个款号结果
# elif 模糊款号 > (阈值) --> 得到大于阈值结果
# elif 精确分类 --> 得到精确分类结果
# elif 模糊分类 > (阈值) --> 得到大于阈值结果
# else 全量直接倒排
# 关键词优先级

import logging
import datetime
import time
import os
import random
import re
# import tensorflow as tf
# import cv2
# from PIL import Image
import numpy as np
import pandas as pd
import chardet
import requests,json
from io import BytesIO
# from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import pickle
import jieba
from rapidfuzz import fuzz, process
from config.param_config import channel_dict, initial_words, brand_list, recall_cols, rank_cols,\
    bad_words, brand_mapping, attr_cols, skc_cols, special_words, word_cols


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# 初始化结巴加入initial_words中词
for iw in initial_words:
    jieba.add_word(iw)

# load各个品牌变量
brand_dict = {}
for brand in brand_list:
    jieba.load_userdict('data/'+brand+'/word_dictionary.txt')
    temp_dict = {}
    # ints_dict chen0806
    with open('data/' + brand + '/ints_dict.pkl', 'rb') as fp:
        temp_dict['ints_dict'] = pickle.load(fp)
    # name_list
    with open('data/'+brand+'/name_list.pkl','rb') as fp:
        temp_dict['name_list'] = pickle.load(fp)
    # recall_ix
    with open('data/'+brand+'/recall_ix.pkl','rb') as fp:
        temp_dict['recall_ix'] = pickle.load(fp)
    # rank_ix
    with open('data/'+brand+'/rank_ix.pkl','rb') as fp:
        temp_dict['rank_ix'] = pickle.load(fp)
    # rank_weigts
    with open('data/'+brand+'/rank_weights.pkl','rb') as fp:
        temp_dict['rank_weights'] = pickle.load(fp)
    temp_dict['all_skcs'] = set(pd.read_parquet('data/'+brand+'/brand_skc_df.parquet').skc.unique())
    # begin_score_dict
    with open('data/'+brand+'/begin_score_dict.pkl','rb') as fp:
        temp_dict['begin_score_dict'] = pickle.load(fp)
    # brand_recall_keys
    with open('data/'+brand+'/brand_recall_keys.pkl','rb') as fp:
        temp_dict['brand_recall_keys'] = pickle.load(fp)
    brand_dict[brand] = temp_dict

def search_stream(brand,query_str):
    query_str = re.sub('[^a-zA-Z0-9\u4e00-\u9fa5]', '', query_str)
    # 先去空白, 避免分词错误
    # query_str = query_str.replace(' ','')
    # 开始搜索
    drop_keys = bad_words #  [' '] +
    res_spus = []
    res_skcs = set([])
    time_start = time.time()

    # 搜索语句校验
    if brand not in brand_list or type(query_str) != str:
        logger.info('wrong params: brand {}, query {}'.format(brand, query_str))
        return res_spus
    else:
        logger.info('valid params: brand {}, query {}'.format(brand,query_str))
    time_end = time.time()
    logger.info('check params in {}s'.format(time_end - time_start))
    # 精确匹配69码
    if re.search('[0-9]{13}',query_str):
        ints_dict = brand_dict[brand]['ints_dict']
        ints_queries = re.findall('[0-9]{13}',query_str)
        for ints_query in ints_queries:
            if query_str in ints_dict:
                res_spus.append(ints_dict[ints_query])
                logger.info('find a specific intscode: {}'.format(res_spus))
                time_end = time.time()
                logger.info('finish in {}s'.format(time_end - time_start))
                return res_spus
            else:
                logger.info('invalid intscode')
        time_end = time.time()
        logger.info('accurate intscode in {}s'.format(time_end - time_start))
        logger.info('finish in {}s'.format(time_end - time_start))
        return res_spus
    if re.search('[0-9a-zA-Z]{6}[0-9]{2,}',query_str):
        for sub_q in re.findall('[0-9a-zA-Z]{6}[0-9]{2,}',query_str):
            if not re.search('jnby|less|croquis|apn|home|personal|note|JNBY|LESS|CROQUIS|APN|HOME|PERSONAL|NOTE',\
                             sub_q):
                name_list = brand_dict[brand]['name_list']
                name_queries = re.findall('[0-9a-zA-Z]{8,}', query_str)
                # 精确款号name
                for name in tqdm(name_list):
                    if name in name_queries:
                        res_spus.append(name)
                        logger.info('find specific spus: {}'.format(name))
                        time_end = time.time()
                        logger.info('finish in {}s'.format(time_end - time_start))
                        return res_spus
                time_end = time.time()
                logger.info('accurate name in {}s'.format(time_end - time_start))
                # 模糊搜索name
                for key in name_queries:
                    res_spus = process.extract(key, name_list, limit=10)
                    res_spus = [d[0] for d in res_spus if d[1] > 80]
                    if len(res_spus) != 0:
                        logger.info('find fuzzy spus: {}'.format(res_spus))
                        time_end = time.time()
                        logger.info('finish in {}s'.format(time_end - time_start))
                        return res_spus
                time_end = time.time()
                logger.info('fuzzy name in {}s'.format(time_end - time_start))
                logger.info('finish in {}s'.format(time_end - time_start))
                return res_spus
    # 分词
    recall_keys = jieba.lcut(query_str)
    # 去重
    recall_keys = list(set(recall_keys))
    # 去停用词
    recall_keys = list(set(recall_keys).difference(set(drop_keys)))
    drop_keys = []
    logger.info('{}'.format(recall_keys))
    time_end = time.time()
    logger.info('cut in {}s'.format(time_end - time_start))
    # 精确召回
    all_skcs = brand_dict[brand]['all_skcs']
    recall_ix = brand_dict[brand]['recall_ix']
    brand_recall_keys = brand_dict[brand]['brand_recall_keys']
    # 先过品牌
    for key in recall_keys:
        if key in brand_recall_keys:
            logger.info('find a brand recall key')
            res_skcs.update(all_skcs.intersection(recall_ix[key]))
            drop_keys.append(key)
    if len(res_skcs) != 0:
        all_skcs = res_skcs.copy()
        res_skcs = set([])
    recall_keys = list(set(recall_keys).difference(set(drop_keys)))
    # 过类别
    for key in recall_keys:
        if key in recall_ix.keys():
            logger.info('find accurate recall key: {}'.format(key))
            res_skcs.update(all_skcs.intersection(recall_ix[key]))
    rank_keys = list(set(recall_keys).difference(set(drop_keys)))
    if len(res_skcs) == 0:
        res_skcs = all_skcs.copy()
    logger.info('recall skc amount: {}'.format(len(res_skcs)))
    time_end = time.time()
    logger.info('accurate recall in {}s'.format(time_end - time_start))
    logger.info('rank keys: {}'.format(rank_keys))
    # # 模糊召回
    # time_end = time.time()
    # logger.info('fuzzy recall in {}s'.format(time_end - time_start))
    # # 排序
    rank_ix = brand_dict[brand]['rank_ix']
    rank_weights = brand_dict[brand]['rank_weights']
    # 初始分
    begin_score_dict = brand_dict[brand]['begin_score_dict']
    res_skcs_dict = {}
    for k in res_skcs:
        res_skcs_dict[k] = begin_score_dict[k]
    # 通用分
    # res_skcs_dict = dict(zip(res_skcs, [1.0 for i in range(len(res_skcs))]))
    # # 排序 （精准无关键字权重)
    # for key in rank_keys:
    #     if key in rank_ix.keys():
    #         logger.info(key)
    #         tmp_list = rank_ix[key]
    #         for skc in tqdm(list(res_skcs_dict.keys())):
    #             if skc in tmp_list:
    #                 res_skcs_dict[skc] += 1.0
    # # 排序 (精准有关键字权重)
    # for key in rank_keys:
    #     if key in rank_ix.keys():
    #         logger.info(key)
    #         tmp_list = rank_ix[key]
    #         for skc in tqdm(list(res_skcs_dict.keys())):
    #             if skc in tmp_list:
    #                 res_skcs_dict[skc] = res_skcs_dict[skc] + 1.0 * rank_weights[key]
    # 排序 (带权重)
    key_list = list(rank_ix.keys())
    key_weight_list = list(rank_weights.keys())
    for key in rank_keys:
        fuzz_keys = process.extract(key, key_list, limit=1)
        fuzz_keys = [d for d in fuzz_keys if d[1] > 80]
        if len(fuzz_keys) > 0:
            logger.info('search key: {}'.format(key))
        else:
            continue
        for fuzz_key in fuzz_keys:
            logger.info('fuzz key: {}'.format(fuzz_key))
            tmp_list = rank_ix[fuzz_key[0]]
            fuzz_weight = fuzz_key[1]/100
            for skc in tqdm(list(res_skcs_dict.keys())):
                if skc in tmp_list and key in key_weight_list:
                    res_skcs_dict[skc] = res_skcs_dict[skc] + 1.0 * rank_weights[key] * fuzz_weight
                elif skc in tmp_list and key not in key_weight_list:
                    res_skcs_dict[skc] = res_skcs_dict[skc] + 1.0 * fuzz_weight
                else:
                    # 非匹配skc加一个极小值
                    res_skcs_dict[skc] = res_skcs_dict[skc] + 1e-4
    # 整理结果
    res_skc2score_list = sorted(res_skcs_dict.items(), key=lambda x: x[1], reverse=True)
    for skc2score in res_skc2score_list:
        if skc2score[0][:-3] not in res_spus:
            res_spus.append(skc2score[0][:-3])
    time_end = time.time()
    logger.info('rank in {}s'.format(time_end - time_start))
    return res_spus


if __name__ == '__main__':
    brand = 'lasumin'
    query_str = 'less2021秋新品舒适短款修身圆领背心'
    result = search_stream(brand,query_str)
    test_df = pd.read_parquet('data/' + brand + '/brand_skc_df.parquet')
    col = 'mall_title'
    test_dict = dict(zip(test_df['name'], test_df[col]))
    for r in result[:20]:
        logger.info(test_dict[r])
    pass