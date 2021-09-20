#!/usr/bin/env python
# encoding: utf-8
"""
@AUTHOR:
@LICENCE: (C)Copyright 2013-2020, JNBY+ Corporation Limited
@CONTACT:
@FILE:
@TIME:
@DESC:
"""
import time
import datetime
import os
import gc
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from collections import Counter
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

"""
对于存储的数据操作, 得到程序需要的dataframe
"""

# 得到模型使用的销售记录
def get_model_users(used_pdf):
    # 购买尺码小于四个
    temp_pdf = used_pdf[['unionid','pro_brand','label']].groupby(['unionid','pro_brand']).nunique().reset_index()
    temp_pdf = temp_pdf.loc[temp_pdf['label'] < 4,['unionid','pro_brand']]
    model_pdf = pd.merge(used_pdf,temp_pdf,on=['unionid','pro_brand'])
    # 购买款式大于五件
    temp_pdf = used_pdf[['unionid','pro_brand','m_product_id']].groupby(['unionid','pro_brand']).nunique().reset_index()[['unionid','pro_brand','m_product_id']]
    temp_pdf = temp_pdf.drop(temp_pdf[(temp_pdf['m_product_id'].isin([1,2,3,4]))].index)
    temp_pdf = temp_pdf.drop(columns = ['m_product_id'])
    model_pdf = pd.merge(model_pdf,temp_pdf,on =['unionid','pro_brand'])
    model_pdf = model_pdf.drop_duplicates(keep = 'first')
    # 其他用户记录
    # other_pdf = used_pdf.drop(used_pdf[(used_pdf['unionid'].isin(model_pdf['unionid']))&(used_pdf['pro_brand'].isin(model_pdf['pro_brand']))].index)
    return model_pdf

def get_dataset(brand,test_date):
    logger.info("start get dataset")
    sale_pdf = pd.read_parquet(os.getcwd()+'/a_load_data/data/size_sale_sdf.parquet')
    sale_pdf = sale_pdf[sale_pdf['pro_brand'] == brand]
    logger.info("get sale dataframe count: {}".format(len(sale_pdf)))
    # 得到模型预测使用的销售记录
    model_pdf = get_model_users(sale_pdf)
    logger.info("get model dataframe count: {}".format(len(model_pdf)))
    # 拆分train_val_test
    # 最近五天作为测试集
    check_date = test_date
    fiveday = datetime.timedelta(days=5)
    last_date = str(datetime.datetime.strptime(check_date, "%Y%m%d") - fiveday)[:10].replace('-', '')
    test_pdf = model_pdf[(model_pdf['billdate']>int(last_date))&(model_pdf['billdate']<=int(check_date))]
    # 用户最后一次购买作为验证集
    model_pdf = model_pdf[model_pdf['billdate']<=int(last_date)]
    val_pdf = model_pdf.sort_values('billdate',ascending=False).drop_duplicates(subset=['unionid','pro_brand'],keep='first')
    model_pdf = pd.concat([model_pdf, val_pdf], axis=0, ignore_index=True)
    train_pdf = model_pdf.drop_duplicates(keep=False)
    logger.info("get dataset finished")
    return train_pdf, val_pdf, test_pdf

# 清洗用户
def drop_no(x):
    labels = x[0]
    counts = x[1]
    res = []
    for i in range(len(labels)):
        if counts[i]/sum(counts) >= 0.2:
            res.append(labels[i])
    return res
def wash_user(pdf,encode_dict):
    # check_pdf = pdf.groupby(['unionid','label'])['m_product_id'].count().reset_index()
    # check_pdf = check_pdf.groupby('unionid').apply(lambda x: [list(x[d]) for d in ['label','m_product_id']]).reset_index()
    # check_pdf['used_no'] = check_pdf[0].map(lambda x: drop_no(x))
    # check_pdf = check_pdf[['unionid','used_no']]
    # pdf = pd.merge(pdf,check_pdf,on='unionid')
    # pdf['label'] = pdf.apply(lambda x: x['label'] if x['label'] in x['used_no'] else 'drop',axis=1)
    # pdf = pdf[pdf['label'] != 'drop']
    # pdf = pdf.drop(columns = ['used_no'])
    check_pdf = pdf.groupby(['unionid','billdate']).apply(lambda x: list(x['label'])).reset_index()
    check_pdf[0] = check_pdf[0].map(lambda x: [encode_dict[d] for d in x])
    check_pdf[0] = check_pdf[0].map(lambda x: 'valid' if max(x)-min(x) < 0.3 else 'drop')
    check_pdf = check_pdf[check_pdf[0] == 'valid']
    pdf = pd.merge(pdf,check_pdf[['unionid','billdate']], on=['unionid','billdate'])
    return pdf

def get_pro_index_feat_sim_dict(brand, feat_cols, train_pdf):
    attr_pdf = pd.read_parquet(os.getcwd()+'/a_load_data/data/size_attr_sdf.parquet')
    attr_pdf = attr_pdf[attr_pdf['pro_brand'] == brand]
    attr_pdf = attr_pdf[['m_product_id']+feat_cols].drop_duplicates()
    for c in feat_cols:
        attr_pdf = attr_pdf[attr_pdf[c].isin(train_pdf[c].unique())]
    product_index = get_integer_dict('m_product_id',attr_pdf)
    # 商品特征编码
    feat_dict = {}
    for c in feat_cols:
        temp_list = []
        temp_list.append(get_onehot_dict(c,train_pdf))
        temp_list.append(get_target_dict(c,train_pdf))
        feat_dict[c] = temp_list
    # 商品相似度
    sim_pdf = attr_pdf
    # sim_pdf = sim_pdf[sim_pdf['m_product_id'].isin(product_index.keys())]
    for c in feat_cols:
        sim_pdf[c] = sim_pdf[c].map(lambda x: list(np.hstack((feat_dict[c][0][x],feat_dict[c][1][x]))))
    for c in feat_cols:
        if c == feat_cols[0]:
            sim_pdf['all'] = sim_pdf[c]
        else:
            sim_pdf['all'] = sim_pdf.apply(lambda x: np.hstack((x['all'],x[c])),axis=1)
    sim_pdf['all'] = sim_pdf['all'].map(lambda x: list(x))
    sim_pdf['m_product_id'] = sim_pdf['m_product_id'].map(lambda x: product_index[x])
    sim_pdf = sim_pdf.sort_values('m_product_id').set_index('m_product_id')
    sim_dict = {}
    # for c in feat_cols+['all']:
    for c in ['all']:
        sim_dict[c] = cosine_similarity(np.asarray(list(sim_pdf[c])))
    return product_index, feat_dict, sim_dict

def generate_feat_df(pdf,user_history_dict):
    func_hist = lambda x: user_history_dict[x['unionid']][1][-len([d for d in user_history_dict[x['unionid']][0] if d<x['billdate']]):]
    func_pro = lambda x: user_history_dict[x['unionid']][2][-len([d for d in user_history_dict[x['unionid']][0] if d<x['billdate']]):][0]
    func_time = lambda x: ([d for d in user_history_dict[x['unionid']][0] if d<x['billdate']]+[x['billdate']])[0]
    func_type = lambda x: user_history_dict[x['unionid']][3][-len([d for d in user_history_dict[x['unionid']][0] if d<x['billdate']]):]
    def time_diff(x):
        diff = (datetime.datetime.strptime(str(x['billdate']), "%Y%m%d")\
                       -datetime.datetime.strptime(str(x['lastdate']), "%Y%m%d")).days
        if diff > 0:
            diff = np.log(diff)
        return diff
    def most_size(x):
        return Counter(x).most_common(1)[0][0]
    # 得到用户日期以前的销售记录
    pdf['hist'] = pdf.apply(func_hist, axis=1)
    pdf['hist_type'] = pdf.apply(func_type, axis=1)
    # 用户购买次数
    pdf['buy_count'] = pdf['hist'].map(lambda x: len(x))
    pdf['buy_count'] = pdf['buy_count']/pdf['buy_count'].max()
    # 用户购买尺码个数
    pdf['size_count'] = pdf['hist'].map(lambda x: len(set(x))-1)
    pdf['size_count'] = pdf['size_count']/pdf['size_count'].max()
    # 最近一次购买的尺码
    pdf['recent_size'] = pdf['hist'].map(lambda x: x[0])
    # 用户购买的最大尺码
    pdf['max_size'] = pdf['hist'].map(lambda x: max(x))
    # 用户购买的最小尺码
    pdf['min_size'] = pdf['hist'].map(lambda x: min(x))
    # 最近一次购买商品对应尺码占比
    # train_pdf['recent_quan'] = train_pdf.apply(lambda x: (x['recent_size']-x['min_size'])/(x['max_size']-x['min_size']) if (x['max_size']-x['min_size'])!=0 else 1,axis = 1)
    pdf['recent_quan'] = pdf.apply(lambda x: x['hist'].count(x['recent_size'])/len(x['hist']),axis=1)
    # 同类商品购买比例
    pdf['same_count'] = pdf.apply(lambda x: x['hist_type'].count(x['pro_big_type'])/len(x['hist_type']),axis=1)
    # 主码
    pdf['most_size'] = pdf['hist'].map(lambda x: most_size(x))
    # # 主码对应尺码占比
    # pdf['most_quan'] = pdf.apply(lambda x: x['hist'].count(x['most_size'])/len(x['hist']),axis=1)
    # 最近一次购买记录
    pdf['lastdate'] = pdf.apply(func_time,axis = 1)
    # 最近一次购买时间差
    pdf['time_diff'] = pdf.apply(time_diff,axis = 1)
    # 最近一次购买的商品
    pdf['lastpro'] = pdf.apply(func_pro,axis = 1)
    return pdf

def generate_feat_df_rec(pdf,user_history_dict):
    func_hist = lambda x: user_history_dict[x][1]
    func_pro = lambda x: user_history_dict[x][2][0]
    func_time = lambda x: user_history_dict[x][0][0]
    func_type = lambda x: user_history_dict[x][3]
    # 填充billdate
    test_date = datetime.datetime.strptime(str(datetime.datetime.now())[:10], "%Y-%m-%d")
    pdf['billdate'] = int(str(test_date)[:10].replace('-',''))
    def time_diff(x):
        diff = (datetime.datetime.strptime(str(x), "%Y%m%d")-test_date).days
        if diff > 0:
            diff = np.log(diff)
        return diff
    def most_size(x):
        return Counter(x).most_common(1)[0][0]
    # 得到用户日期以前的销售记录
    pdf['hist'] = pdf['unionid'].map(func_hist)
    pdf['hist_type'] = pdf['unionid'].map(func_type)
    logger.info('1')
    # 用户购买次数
    pdf['buy_count'] = pdf['hist'].map(lambda x: len(x))
    pdf['buy_count'] = pdf['buy_count']/pdf['buy_count'].max()
    logger.info('2')
    # 用户购买尺码个数
    pdf['size_count'] = pdf['hist'].map(lambda x: len(set(x))-1)
    pdf['size_count'] = pdf['size_count']/pdf['size_count'].max()
    logger.info('3')
    # 最近一次购买的尺码
    pdf['recent_size'] = pdf['hist'].map(lambda x: x[0])
    logger.info('4')
    # 用户购买的最大尺码
    pdf['max_size'] = pdf['hist'].map(lambda x: max(x))
    logger.info('5')
    # 用户购买的最小尺码
    pdf['min_size'] = pdf['hist'].map(lambda x: min(x))
    logger.info('6')
    # 最近一次购买商品对应尺码占比
    # train_pdf['recent_quan'] = train_pdf.apply(lambda x: (x['recent_size']-x['min_size'])/(x['max_size']-x['min_size']) if (x['max_size']-x['min_size'])!=0 else 1,axis = 1)
    pdf['recent_quan'] = pdf.apply(lambda x: x['hist'].count(x['recent_size'])/len(x['hist']),axis=1)
    logger.info('7')
    # 同类商品购买比例
    pdf['same_count'] = pdf.apply(lambda x: x['hist_type'].count(x['pro_big_type'])/len(x['hist_type']),axis=1)
    logger.info('8')
    # 主码
    pdf['most_size'] = pdf['hist'].map(lambda x: most_size(x))
    logger.info('9')
    # # 主码对应尺码占比
    # pdf['most_quan'] = pdf.apply(lambda x: x['hist'].count(x['most_size'])/len(x['hist']),axis=1)
    # 最近一次购买记录
    pdf['lastdate'] = pdf['unionid'].map(func_time)
    logger.info('10')
    # 最近一次购买时间差
    pdf['time_diff'] = pdf['billdate'].map(time_diff)
    logger.info('11')
    # 最近一次购买的商品
    pdf['lastpro'] = pdf['unionid'].map(func_pro)
    logger.info('12')
    return pdf

# 用户及对应主码dict
def get_main_dict(df):
    # label是encode后label, 需要同模型预测结果decode
    temp_pdf = df[['unionid','label','m_product_id']]
    main_pdf = temp_pdf.groupby(['unionid','label'])['m_product_id'].count().reset_index().sort_values('m_product_id',ascending=False).drop_duplicates(subset=['unionid'],keep='first')
    main_pdf = main_pdf[['unionid','label']].rename(columns = {'label':'most_size'})
    main_dict = dict(zip(main_pdf.loc[:,'unionid'], main_pdf.loc[:,'most_size']))
    return main_dict

# 用户及对应类别的主码
def get_type_main_dict(df):
    # label是encode后label, 需要同模型预测结果decode
    temp_pdf = df[['unionid','pro_big_type','label','m_product_id']]
    type_main_pdf = temp_pdf.groupby(['unionid','pro_big_type','label'])['m_product_id'].count().reset_index().sort_values('m_product_id',ascending=False).drop_duplicates(subset=['unionid','pro_big_type'],keep='first')
    type_main_pdf = type_main_pdf[['unionid','pro_big_type','label']].rename(columns={'label':'type_most_size'})
    type_main_dict = dict(type_main_pdf.groupby('unionid').apply(lambda x: dict(zip(x['pro_big_type'],x['type_most_size']))))
    return type_main_dict

"""
特征构造、编码解码、生成模型训练、验证、测试集
"""
def encode_decode(brand):
    if brand == 'JNBY':
        from s_size_classify.tools.config import jnby_encode_dict, jnby_decode_dict
        encode_dict = jnby_encode_dict
        decode_dict = jnby_decode_dict
    elif brand == 'LESS':
        from s_size_classify.tools.config import less_encode_dict, less_decode_dict
        encode_dict = less_encode_dict
        decode_dict = less_decode_dict
    elif brand == 'CROQUIS':
        from s_size_classify.tools.config import croquis_encode_dict, croquis_decode_dict
        encode_dict = croquis_encode_dict
        decode_dict = croquis_decode_dict
    elif brand == 'tjnby':
        from s_size_classify.tools.config import tjnby_encode_dict, tjnby_decode_dict
        encode_dict = tjnby_encode_dict
        decode_dict = tjnby_decode_dict
    else:
        logger.info("The brand {} does not exist".format(brand))
        exit(0)
    return encode_dict,decode_dict

# 得到特征对应target编码dict
def get_target_dict(col_name,train_pdf):
    e_y = train_pdf['label'].mean()
    col_dict = dict(train_pdf.groupby(col_name).apply(lambda x: list(x['label'])))
    for d in col_dict:
        col_dict[d] = np.mean(col_dict[d])/e_y
    return col_dict

# 得到特征对应integer编码dict
def get_integer_dict(col_name,train_pdf):
    col_list = train_pdf[col_name].unique()
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(col_list)
    col_dict = {}
    for i in range(len(col_list)):
        col_dict[col_list[i]] = integer_encoded[i]
    return col_dict

# 得到特征对应onehot编码dict
def get_onehot_dict(col_name,train_pdf):
    col_list = train_pdf[col_name].unique()
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(col_list)
    # onehot encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    col_dict = {}
    for i in range(len(col_list)):
        col_dict[col_list[i]] = onehot_encoded[i]
    return col_dict

# 得到embedding编码
def get_embed_dict(col_name,train_pdf):
    tag_set = train_pdf[col_name].unique()
    embedding_dim = 4
    embedding_params=tf.Variable(tf.random.truncated_normal([len(tag_set),embedding_dim])).numpy()
    col_dict = {}
    for i in range(len(tag_set)):
        col_dict[tag_set[i]] = embedding_params[i]
    return col_dict

# 生成销售模型数据集
def generate_dataset(pdf, product_index, user_index, feat_cols, user_feat_cols, feat_dict):
    t_start=time.time()
    # 加入最近一次购买商品和目标商品的相似度 lastsim
    # sim_cols = ['sim_'+c for c in feat_cols+['all']]
    sim_cols = ['sim_'+c for c in ['all']]
    X = np.array(list(pdf['unionid'].map(lambda x: user_index[x])))[:,np.newaxis]
    X = np.hstack((X,np.array(list(pdf['m_product_id'].map(lambda x: product_index[x])))[:,np.newaxis]))
    logger.info('generated user & pro index')
    for c in feat_cols:
        X = np.hstack((X,np.array(list(pdf[c].map(lambda x: feat_dict[c][0][x])))))
        X = np.hstack((X,np.array(list(pdf[c].map(lambda x: feat_dict[c][1][x])))[:,np.newaxis]))
    logger.info('generated pro feats')
    for c in user_feat_cols:
        X = np.hstack((X,np.array(list(pdf[c]))[:,np.newaxis]))
    logger.info('generated user feats')
    for c in sim_cols:
        X = np.hstack((X,np.array(list(pdf[c]))[:,np.newaxis]))
    logger.info('generated sim feats')
    y = pdf['label'].values
    t_end = time.time()
    logger.info("generate dataset time cost: {}s".format(t_end-t_start))
    logger.info("X shape: {}".format(X.shape))
    logger.info("Y shape: {}".format(y.shape))
    return X,y

# new
def generate_dataset_rec(pdf, product_index, user_index, feat_cols, user_feat_cols, feat_dict):
    t_start=time.time()
    # 加入最近一次购买商品和目标商品的相似度 lastsim
    # sim_cols = ['sim_'+c for c in feat_cols+['all']]
    sim_cols = ['sim_'+c for c in ['all']]
    X = np.array(list(pdf['unionid'].map(lambda x: user_index[x])))[:,np.newaxis]
    X = np.hstack((X,np.array(list(pdf['m_product_id'].map(lambda x: product_index[x])))[:,np.newaxis]))
    logger.info('generated user & pro index')
    for c in feat_cols:
        X = np.hstack((X,np.array(list(pdf[c].map(lambda x: feat_dict[c][0][x])))))
        X = np.hstack((X,np.array(list(pdf[c].map(lambda x: feat_dict[c][1][x])))[:,np.newaxis]))
    logger.info('generated pro feats')
    for c in user_feat_cols:
        X = np.hstack((X,np.array(list(pdf[c]))[:,np.newaxis]))
    logger.info('generated user feats')
    for c in sim_cols:
        X = np.hstack((X,np.array(list(pdf[c]))[:,np.newaxis]))
    logger.info('generated sim feats')
    t_end = time.time()
    logger.info("generate dataset time cost: {}s".format(t_end-t_start))
    logger.info("X shape: {}".format(X.shape))
    return X


def decode_label(decode_dict, actual, result):
    func1 = lambda x: list(decode_dict.keys())[np.argmin(abs(x-list(decode_dict.keys())))]
    func2 = lambda x: decode_dict[x]
    y_hat = np.array(list(map(func1,result)))
    y_hat = np.array(list(map(func2,y_hat)))
    y = np.array(list(map(func2,actual)))
    return y, y_hat

# 预测结果解码dict
def get_pred_decode_dict(product_index, encode_dict, brand):
    attr_pdf = pd.read_parquet(os.getcwd()+'/a_load_data/data/size_attr_sdf.parquet')
    attr_pdf = attr_pdf[attr_pdf['pro_brand'] == brand]
    attr_pdf['m_product_id'] = attr_pdf['m_product_id'].map(lambda x: product_index[x] if x in product_index.keys() else -1)
    attr_pdf = attr_pdf.drop(attr_pdf[attr_pdf['m_product_id']== -1].index)
    attr_pdf['encode_label'] = attr_pdf['label'].map(lambda x: encode_dict[x])
    pred_decode_dict = dict(attr_pdf.groupby('m_product_id').apply(lambda x: list(set(x['encode_label']))))
    return pred_decode_dict

# 预测结果转化成no后两位
def get_res_pdf(X,result,func1,func2):
    res_pdf = pd.DataFrame(X[:,:2].astype(np.int64),columns = ['user_index', 'product_index'])
    res_pdf['y_pred'] = result
    res_pdf['y_pred'] = res_pdf.apply(func1, axis=1)
    res_pdf['y_pred'] = res_pdf['y_pred'].map(func2)
    return res_pdf

def get_rec_dict(rec_df):
    rec_dict = rec_df.groupby('unionid').apply(lambda x: dict(zip(list([d for d in x['m_product_id']]),list([d[-2:] for d in x['no']]))))
    return rec_dict

# 主码替换
def replace_most_no(brand, rec_dict,other_dict):
    t_start = time.time()
    logger.info("get most no dict for unionid")
    sale_pdf = pd.read_parquet(os.getcwd()+'/a_load_data/data/size_sale_sdf.parquet')
    sale_pdf = sale_pdf[sale_pdf['pro_brand'] == brand]
    most_df = sale_pdf.groupby('unionid').apply(lambda x: list([d for d in x['label']])).reset_index()
    most_df[0] = most_df[0].apply(lambda x: Counter(x).most_common(1)[0][0])
    most_no_dict = dict(zip(most_df['unionid'],most_df[0]))
    logger.info("start most no replacement")
    new_rec_dict = {}
    for uid in tqdm(most_no_dict.keys()):
        if uid in rec_dict.keys():
            new_rec_dict[uid] = rec_dict[uid]
            for pid in new_rec_dict[uid].keys():
                new_rec_dict[uid][pid] = most_no_dict[uid]
    new_rec_dict[other_dict[brand]] = rec_dict[other_dict[brand]]
    t_end = time.time()
    logger.info("replace successfully: {}".format(t_end-t_start))
    return new_rec_dict,most_no_dict

# 主码替换结果修正
def get_most_decode_dict(encode_dict, brand):
    attr_pdf = pd.read_parquet(os.getcwd()+'/a_load_data/data/size_attr_sdf.parquet')
    attr_pdf = attr_pdf[attr_pdf['pro_brand'] == brand]
    attr_pdf['encode_label'] = attr_pdf['label'].map(lambda x: encode_dict[x])
    most_decode_dict = dict(attr_pdf.groupby('m_product_id').apply(lambda x: list(set(x['encode_label']))))
    return most_decode_dict

# 给rec_df填充商品特征
def get_rec_attr_feat(brand,rec_df,feat_cols):
    attr = pd.read_parquet(os.getcwd()+'/a_load_data/data/size_attr_sdf.parquet')
    attr = attr[attr['pro_brand'] == brand]
    t_cols = ['m_product_id','pro_brand'] + feat_cols
    attr = attr[t_cols]
    attr = attr.drop_duplicates(subset=['m_product_id'],keep='first')
    rec_df = pd.merge(rec_df,attr,on=['m_product_id'])
    return rec_df

def generate_rec_df(brand,rec_cols,rec_df):
    t_start = time.time()
    n_rec = pd.read_parquet(os.getcwd()+'/n_cv_rec/data/cv_rec_all_'+brand+'.parquet')
    n_rec['colors_code'] = n_rec['no'].map(lambda x: x[-5:-2])
    n_rec['sizes_code'] = n_rec['no'].map(lambda x: x[-2:])
    n_rec = n_rec[['unionid','m_product_id','isnew','score','colors_code','sizes_code']]
    rec_df = pd.merge(rec_df,n_rec,on=['unionid','m_product_id'])
    attr = pd.read_parquet(os.getcwd()+'/a_load_data/data/size_attr_sdf.parquet')
    attr = attr[attr['pro_brand'] == brand]
    attr = attr.rename(columns = {"pro_brand":"brand","pricelist":"price"})
    t_cols = rec_cols.copy()
    t_cols.remove('unionid')
    t_cols.remove('isnew')
    t_cols.remove('score')
    attr = attr[t_cols]
    attr['colors_code'] = attr['no'].map(lambda x: x[-5:-2])
    attr['sizes_code'] = attr['no'].map(lambda x: x[-2:])
    rec_df = pd.merge(rec_df,attr,on=['m_product_id','sizes_code','colors_code'])
    rec_df = rec_df[rec_cols+['recom_size']]
    t_end = time.time()
    logger.info("generate res_df: {}".format(t_end-t_start))
    return rec_df