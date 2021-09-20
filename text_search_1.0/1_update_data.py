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
from config.param_config import channel_dict, initial_words, brand_list, recall_cols, rank_cols,\
    bad_words, brand_mapping, attr_cols, skc_cols, special_words, word_cols

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# 初始化结巴加入initial_words中词
for iw in initial_words:
    jieba.add_word(iw)

# 滑动窗口平滑
def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')  # numpy的卷积函数

#归一化到0~1区间
def normalize(interval):
    min_ = interval.min()
    max_ = interval.max()
    if min_ == max_:
        res = np.ones(interval.shape)
        return res
    if min_ >= 1.0:
        interval = interval/max_
        min_ = min_/max_
        max_ = 1.0
    aug_param = (1.0-min_)/(max_-min_)
    res = (interval-min_)*aug_param+min_
    return res

# 字段非通用处理
def process_attr(df):
    # 所有缺失值填充
    all_cols = list(df.columns)
    for col in all_cols:
        if df[col].dtype == 'O':
            df[col] = df[col].fillna('unknown')
        elif df[col].dtype in ['int','int32','int64']:
            df[col] = df[col].fillna(0)
        elif df[col].dtype in ['float','float32','float64']:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna('unknown')

    # year
    df['abbr_year'] = df['year'].map(lambda x: x[2:])
    df['replace_band'] = df['band'].map(lambda x: x.replace('-',''))
    df['m_product_id'] = df['m_product_id'].map(lambda x: str(x))
    df['topic'] = df['topic'].map(lambda x: x.replace(' ', ''))

    # 去除空格
    for col in recall_cols+rank_cols:
        df[col] = df[col].map(lambda x: x.replace(' ',''))

    return df

def process_mall_title(df):

    df['mall_title'] = df['mall_title'].map(lambda x: x.replace(' ', ''))
    for special in special_words:
        df['mall_title'] = df['mall_title'].map(lambda x: x.replace(special, ''))
    df['mall_title'] = df['mall_title'].map(
        lambda x: re.sub('[0-9]?[0-9]?[0-9]{2}[年春夏秋冬][春夏秋冬]{0,2}[季装]?', '', x))
    df['mall_title'] = df['mall_title'].map(lambda x: x if not re.search('^.+[\u4e00-\u9fa5\)](?=[a-zA-Z0-9]{7})', x) \
        else re.findall('^.+[\u4e00-\u9fa5\)](?=[a-zA-Z0-9]{7})', x)[0])

    return df

# 保存品牌搜索记录
def get_brand_query_record(df,brand):
    query_df = df[(df['biz_type'] == 3)]
    query_df['query'] = query_df['content'].map(lambda x: re.findall('(?<=搜索了“).+(?=”)', x)[0] \
        if re.findall('(?<=搜索了\“).+(?=\”)', x) else '')
    query_list = query_df['query'].unique().tolist()
    with open('data/'+brand+'/query_record.txt','w',encoding='utf-8') as f:
        for q in query_list:
            f.write(q)
            f.write('\n')
    logger.info('{} get {} query records'.format(brand,len(query_list)))
    return True

# 筛选可用商品
def brand_filter(skc_df, test_date, brand, channel = 0):
    df = skc_df.copy()
    # 过滤去掉
    if channel != 0:
        logger.info('当前暂时只支持微商城搜索场景')
        exit(0)

    # 先过滤品牌
    df = df[df['brand_name'].isin(brand_mapping[brand])]
    c_year = test_date[:4]
    c_month = int(test_date[4:6])
    # 奥莱
    if brand == 'samo':
        if c_month >= 2 and c_month <= 9:
            tmp_df1 = df[(df['year'].isin([str(int(c_year)-1),str(int(c_year)-2)]))\
                    &(df['big_season'].isin(['春夏','秋冬']))]
            df = tmp_df1.copy()
            del tmp_df1
        elif c_month >=10 and c_month <= 12:
            tmp_df1 = df[(df['year'].isin([str(int(c_year)-1),str(int(c_year)-2)]))\
                    &(df['big_season'].isin(['秋冬']))]
            tmp_df2 = df[(df['year'].isin([c_year, str(int(c_year) - 1)])) \
                         & (df['big_season'].isin(['春夏']))]
            df = pd.concat([tmp_df1,tmp_df2])
            del tmp_df1, tmp_df2
        else:
            tmp_df1 = df[(df['year'].isin([str(int(c_year) - 2), str(int(c_year) - 3)])) \
                         & (df['big_season'].isin(['秋冬']))]
            tmp_df2 = df[(df['year'].isin([str(int(c_year) -1), str(int(c_year) - 2)])) \
                         & (df['big_season'].isin(['春夏']))]
            df = pd.concat([tmp_df1,tmp_df2])
            del tmp_df1, tmp_df2
    # jnbyhome 特殊逻辑
    elif brand == 'jnbyh':
        tmp_df1 = df[(df['year']>'2016')]
        tmp_df2 = df[(df['year'] == '2016')&(df['big_season']=='秋冬')]
        df = pd.concat([tmp_df1,tmp_df2])
        del tmp_df1, tmp_df2
    # 非奥莱
    else:
        if c_month >= 5 and c_month <= 10:
            tmp_df1 = df[(df['year'].isin([c_year])) \
                         & (df['big_season'].isin(['春夏', '秋冬']))]
            df = tmp_df1.copy()
            del tmp_df1
        elif c_month >= 11:
            tmp_df1 = df[(df['year'].isin([c_year])) \
                         & (df['big_season'].isin(['秋冬']))]
            tmp_df2 = df[(df['year'].isin([str(int(c_year)+1)])) \
                         & (df['big_season'].isin(['春夏']))]
            df = pd.concat([tmp_df1,tmp_df2])
            del tmp_df1, tmp_df2
        else:
            tmp_df1 = df[(df['year'].isin([str(int(c_year)-1)])) \
                         & (df['big_season'].isin(['秋冬']))]
            tmp_df2 = df[(df['year'].isin([c_year])) \
                         & (df['big_season'].isin(['春夏']))]
            df = pd.concat([tmp_df1, tmp_df2])
            del tmp_df1, tmp_df2

    # 是否过滤库存
    df.to_parquet('data/' + brand + '/brand_skc_df.parquet')
    logger.info('{} has {} available skcs in channel {}'.format(brand,len(df),channel_dict[channel]))

    return df

def brand_filter_v2(skc_df, brand_mall_df, test_date, brand, channel = 0):
    df = skc_df.copy()
    # 过滤去掉
    if channel != 0:
        logger.info('当前暂时只支持微商城搜索场景')
        exit(0)
    # 先过滤品牌
    # df = df[df['brand_name'].isin(brand_mapping[brand])]
    df = df[df['name'].isin(brand_mall_df['product_no'].unique())]
    c_year = test_date[:4]
    c_month = int(test_date[4:6])
    # 将mall_classify 加入到 df
    mc_dict = dict(zip(brand_mall_df['product_no'],brand_mall_df['mall_classify']))
    df['mall_classify'] = df['name'].map(lambda x: mc_dict[x])
    # mall_title 替换 df 中 mall_title
    mt_dict = dict(zip(brand_mall_df['product_no'],brand_mall_df['mall_title']))
    df['mall_title'] = df['name'].map(lambda x: mt_dict[x])

    # 是否过滤库存
    df.to_parquet('data/' + brand + '/brand_skc_df.parquet')
    logger.info('{} has {} available skcs in channel {}'.format(brand, len(df), channel_dict[channel]))

    return df

# 倒排索引更新
def generate_inverted_index(df,cols):
    result_ix = {}
    for col in cols:
        temp_df = df.groupby(col).apply(lambda x: set([d for d in x['skc']])).reset_index()
        inverted_dict = dict(zip(temp_df[col], temp_df[0]))
        result_ix.update(inverted_dict)
    # 最后统一删除
    # for bad in bad_words:
    #     if bad in result_ix:
    #         del result_ix[bad]
    return result_ix

def update_recall_with_synonym(ix, synonym_dict):
    result_ix = ix.copy()
    for k,vs in synonym_dict.items():
        if k in result_ix:
            for v in vs:
                if v in result_ix:
                    result_ix[k].update(result_ix[v])
        else:
            result_ix[k] = set([])
            for v in vs:
                if v in result_ix:
                    result_ix[k].update(result_ix[v])
    return result_ix

# mall_classify 构建倒排索引加入 recall_ix
def update_recall_with_mc(ix,df):
    result_ix = ix.copy()
    tmp_dict = dict(zip(df['skc'],df['mall_classify'].map(lambda x: [d for d in x.split('_') if d not in bad_words])))
    update_ix = {}
    for key, values in tqdm(tmp_dict.items()):
        for value in values:
            if value not in update_ix:
                update_ix[value] = set([key])
            else:
                update_ix[value].add(key)
    update_ix = {key: value for key, value in update_ix.items() if key not in result_ix}
    result_ix.update(update_ix)

    return result_ix


# 复杂字符串更新rank_ix, 当前只使用了mall_title
def complex_update_rank_ix(ix,df,complex_col_name='mall_title'):
    result_ix = ix.copy()
    tmp_dict = dict(zip(df['skc'],df[complex_col_name].map(lambda x: [d for d in jieba.lcut(x) if d not in bad_words])))
    update_ix = {}
    for key, values in tqdm(tmp_dict.items()):
        for value in values:
            if value not in update_ix:
                update_ix[value] = set([key])
            else:
                update_ix[value].add(key)
    update_ix = {key:value for key,value in update_ix.items() if key not in result_ix}
    result_ix.update(update_ix)

    return result_ix

# 初始化分数: 当前根据年月, 越新分数越高
def get_begin_score_dict(df,brand):
    c_list = list(df.sort_col.unique())
    c_list.sort()
    c_dict = {}
    for i, c in enumerate(c_list):
        c_dict[c] = 1.0 + 0.01 / len(c_list) * (i + 1)
    begin_score_dict = dict(zip(df['skc'], df['sort_col'].map(lambda x: c_dict[x])))

    with open('data/' + brand + '/begin_score_dict.pkl', 'wb') as fp:
        pickle.dump(begin_score_dict, fp)
    return

def get_begin_score_dict_click(sdf,ldf,brand,log_date):
    tmp_df = sdf[['skc','name']].copy()
    #find商品码
    # click_df = ldf.query(f'biz_type == "2" and page_name == "商品详情页" and brand_name == "{brand}"').copy()
    click_df = ldf[(ldf['biz_type']==2)&(ldf['page_name']=="商品详情页")&(ldf['create_time']>log_date)]
    pattern = "(?<=[\u4e00-\u9fa5])[0-9A-Z]{8,9}(?=”)"
    click_df['name'] = click_df['content'].map(lambda x: re.findall(pattern,x)[0] if re.search(pattern,x) else None)
    click_df = click_df.loc[(~click_df['name'].isna()),['session_id','name']]
    click_df = click_df.groupby('name')['session_id'].nunique().reset_index()
    spu2click = dict(zip(click_df['name'],click_df['session_id']))
    tmp_df['click'] = tmp_df['name'].map(lambda x: spu2click[x] if x in spu2click else 0)
    tmp_df['click_rank'] = tmp_df['click'].rank(method = 'dense',ascending=True)
    c_list = list(tmp_df.click_rank.unique())
    c_list.sort()
    c_dict = {}
    for i, c in enumerate(c_list):
        c_dict[c] = 1.0 + 0.01 / len(c_list) * (i + 1)
    tmp_df['click_rank'].map(lambda x: c_dict[x])
    begin_score_dict = dict(zip(tmp_df['skc'], tmp_df['click_rank'].map(lambda x: c_dict[x])))
    with open('data/' + brand + '/begin_score_dict.pkl', 'wb') as fp:
        pickle.dump(begin_score_dict, fp)
    return

# 得到同义词dict(支持召回)
def process_synonym():
    with open('data/synonym.txt','r') as f:
        lines = f.readlines()
        lines = [d.strip() for d in lines]
    synonym_dict = {}
    for line in lines:
        synonym_dict[line.split(' ')[0]] = line.split(' ')[1].split('_')
    return synonym_dict

# 处理近义词和product_name得到更泛化的召回词
def process_product_name(df,brand):
    # synonym_dict = process_synonym()
    # pn2pns
    pn_dict = {}
    for i in df['product_name'].unique():
        words = jieba.lcut(i, cut_all=False)
        for w in words:
            if w in bad_words:
                continue
            if pn_dict.get(w, False):
                pn_dict[w].append(i)
            else:
                pn_dict[w] = [i]
    # for k,vs in synonym_dict.items():
    #     if k in pn_dict:
    #         for v in vs:
    #             if v in pn_dict:
    #                 pn_dict[k] = list(set(pn_dict[k] + pn_dict[v]))
    #     else:
    #         pn_dict[k] = set([])
    #         for v in vs:
    #             if v in pn_dict:
    #                 pn_dict[k] = list(set(pn_dict[k] + pn_dict[v]))
    extra_words = list(pn_dict.keys())
    # pn2skcs
    extra_dict = {}
    tmp_dict = generate_inverted_index(df, ['product_name'])
    for k,vs in tqdm(pn_dict.items()):
        tmp_list = []
        for v in vs:
            tmp_list += list(tmp_dict[v])
        extra_dict[k] = set(tmp_list)
    return extra_dict, extra_words

# 整理分词器需要加入的词集 (特殊词 + 属性 + mall_classify +MALL_TITLE)
def get_word_dictionary(df,brand,extra_words,synonym_words):
    word_list = []
    word_list += synonym_words
    # 商品分类mapping (例如: 衬: 衬衣)
    cate_mapping = {}
    for col in word_cols:
        df[col] = df[col].map(lambda x: x.replace(' ',''))
        word_list += df[col].unique().tolist()
    # 加入处理后的product_name
    word_list += extra_words
    # 加入mall_classify
    mc_list = list(df['mall_classify'].unique())
    for mc in mc_list:
        word_list += mc.split('_')
    # 加入mall_title中词汇
    mt_list = list(df['mall_title'].unique())
    for mt in tqdm(mt_list):
        word_list += jieba.lcut(mt)
    # 去除bad_words
    word_list = [d for d in word_list if d not in bad_words]
    # 去重
    word_list = list(set(word_list))
    logger.info('{} get {} words'.format(brand, len(word_list)))
    # 写入分词txt
    with open('data/'+brand+'/word_dictionary.txt','w',encoding='utf-8') as f:
        for word in word_list:
            f.write(word)
            f.write('\n')
    return True

if __name__ == '__main__':

    # 保存每个品牌的在售商品(所有可展示商品)
    time_start = time.time()
    test_date = datetime.datetime.strptime(str(datetime.datetime.now())[:10], "%Y-%m-%d")
    log_interval = datetime.timedelta(days=7)
    log_date = test_date - log_interval
    test_date = str(test_date)[:10].replace('-', '')

    attr_df = pd.read_parquet('data/data_attr_sdf.parquet')
    skc_df = pd.read_parquet('data/skc_df.parquet')
    skc_df = pd.merge(skc_df[skc_cols], attr_df[attr_cols], on=['name', 'skc'])
    skc_df = process_attr(skc_df)
    mall_df = pd.read_parquet('data/mall_df.parquet')
    # mall_df 只保留售卖中
    mall_df = mall_df[(mall_df['mall_iscansell'] == 1) & (mall_df['mall_isputway'] == 0)]
    log_df = pd.read_parquet('data/log_df.parquet')
    del attr_df
    mt_sep_res = []
    for brand in brand_list:
        brand_log_df = log_df[(log_df['brand_name'] == brand)]
        # 保存品牌搜索记录
        get_brand_query_record(brand_log_df, brand)
        # mall_info下对应品牌商品
        brand_mall_df = mall_df[mall_df['brand_name'] == brand]
        # 得到品牌可用商品
        # brand_skc_df = brand_filter(skc_df, test_date, brand=brand, channel=0)
        brand_skc_df = brand_filter_v2(skc_df, brand_mall_df, test_date, brand=brand, channel=0)
        brand_skc_df = process_mall_title(brand_skc_df)
        mt_sep_res += list(brand_skc_df.mall_title.unique())
        # 构建初始分
        # get_begin_score_dict(brand_skc_df,brand)
        get_begin_score_dict_click(brand_skc_df,brand_log_df,brand,log_date)
        # 近义词dict (key <- values的商品并集)
        synonym_dict = process_synonym()
        # 处理product_name和近义词得到关键字
        extra_dict, extra_words = process_product_name(brand_skc_df,brand)
        # 得到每个品牌需要加入的词典
        get_word_dictionary(brand_skc_df, brand, extra_words, list(synonym_dict.keys()))
        # 保存国标码
        ints_dict = dict(zip(brand_skc_df.loc[brand_skc_df['intscode']!='unknown','intscode'],\
                             brand_skc_df.loc[brand_skc_df['intscode']!='unknown','name']))
        with open('data/' + brand + '/ints_dict.pkl', 'wb') as fp:
            pickle.dump(ints_dict, fp)
        # 保存款号
        name_list = brand_skc_df['name'].unique().tolist()
        with open('data/' + brand + '/name_list.pkl', 'wb') as fp:
            pickle.dump(name_list, fp)
        # 保存品牌关键字
        brand_recall_keys = []
        for c in ['brand','brand_name','pro_brand']:
            brand_recall_keys += brand_skc_df[c].unique().tolist()
        brand_recall_keys = list(set(brand_recall_keys))
        with open('data/'+brand+'/brand_recall_keys.pkl','wb') as fp:
            pickle.dump(brand_recall_keys,fp)
        # 保存倒排索引
        recall_ix = generate_inverted_index(brand_skc_df,recall_cols)
        # 加入处理后的product_name
        for k,v in extra_dict.items():
            if k not in recall_ix:
                recall_ix.update({k:v})
        # 加入mall_classify
        recall_ix = update_recall_with_mc(recall_ix,brand_skc_df)
        # 近义词更新倒排索引
        recall_ix = update_recall_with_synonym(recall_ix, synonym_dict)
        for bad in bad_words:
            if bad in recall_ix:
                del recall_ix[bad]
        with open('data/'+brand+'/recall_ix.pkl','wb') as fp:
            pickle.dump(recall_ix,fp)
        logger.info('generated {} recall_ix with {} keys: {}'.format(brand,len(recall_ix),recall_ix.keys()))
        rank_ix = generate_inverted_index(brand_skc_df,rank_cols)
        # 加入mall_title
        rank_ix = complex_update_rank_ix(rank_ix,brand_skc_df,'mall_title')
        for bad in bad_words:
            if bad in rank_ix:
                del rank_ix[bad]
        # 召回词加入排序索引中,优化露出结果
        rank_ix.update(recall_ix)
        with open('data/'+brand+'/rank_ix.pkl','wb') as fp:
            pickle.dump(rank_ix,fp)
        # 排序字段权重
        rank_weights = dict(zip(list(rank_ix.keys()), [1.0 for d in list(rank_ix.keys())]))
        for key,value in rank_weights.items():
            tmp_keys = []
            for col in ['year','abbr_year','big_season', 'season', 'band', 'replace_band']:
                tmp_keys += list(brand_skc_df[col].unique())
            if key in tmp_keys:
                rank_weights[key] = 2.0
            tmp_keys = []
            for col in ['topic', 'style_label_two','integral_line','tg','tl','tx']:
                tmp_keys += list(brand_skc_df[col].unique())
            if key in tmp_keys:
                rank_weights[key] = 1.6
            tmp_keys = []
            for col in ['scmatertype3','colclass','coldeepshallow']:
                tmp_keys += list(brand_skc_df[col].unique())
            if key in tmp_keys:
                rank_weights[key] = 1.2
            tmp_keys = []
            for col in ['scmatertype1','clr_description','colsystem']:
                tmp_keys += list(brand_skc_df[col].unique())
            if key in tmp_keys:
                rank_weights[key] = 1.4
            tmp_keys = []
            for col in ['pro_category','scmatertype2','color_name','bomname']:
                tmp_keys += list(brand_skc_df[col].unique())
            if key in tmp_keys:
                rank_weights[key] = 1.8
            # 提高召回词排序权重，优化露出结果
            if key in recall_ix.keys():
                rank_weights[key] = 10.0
            for col in ['small_class']:
                tmp_keys += list(brand_skc_df[col].unique())
            if key in tmp_keys:
                rank_weights[key] = 20.0
        with open('data/'+brand+'/rank_weights.pkl','wb') as fp:
            pickle.dump(rank_weights,fp)
        logger.info('generated {} rank_ix with {} keys: {}'.format(brand,len(rank_ix),rank_ix.keys()))
    # 备份查看mall_title分词结果
    f1 = open('data/mt_sep_res.txt', 'w')
    f2 = open('data/mt_sep_words.txt','w')
    mt_sep_words = []
    for w in tqdm(mt_sep_res):
        # logger.info(w)
        # logger.info(','.join(jieba.lcut(w)))
        f1.write(','.join(jieba.lcut(w)) + '\n')
        mt_sep_words += jieba.lcut(w)
    f1.close()
    mt_sep_words = list(set(mt_sep_words))
    mt_sep_words = [d for d in mt_sep_words if d not in bad_words]
    for w in tqdm(mt_sep_words):
        f2.write(w+'\n')
    f2.close()

    time_end = time.time()
    logger.info("finished update data in {}s".format(time_end - time_start))


    pass