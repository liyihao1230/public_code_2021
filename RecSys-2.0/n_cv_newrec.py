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
import os
import logging

SPARK_HOME = '/opt/spark-2.4.7-bin-hadoop2.7'

# ++++ bigdata server ++++
PYTHON_PATH = "/root/anaconda3/envs/rec/bin/python3.7"
PYSPARK_LIB_PATH = "/root/anaconda3/envs/rec/lib/python3.7/site-packages"

# ++++ GPU server ++++
# PYTHON_PATH = "/home/administrator/anaconda3/envs/rec_1.1/bin/python3.7"
# PYSPARK_LIB_PATH = "/home/administrator/anaconda3/envs/rec_1.1/lib/python3.7/site-packages"

os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH
os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"
os.environ["HADOOP_USER_NAME"] = "admin"

from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, DoubleType, IntegerType

from a_load_data.cv_loader import CVDataLoader, upload_rec_res, upload_rec_res_loop
from a_load_data.qty_loader import QtyDataLoader
from n_cv_rec.tools.config import item_brand, item_style, attr_fields, big_type_list, brand_list, mbr_fields, mkt_fields, box_fields, qty_fields
from n_cv_rec.tools.data_process import get_newbee, get_sale, filter_purchased, plan_fix_score, fill_sku

import time
import datetime
import argparse
import re
import numpy as np
import pandas as pd
import pickle
import joblib

conf = SparkConf()
conf.set("spark.pyspark.python", PYTHON_PATH) \
    .set("spark.pyspark.driver.python", PYTHON_PATH) \
    .set("spark.executorEnv.PYSPARK_PYTHON", PYTHON_PATH) \
    .set("spark.executorEnv.PYSPARK_DRIVER_PYTHON", PYTHON_PATH) \
    .set("spark.yarn.appMasterEnv.PYSPARK_PYTHON", PYTHON_PATH) \
    .set("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON", PYTHON_PATH) \
    .set("spark.sql.execution.arrow.enabled", "true") \
    .set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation", "true") \
    .set("spark.sql.broadcastTimeout", "36000") \
    .set("spark.debug.maxToStringFields", "100") \
    .set("spark.driver.memory", "10g") \
    .set("spark.executor.memory", "8g") \
    .set("spark.driver.maxResultSize", "10g") \
    .set("spark.scheduler.listenerbus.eventqueue.capacity", "200000")

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# 上传的table
main_table = 'jnby_ads.ads_jnby_rec_res_model_main_test'
new_table = 'jnby_ads.ads_jnby_rec_res_model_new_test'
all_table = 'jnby_ads.ads_jnby_rec_res_model_1_test'


if __name__ == '__main__':
    time_start = time.time()
    # 设置脚本参数
    parser = argparse.ArgumentParser()
    parser.add_argument("brand", help="the brand we want to use")
    parser.add_argument("labelname", help="the label we gonna use")
    args = parser.parse_args()
    brand = args.brand
    labelname = args.labelname
    logger.info('brand')
    logger.info('labelname')
    label_list = ['all', 'pro_big_type', 'style_label_two', 'm_dim21_id', 'bomcode', 'design_empid','origin']
    if labelname not in label_list or brand not in brand_list:
        logger.info('arg error, exit')
        exit(0)

    # 1. 数据读取
    interval = datetime.timedelta(days=550)
    oneday = datetime.timedelta(days=1)
    test_date = datetime.datetime.strptime(str(datetime.datetime.now())[:10], "%Y-%m-%d")
    # 测试用
    # test_date = datetime.datetime.strptime('2021-01-30', "%Y-%m-%d")

    test_date = test_date - oneday
    min_date = test_date - interval
    test_date = str(test_date)[:10].replace('-','')
    min_date = str(min_date)[:10].replace('-','')
    # 调用cv_loader得到数据集
    cv_loader = CVDataLoader(test_date,min_date)

    # brand = 'JNBY'
    # labelname = 'design_empid'
    # if brand == 'JNBY':
    #     # 测试使用 读取rec
    #     # cv_loader.load_rec_data()
    #     cv_loader.load_box_data()
    #     cv_loader.load_plan_data()
    #     cv_loader.data_combine()
    #     logger.info('placeholder')
    # 读取预存相似度
    # with open('/root/lyh/cv2/data/'+brand+'_'+labelname+'_sim_dict.pickle', 'rb') as fp:   #Pickling
    #     sim_dict = pickle.load(fp)
    # with open('/root/lyh/cv2/data/'+brand+'_origin_final_pooling_sim_dict.pkl','rb') as fp:
    #     sim_dict = joblib.load(fp)
    with open('/root/workspace/lyh/image_feature/'+brand+'_skc_img_sim_dict.pkl','rb') as fp:
        sim_dict = joblib.load(fp)
    pro_list = list(sim_dict.keys())
    logger.info("sim_dict done")
    # 读取销售记录
    sale_dict, loop_num, check_max, most_dict = get_sale(brand, test_date, pro_list)
    logger.info("sale done")
    # 获取新品推荐池
    newbee = get_newbee(brand,check_max,test_date,pro_list)
    logger.info("newbee done")
    if len(newbee) == 0:
        logger.info("no newbee")
        rec_main = pd.read_parquet(os.getcwd() + '/d_rank/data/rank_result_' + brand + '.parquet')
        cols = ['unionid', 'm_product_id', 'm_productalias_id', 'score', 'name', 'value', 'price', 'brand', 'no',
                'imageurl']
        rec_main = rec_main[cols]
        # 主模型结果保存
        rec_main.sort_values(['unionid', 'score'], ascending=False, ignore_index=True)
        rec_main['isnew'] = False
        rec_main = rec_main.drop_duplicates(subset=['unionid', 'm_product_id'], keep='first')
        # 全部结果保存
        rec_main.sort_values(['unionid', 'score'], ascending=False, ignore_index=True)
        rec_main.to_parquet(os.getcwd() + '/n_cv_rec/data/cv_rec_all_' + brand + '.parquet')
        time_end = time.time()
        logger.info("finished n_cv_newrec:time cost: {}s".format(time_end - time_start))
        exit(0)
    # 初始化推荐结果使用的dict -> {'unionid':{'product':score}}
    rec_dict = {}
    for uid in sale_dict.keys():
        temp_dict = {}
        for i in range(len(newbee)):
            temp_dict[newbee[i]] = 0.0
        rec_dict[uid] = temp_dict
    # 遍历得到推荐分数
    for uid in rec_dict.keys():
        for i in range(loop_num):
            hist = sale_dict[uid][i]
            for aim in rec_dict[uid].keys():
                temp = sim_dict[hist][aim]
                if hist == aim:
                    temp = 0.0
                if temp > rec_dict[uid][aim]:
                    rec_dict[uid][aim] = temp
    logger.info("rec_score done")
    # 整理推荐结果
    u_col = list(rec_dict.keys())*len(newbee)
    u_col.sort()
    p_col = newbee*len(rec_dict)
    rec_df = pd.DataFrame(zip(u_col,p_col),columns=['unionid','product'])
    rec_df['score'] = rec_df.apply(lambda x: rec_dict[x['unionid']][x['product']],axis=1)
    rec_df = rec_df[rec_df['score']!=0.0]
    rec_df = rec_df.sort_values(['unionid','score'],ascending=False,ignore_index=True)
    rec_df['m_product_id'] = rec_df['product'].map(lambda x: x[:-4])
    rec_df['colors_code'] = rec_df['product'].map(lambda x: x[-3:])
    rec_df['m_product_id'] = rec_df['m_product_id'].astype('int')
    # 过滤购买过的商品
    rec_df = filter_purchased(rec_df,brand)
    logger.info("filter hist done")
    # 库存调整分数
    rec_df = plan_fix_score(rec_df)
    logger.info("filter plan done")
    # 每款只推一色
    rec_df = rec_df.drop_duplicates(subset=['unionid','m_product_id'],keep = 'first')
    logger.info("skc done")
    # 推荐多少件新品
    # rec_df = rec_df.groupby('unionid').head(10)
    # 找到最后sku
    rec_df = rec_df[['unionid','m_product_id','colors_code','score']]
    rec_df['sizes_code'] = rec_df['unionid'].map(lambda x: most_dict[x])
    attr = pd.read_parquet(os.getcwd()+'/a_load_data/data/cv_attr_sdf.parquet')
    attr = attr[attr['pro_brand'] == brand]
    attr = attr.rename(columns={'pro_brand':'brand','pricelist':'price'})
    attr = attr[['m_product_id','m_productalias_id','name','no','brand','value','price','imageurl','colors_code']]
    attr['sizes_code'] = attr['no'].map(lambda x: x[-2:])
    rec_df = fill_sku(rec_df,attr)
    logger.info("sku done")
    # 跟商品表merge得到最后的推荐结果
    rec_df = pd.merge(rec_df,attr[['m_productalias_id','name','value','price','brand','no','imageurl']],on='m_productalias_id')
    # 保留推荐结果表相同的列
    cols = ['unionid','m_product_id','m_productalias_id','score','name','value','price','brand','no','imageurl']
    rec_df = rec_df[cols]
    # 新品推荐保存
    rec_df = rec_df.sort_values(['unionid','score'],ascending=False,ignore_index = True)
    rec_df.to_parquet(os.getcwd()+'/n_cv_rec/data/cv_rec_new_'+brand+'.parquet')
    # upload_rec_res(rec_df,brand,new_table)
    upload_rec_res_loop(rec_df,brand,new_table)
    # 新品混老品
    # 测试使用
    # rec_main = pd.read_parquet(os.getcwd()+'/a_load_data/data/cv_rec_sdf.parquet')
    # rec_main = rec_main[cols]
    # rec_main = rec_main[rec_main['brand'] == brand]
    # rec_main = rec_main.loc[:1000000,:]
    # 正式使用
    rec_main = pd.read_parquet(os.getcwd()+'/d_rank/data/rank_result_'+brand+'.parquet')
    rec_main = rec_main[cols]
    # 主模型结果保存
    rec_main.sort_values(['unionid','score'],ascending=False,ignore_index = True)
    # upload_rec_res(rec_main,brand,main_table)
    upload_rec_res_loop(rec_main,brand,main_table)
    # 保留多少新品
    # rec_df = rec_df.groupby('unionid').head(10)
    top = rec_main['score'].max()
    bottom = 0
    # 调整新品分数
    rec_df['score'] = rec_df['score'].map(lambda x: (x-0/top-0)*0.7)
    rec_main['isnew'] = False
    rec_df['isnew'] = True
    rec_df = pd.concat([rec_main,rec_df])
    rec_df = rec_df.drop_duplicates(subset = ['unionid','m_product_id'],keep = 'first')
    # 全部结果保存
    rec_df.sort_values(['unionid','score'],ascending=False,ignore_index = True)
    rec_df.to_parquet(os.getcwd()+'/n_cv_rec/data/cv_rec_all_'+brand+'.parquet')
    # upload_rec_res(rec_df,brand,all_table)
    # upload_rec_res_loop(rec_df,brand,all_table)

    time_end=time.time()
    logger.info("finished n_cv_newrec:time cost: {}s".format(time_end-time_start))