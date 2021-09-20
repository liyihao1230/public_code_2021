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
import time
import datetime
import argparse

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

from s_size_classify.tools.config import user_feat_cols, feat_cols
from s_size_classify.tools.data_process import get_dataset, wash_user, get_pro_index_feat_sim_dict, generate_feat_df
from s_size_classify.tools.data_process import get_integer_dict
from s_size_classify.tools.data_process import encode_decode, generate_dataset, get_pred_decode_dict, get_res_pdf


import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from tqdm import tqdm, trange

conf = SparkConf()
conf.set("spark.pyspark.python", PYTHON_PATH) \
    .set("spark.pyspark.driver.python", PYTHON_PATH) \
    .set("spark.executorEnv.PYSPARK_PYTHON", PYTHON_PATH) \
    .set("spark.executorEnv.PYSPARK_DRIVER_PYTHON", PYTHON_PATH) \
    .set("spark.yarn.appMasterEnv.PYSPARK_PYTHON", PYTHON_PATH) \
    .set("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON", PYTHON_PATH) \
    .set("spark.sql.execution.arrow.enabled", "%tbrue") \
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

# 模型
lgbm =lgb.LGBMRegressor(objective='regression',silent = False,
                                  num_leaves = 2**5-1,
#                                   subsample = 0.9,colsample_bytree = 0.9,
                                  n_estimators = 5000,learning_rate = 0.05,
                                  min_child_samples = 20,
#                                   gpu_device_id = 2,gpu_platform_id = 0,
#                                   device = 'gpu',gpu_use_dp = True, max_bin = 128
                                )

if __name__ == '__main__':
    # 设置脚本参数
    parser = argparse.ArgumentParser()
    parser.add_argument("brand", help="the brand we want to use")
    args = parser.parse_args()
    brand = args.brand
    # 1. 数据读取
    interval = datetime.timedelta(days=550)
    oneday = datetime.timedelta(days=1)
    test_date = datetime.datetime.strptime(str(datetime.datetime.now())[:10], "%Y-%m-%d")
    test_date = test_date - oneday
    min_date = test_date - interval
    test_date = str(test_date)[:10].replace('-','')
    min_date = str(min_date)[:10].replace('-','')

    time_total = time.time()

    train_pdf, val_pdf, test_pdf = get_dataset(brand,test_date)
    encode_dict, decode_dict = encode_decode(brand)

    # 用户清洗
    train_pdf = wash_user(train_pdf,encode_dict)
    # val_pdf = wash_user(val_pdf,encode_dict)
    # test_pdf = wash_user(test_pdf,encode_dict)
    # 获得回归标签
    train_pdf['label'] = train_pdf['label'].map(lambda x:encode_dict[x])
    val_pdf['label'] = val_pdf['label'].map(lambda x:encode_dict[x])
    test_pdf['label'] = test_pdf['label'].map(lambda x:encode_dict[x])
    # 此处加入用户特征到记录里
    user_history_dict = dict(train_pdf.sort_values('billdate',ascending = False).groupby('unionid').\
                         apply(lambda x: list([list(x[col]) for col in ['billdate','label','m_product_id','pro_big_type']])))
    # 用户index
    user_index = get_integer_dict('unionid',train_pdf)
    # 商品index
    # product_index = get_integer_dict('m_product_id',train_pdf)
    product_index, feat_dict, sim_dict = get_pro_index_feat_sim_dict(brand, feat_cols, train_pdf)

    time_start = time.time()
    # 删除每个用户的最早的记录
    temp_pdf = train_pdf.groupby('unionid')['billdate'].min().reset_index()
    train_pdf = train_pdf.drop(pd.merge(train_pdf,temp_pdf,on=['unionid','billdate'],right_index = True).index)
    train_pdf = generate_feat_df(train_pdf,user_history_dict)
    time_end = time.time()
    logger.info("build train_pdf time cost: {}s".format(time_end-time_start))

    time_start = time.time()
    val_pdf = val_pdf[val_pdf['unionid'].isin(user_index.keys())]
    val_pdf = val_pdf[val_pdf['m_product_id'].isin(product_index.keys())]
    val_pdf = generate_feat_df(val_pdf,user_history_dict)
    time_end = time.time()
    logger.info("build val_pdf time cost: {}s".format(time_end-time_start))

    time_start = time.time()
    test_pdf = test_pdf[test_pdf['unionid'].isin(user_index.keys())]
    test_pdf = test_pdf[test_pdf['m_product_id'].isin(product_index.keys())]
    test_pdf = generate_feat_df(test_pdf,user_history_dict)
    time_end = time.time()
    logger.info("build test_pdf time cost: {}s".format(time_end-time_start))

    time_start = time.time()
    # # 去掉lastpro不在sim_dict矩阵中的记录
    # train_pdf = train_pdf[train_pdf['lastpro'].isin(product_index.keys())]
    # val_pdf = val_pdf[val_pdf['lastpro'].isin(product_index.keys())]
    # test_pdf = test_pdf[test_pdf['lastpro'].isin(product_index.keys())]
    # for c in feat_cols+['all']:
    for c in ['all']:
        train_pdf['sim_'+c] = train_pdf.apply(lambda x: sim_dict[c][product_index[x['m_product_id']],product_index[x['lastpro']]],axis=1)
        val_pdf['sim_'+c] = val_pdf.apply(lambda x: sim_dict[c][product_index[x['m_product_id']],product_index[x['lastpro']]],axis=1)
        test_pdf['sim_'+c] = test_pdf.apply(lambda x: sim_dict[c][product_index[x['m_product_id']],product_index[x['lastpro']]],axis=1)
    time_end = time.time()
    logger.info("generate all sim time cost: {}s".format(time_end-time_start))
    X_train,y_train = generate_dataset(train_pdf, product_index, user_index, feat_cols, user_feat_cols, feat_dict)
    X_val,y_val = generate_dataset(val_pdf, product_index, user_index, feat_cols, user_feat_cols, feat_dict)
    X_test,y_test = generate_dataset(test_pdf, product_index, user_index, feat_cols, user_feat_cols, feat_dict)
    # 模型训练
    lgbm.fit(X_train,y_train,early_stopping_rounds=200,eval_set = [(X_train,y_train),(X_val,y_val)],eval_metric = ['l1'],verbose = 50)
    # 模型评估
    result = lgbm.predict(X_test)
    pred_decode_dict = get_pred_decode_dict(product_index, encode_dict, brand)
    func1 = lambda x: pred_decode_dict[x['product_index']][np.argmin(abs(x['y_pred'] - pred_decode_dict[x['product_index']]))]
    func2 = lambda x: decode_dict[x]
    res_pdf = get_res_pdf(X_test,result,func1,func2)
    # 效果评估
    res_pdf['y_true'] = y_test
    res_pdf['y_true'] = res_pdf['y_true'].map(func2)
    acc = len(res_pdf[res_pdf['y_pred'] == res_pdf['y_true']])/len(res_pdf)
    logger.info("The accuracy for test dataset is {}".format(acc))
    # 预测尺码与目标尺码相差一
    res_pdf['new_pred'] = res_pdf['y_pred'].map(lambda x: int(x))
    res_pdf['new_true'] = res_pdf['y_true'].map(lambda x: int(x))
    dev_1 = len(res_pdf[abs(res_pdf['new_pred']-res_pdf['new_true'])<=1])/len(res_pdf)
    logger.info("The deviation not larger than one rate".format(dev_1))
    time_total_end = time.time()
    logger.info("total time cost: {}s".format(time_total_end-time_total))

    with open(os.getcwd()+'/s_size_classify/data/lgbm_'+brand+'.pkl','wb') as f:
        joblib.dump(lgbm,f)
    with open(os.getcwd()+'/s_size_classify/data/user_history_dict_'+brand+'.pkl','wb') as f:
        joblib.dump(user_history_dict,f)
    with open(os.getcwd()+'/s_size_classify/data/feat_dict_'+brand+'.pkl','wb') as f:
        joblib.dump(feat_dict,f)
    with open(os.getcwd()+'/s_size_classify/data/sim_dict_'+brand+'.pkl','wb') as f:
        joblib.dump(sim_dict,f)
    with open(os.getcwd()+'/s_size_classify/data/user_index_'+brand+'.pkl','wb') as f:
        joblib.dump(user_index,f)
    with open(os.getcwd()+'/s_size_classify/data/product_index_'+brand+'.pkl','wb') as f:
        joblib.dump(product_index,f)
    logger.info("save data pickle successfully!")