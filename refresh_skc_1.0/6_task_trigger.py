import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import pickle
import json
import datetime
import time
from tqdm import tqdm, trange
import re

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# 下游任务名称
task_list = ['matting_image','outfit_compat','pro_retrival','new_pro_sim']
# 下游任务对应监控的图片集
task_mapping = {'outfit_compat':['skc'],'pro_retrival':['skc','wx_spu'],
                'new_pro_sim':['skc'],'matting_image':['skc']}
# 下游任务对应监控的更新信息文件
data_mapping = {'skc':'2_ckpt.json','wx_spu':'3_ckpt.json','box_skc':'4_ckpt.json'}
# 图片更新数执行任务参数
threshold_mapping = {'outfit_compat':0,'pro_retrival':0,'new_pro_sim':0,'matting_image':0}
# 下游任务脚本
execute_mapping = {'outfit_compat':['cd /home/administrator/lyh/transfer_learning/outfit_compat_1.0/\
                    && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                    0_update_data.py > 0_update_data.log',
                                    'cd /home/administrator/lyh/transfer_learning/outfit_compat_1.0/\
                    && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                    3_y_torch_gennet.py > 3_y_torch_gennet.log',
                                    'cd /home/administrator/lyh/transfer_learning/outfit_compat_1.0/\
                    && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                    0_sync_data2pro.py > 0_sync_data2pro.log'],
                    #                 'cd /home/administrator/lyh/transfer_learning/outfit_compat_1.0/\
                    # && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                    # 0_request_reload.py > 0_request_reload.log'],
                   # 'pro_retrival': ['python test.py'],
                   'pro_retrival': ['cd /home/administrator/lyh/unet/pro_retrival_2.0/\
                   && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                   0_save_by_cate.py > 0_save_by_cate.log',
                                    'cd /home/administrator/lyh/unet/pro_retrival_2.0/\
                   && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                   0_load4wx.py > 0_load4wx.log',
                                    'cd /home/administrator/lyh/unet/pro_retrival_2.0/\
                   && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                   1_gen_dataset.py > 1_gen_dataset.log',
                                    'cd /home/administrator/lyh/unet/pro_retrival_2.0/\
                   && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                   3_gen_feat_dict.py --ishard > 3_gen_feat_dict.log',
                                    'cd /home/administrator/lyh/unet/pro_retrival_2.0/\
                   && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                   3_gen_baseline.py > 3_gen_baseline.log',
                                    'cd /home/administrator/lyh/unet/pro_retrival_2.0/\
                   && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                   3_gen4wx.py --ishard > 3_gen4wx_ishard.log',
                                    'cd /home/administrator/lyh/unet/pro_retrival_2.0/\
                   && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                   3_gen4wx.py --baseline > 3_gen4wx_baseline.log'],
                   #                  'cd /home/administrator/lyh/unet/pro_retrival_2.0/\
                   # && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                   # 0_sync_data.py > 0_sync_data.log'],
                   'new_pro_sim': ['cd /home/administrator/lyh/transfer_learning/MyDataset/\
                    && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                    0_gen_dataset.py > 0_gen_dataset.log',
                                   'cd /home/administrator/lyh/transfer_learning/MyDataset/\
                    && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                    2_gen_feat_n_sim.py > 2_gen_feat_n_sim.log',
                                   'cd /home/administrator/lyh/transfer_learning/MyDataset/\
                    && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                    3_sync_feat_n_sim.py > 3_sync_feat_n_sim.log'],
                    'matting_image':['cd /home/administrator/Dataset/matting_image/\
                    && /home/administrator/anaconda3/envs/unet/bin/python3.7 \
                    matting_image.py > matting_image.log']
                   }

# 执行更新后, 记录时间
def execute_trigger(ckpt,task_name,test_date):
    sentences = execute_mapping[task_name]
    try:
        for sentence in sentences:
            os.system(sentence)
    except:
        logger.info('update error: task_name {} test_date {}'.format(task_name,test_date))
    for ck in ckpt:
        if ck['task_name'] == task_name:
            ck['execute_date'] = test_date
    return ckpt

# 读取上次执行任务时间, 读取更新数据, 判断是否执行更新
def check_last_trigger(ckpt,task_name,test_date,threshold = 100):
    # 默认时间
    last_date = '2021-06-01'
    # 获取上次更新时间
    for ck in ckpt:
        if ck['task_name'] == task_name:
            last_date = ck['execute_date']
    # 获取需要检查的数据源
    check_list = task_mapping[task_name]
    # 统计上次执行以后数据源更新数量
    count = 0
    for c in check_list:
        data_file = data_mapping[c]
        if os.path.exists(data_file):
            with open(data_file, 'rb') as fp:
                data = json.load(fp)
            for d in data:
                if d['update_time'] > last_date:
                    count += d['update_count']
        else:
            continue
    # 更新图片数量大于设定更新阈值时触发
    if count >= threshold:
        return True
    return False

if __name__ == '__main__':
    logger.info('start task trigger')
    time_start = time.time()
    test_date = datetime.datetime.strptime(str(datetime.datetime.now())[:10], "%Y-%m-%d")
    test_date = str(test_date)[:10]
    # 判断是否有历史记录
    if os.path.exists('6_last_execute.json'):
        with open('6_last_execute.json', 'rb') as fp:
            ckpt = json.load(fp)
        logger.info('before {}'.format(ckpt))
    else:
        ckpt = [{'task_name': 'outfit_compat', 'execute_date': test_date},\
                {'task_name': 'pro_retrival', 'execute_date': test_date},\
                {'task_name': 'new_pro_sim', 'execute_date': test_date},\
                {'task_name': 'matting_images', 'execute_date': test_date}]
        with open('6_last_execute.json', 'w') as json_file:
            json.dump(ckpt, json_file, ensure_ascii=False)
        logger.info('new: {}'.format(ckpt))

    for task_name in task_list:
        threshold = threshold_mapping[task_name]
        if check_last_trigger(ckpt, task_name, test_date, threshold):
            ckpt = execute_trigger(ckpt, task_name, test_date)
            logger.info('execute {} at {}'.format(task_name, test_date))
        else:
            logger.info('keep {} at {}'.format(task_name, test_date))
            continue
    with open('6_last_execute.json', 'w') as json_file:
        json.dump(ckpt, json_file, ensure_ascii=False)

    time_end = time.time()
    logger.info('6_task_trigger finished in {}s'.format(time_end - time_start))

    if os.path.exists(os.getcwd()+'/running.txt'):
        os.system('rm running.txt')

    pass