import os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def set_brand_other(x):
    if x['brand'] == 'CROQUIS':
        x['wxopenid'] = 'rec_2504948039'
    elif x['brand'] == 'JNBY':
        x['wxopenid'] = 'rec_2738574294'
    elif x['brand'] == 'LESS':
        x['wxopenid'] = 'rec_2822095692'
    elif x['brand'] == 'tjnby':
        x['wxopenid'] = 'rec_4'
    return x

other_dict = {'JNBY':'rec_2738574294',
              'LESS':'rec_2822095692',
              'CROQUIS':'rec_2504948039',
              'tjnby':'rec_4'}


# 处理年份
def f_year(x,year,month):
    if x['pro_year'] > year and (12  - int(month) + int(x['pro_band'][:2]))>6:
        x['pro_year'] = str(int(x['pro_year'])-1)
    elif x['pro_year'] == year and (int(x['pro_band'][:2]) - int(month))>6:
        x['pro_year'] = str(int(x['pro_year'])-1)
    return x['pro_year']

# 用户无销售记录, 或者无主码, 用均码填充most_dict
def fix_dict(unionid_list, most_dict):
    for uid in unionid_list:
        if uid not in most_dict.keys():
            most_dict[uid] = '03'
    return most_dict

# 得到销售数据
def get_sale(brand, test_date, pro_list):
    sale = pd.read_parquet(os.getcwd()+'/a_load_data/data/cv_sale_sdf.parquet')
    sale = sale[sale['pro_brand'] == brand]
    sale['pro_year'] = sale.apply(lambda x: f_year(x,test_date[:4],test_date[4:6]),axis=1)
    # 是否要求商品有图片, 若有图片, 则需要drop图片为空的销售记录
    sale = sale.dropna(subset=['imageurl'])
    sale = sale.sort_values('billdate',ascending = False)
    # 计算最新波段
    # sale['design_year'] = sale['design_year'].astype('str')
    # check_max = (sale['design_year']+sale['pro_band']).max()
    check_max = test_date[:7]
    # 获得推荐结果, 每个小类保留一个最高分
    # 测试使用
    # rec = pd.read_parquet(os.getcwd()+'/a_load_data/data/cv_rec_sdf.parquet')
    # rec = rec[rec['brand'] == brand]
    # 正式使用
    rec = pd.read_parquet(os.getcwd()+'/d_rank/data/rank_result_'+brand+'.parquet')
    rec = rec[rec['unionid']!=other_dict[brand]]
    if len(rec) == 0:
        logger.info('empty rec!')
        raise ValueError("rec")
    rec['colors_code'] = rec['no'].map(lambda x: x[-5:-2])
    rec = rec[['unionid','m_product_id','m_productalias_id','colors_code','score']]
    attr = pd.read_parquet(os.getcwd()+'/a_load_data/data/cv_attr_sdf.parquet')
    attr = attr[['m_product_id','pro_small_type']].drop_duplicates(subset = ['m_product_id'],keep='first')
    type_dict = dict(zip(attr['m_product_id'],attr['pro_small_type']))
    rec = rec[rec['m_product_id'].isin(type_dict.keys())]
    rec['pro_small_type'] = rec['m_product_id'].map(lambda x: type_dict[x])
    rec = rec.sort_values(['unionid','pro_small_type','score'],ascending=False)
    rec = rec.drop_duplicates(subset=['unionid','pro_small_type'],keep = 'first')
    # 推荐结果过滤用户
    unionid_list = list(rec['unionid'].unique())
    sale = sale[sale['unionid'].isin(unionid_list)]
    if len(sale) == 0:
        logger.info('empty sale!')
        raise ValueError("sale")
    # 用户购买的主码most_dict
    sale['most'] = sale['no'].map(lambda x: x[-2:])
    most = sale.groupby(['unionid','most'])['m_product_id'].count().reset_index()
    most = most.sort_values(['unionid','m_product_id'],ascending = False).drop_duplicates(subset=['unionid'],keep='first')
    most = most[most['m_product_id']>=3]
    most_dict = dict(zip(most.loc[:,'unionid'], most.loc[:,'most']))
    most_dict = fix_dict(unionid_list, most_dict)
    # 根据销售将待拼接推荐结果根据small_type过滤
    sale_type_dict = dict(sale.groupby('unionid').apply(lambda x: list(set(x['pro_small_type']))))
    # 选择不填充的小类
    dont_used = ['连体衣','泳装','内衣','皮衣']
    temp_ids = sale_type_dict.keys()
    for uid in unionid_list:
        if uid not in temp_ids:
            sale_type_dict[uid] = dont_used
        else:
            sale_type_dict[uid] = sale_type_dict[uid] + dont_used
    rec['pro_small_type'] = rec.apply(lambda x: x['pro_small_type'] \
        if x['pro_small_type'] not in sale_type_dict[x['unionid']] \
        else 'had',axis=1)
    rec = rec[rec['pro_small_type'] != 'had']
    # 推荐结果拼接
    sale = pd.concat([sale[['unionid','m_product_id','m_productalias_id','colors_code']],
                      rec[['unionid','m_product_id','m_productalias_id','colors_code']]])
    # 保留hist为了观察效果
    # hist = sale.copy()
    # 得到skc列
    # sale['product'] = sale['m_product_id'].astype('str') + +sale['colors_code']
    sale['product'] = sale.apply(lambda x: str(x['m_product_id']) + \
                                           '_' + x['colors_code'],axis=1)

    # skc列表过滤
    sale =sale[sale['product'].isin(pro_list)]
    if len(sale) == 0:
        logger.info('empty sale pro images!')
        raise ValueError("sale with images")
    # 用户购买记录, 用于计算相似度推荐
    sale = sale.groupby('unionid').apply(lambda x: list(x['product'])[:10]).reset_index()
    sale = sale.rename(columns = {0:'hist'})
    loop_num = sale['hist'].map(lambda x: len(x)).min()
    sale_dict = dict(zip(sale['unionid'],sale['hist']))
    return sale_dict, loop_num, check_max, most_dict

# 获得用来推荐的新品list
def get_newbee(brand, check_max, test_date, pro_list):
    # 使用box中的新品作为推荐商品池
    box = pd.read_parquet(os.getcwd()+'/a_load_data/data/cv_box_sdf.parquet')
    box = box[box['brand']==brand]
    attr = pd.read_parquet(os.getcwd()+'/a_load_data/data/cv_attr_sdf.parquet')
    # 商品表关联得到商品信息
    attr = attr[attr['pro_brand'] == brand]
    attr['pro_year'] = attr.apply(lambda x: f_year(x,test_date[:4],test_date[4:6]),axis=1)
    attr['m_product_id'] = attr['m_product_id'].astype('str')
    # 用box商品池过滤
    # newbee = pd.merge(attr, box[['m_product_id']].astype('str'), on = ['m_product_id'])
    # 不用box商品池过滤
    newbee = attr.copy()
    # 去掉无图商品
    newbee = newbee.dropna(subset=['imageurl'])
    # 根据销售数据和波段判断新品
    newbee['check_band'] = newbee['pro_year']+newbee['pro_band']
    newbee = newbee[newbee['check_band']>=check_max]
    newbee = newbee.drop(columns=['m_productalias_id'])
    # 得到skc列
    newbee['product'] = newbee['m_product_id'].astype('str')+newbee['colors_code']
    # 对于颜色去重
    newbee = newbee.drop_duplicates(subset=['product'],keep='first')
    # skc列表过滤
    newbee =newbee[newbee['product'].isin(pro_list)]
    return list(newbee['product'])

# 过滤购买过的商品
def filter_purchased(rec_df,brand):
    sale = pd.read_parquet(os.getcwd()+'/a_load_data/data/cv_sale_sdf.parquet')
    sale = sale[sale['pro_brand'] == brand]
    sale = sale.dropna(subset=['imageurl'])
    sale = sale[['unionid','m_product_id']]
    sale = sale[sale['unionid'].isin(list(rec_df['unionid'].unique()))]
    purchased = pd.merge(rec_df,sale,on = ['unionid','m_product_id'])
    rec_df = pd.concat([rec_df,purchased])
    rec_df = rec_df.drop_duplicates(subset=['unionid','product'],keep = False)
    return rec_df

# 产量修正得分
def plan_fix_score(rec_df):
    # 备份, 防止无产量
    backup = rec_df.copy()
    # 产量所谓商品评价
    plan = pd.read_parquet(os.getcwd()+'/a_load_data/data/cv_plan_sdf.parquet')
    plan['product'] = plan['m_product_id'].astype('str')+plan['colorid']
    nums_max = plan['plan_nums'].max()
    plan = plan[plan['plan_nums']!=0]
    plan['ratio'] = plan['plan_nums'].map(lambda x: x/nums_max)
    plan_dict = dict(zip(plan['product'],plan['ratio']))
    # 保留有产量的推荐结果
    rec_df = rec_df[rec_df['product'].isin(plan_dict.keys())]
    rec_df['score'] = rec_df.apply(lambda x: x['score']*plan_dict[x['product']],axis=1)
    # 标准化score
    score_max = rec_df['score'].max()
    if np.isnan(score_max):
        logger.info('reset rec, cuz no rec after plan filter!')
        return backup
    else:
        rec_df['score'] = rec_df['score'].map(lambda x: x/score_max)
    return rec_df

# 缺失条码sku填充(若主码无商品, 同skc任意填充)
def fill_sku(rec_df,attr):
    fill_front = attr[['m_product_id','colors_code','sizes_code','m_productalias_id']]
    fill_back = fill_front.drop_duplicates(subset=['m_product_id','colors_code'],keep = 'first')
    def f_fill(x):
        if x['m_productalias_id'] == 0:
            return fill_back.loc[(fill_back['m_product_id'] == x['m_product_id'])& \
                                 (fill_back['colors_code'] == x['colors_code']),'m_productalias_id'].values[0]
        return x['m_productalias_id']
    # 缺失尺码填充(若主码无商品, 同skc任意填充)
    rec_df = pd.merge(rec_df,fill_front,on = ['m_product_id','colors_code','sizes_code'], how = 'left')
    rec_df['m_productalias_id'] = rec_df['m_productalias_id'].fillna(0.0)
    rec_df['m_productalias_id'] = rec_df['m_productalias_id'].astype('int')
    rec_df['m_productalias_id'] = rec_df.apply(lambda x: f_fill(x),axis=1)
    return rec_df