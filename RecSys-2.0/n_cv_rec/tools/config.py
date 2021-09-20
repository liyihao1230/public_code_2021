#!/usr/bin/env python
# encoding: utf-8
"""
@AUTHOR: cuizhengliang
@LICENCE: (C)Copyright 2013-2020, JNBY+ Corporation Limited
@CONTACT: czlwork@qq.com
@FILE: config.py
@TIME: 2020/12/18 9:24
@DESC:
"""

sale_fields = [
    "retailid",             # 零售自增ID，代表小票唯一ID
    "itemid",
    "billdate",
    "m_productalias_id",
    "tot_qty",
    "c_vip_id",
    "unionid",
    "tot_amt_list",
    "discount",
    "qty",
    "brand_nature",
    "default05",
    # 消费主体门店及地理位置
    "c_store_id",
    "province",
    "city",
]

item_fields = [
    "m_productalias_id",
    "no",
    "m_product_id",
    "value",
    "pro_band",
    "pro_brand",
    "pro_year",
    "pro_big_season",
    "pro_big_type",
    "pro_small_type",
    "pro_category",
    "pricelist",
    "colors",
    "colors_code",
    "pro_attr",
    "style_label_three",
    "style_label_two",
    "style_label_one",
    "fabelement",
    "bomname",
    "bomcode"
]

big_type_list = [
    '上装', '下装', '通身', '家居服饰',
    '连衣裙', '外套','可内可外',
    '套装', '内配',
    # '饰品', '家居用品', '内衣', '泳装',
    # '卧室用品', '浴室用品', '书房用品', '厨房用品', '户外用品', '家具', '收纳用品',
    # 'POS类', '建材类', '标识类', '设备类', '陈列类', '软道类', '辅料', '功能类', '辅件类', '其它', '礼品类'
]

# item_brand = {
#     'JNBY美版': 'JNBY', 'JNBY': 'JNBY', 'JNBY饰品': 'JNBY', 'JNBY工服': 'JNBY', 'JNBY-JJ': 'JNBY', 'JNBY欧版': 'JNBY',
#     '婴童饰品': 'tjnby', '童装饰品': 'tjnby', '婴童': 'tjnby', '童装': 'tjnby', '童装工服': 'tjnby', 'jnby': 'tjnby',
#     'LESS工服': 'LESS', 'LESS饰品': 'LESS', 'LESS': 'LESS',
#     'CROQUIS工服': 'CROQUIS', 'CROQUIS饰品': 'CROQUIS', 'CROQUIS': 'CROQUIS',
#
#     '蓬马工服': 'Pomme', '蓬马饰品': 'Pomme', '蓬马': 'Pomme',
#     'SAMO饰品': 'SAMO',
#     'LASU MIN SOLA饰品': 'LASU', 'LASU MIN SOLA': 'LASU',
#     'A PERSONAL NOTE 73饰品': 'APN', 'A PERSONAL NOTE 73': 'APN',
#     'SAMO': 'SAMO',
#     'JNBYHOME': 'JNBYHOME',
# }

item_brand = {
    'JNBY': 'JNBY',
    '婴童': 'tjnby', '童装': 'tjnby', '婴童内衣': 'tjnby',
    'LESS': 'LESS',
    'CROQUIS': 'CROQUIS',
}

'''
    used by s_size_classify
'''

item_style = {
    '暗黑':'个性', '酷':'个性', '玩味':'个性', '意趣':'个性', '大气':'个性',
    '轻松':'日常', '舒适':'日常',
    '时髦':'新潮', '运动':'新潮', '前卫':'新潮', '街头':'新潮', '夸张新潮':'新潮',
    '复古':'雅痞', '怀旧':'雅痞', '低调文艺':'雅痞', '日系':'雅痞',
    '都市':'简约', '绅士':'简约', '商务休闲':'简约', '利落':'简约', '精致':'简约', '极简':'简约',
}

attr_fields = [
    "m_productalias_id", "m_product_id", "pro_brand",
    "pro_year", "pro_band",
    "pro_big_type", "pro_small_type", "m_dim21_id",
    "mark_style", "design_empid","code", "bomcode",
    "colors_code", "name", "no", "pricelist", "value", "imageurl", "pt"
]

mbr_fields = [
    "c_vip_id", "unionid", "sex", "age",
    "c_store_id", "c_customer_id",
    "pt"
]

mkt_fields = [
    "m_productalias_id", "c_vip_id", "unionid",
    "qty", "billdate",
    "c_store_id", "province",
    "brand_nature","default05",
    "pt"
]

rec_fields = [
    "unionid", "m_product_id", "m_productalias_id",
    "score", "name", "value", "price", "brand",
    "no", "imageurl"
]

box_fields = [
    "m_product_id", "name", "value", "brand",
    "pricelist","imageurl","m_productalias_id"
]

plan_fields = [
            "m_productalias_id", "m_product_id",
            "colorid", "sizeid", "nums", "plansub_setdate", "pt"
]

qty_fields = [
    "m_product_id", "pro_brand",
    "sqty_7d", "sqty_15d", "sqty_1m",
    "cqty_7d", "cqty_15d", "cqty_1m"
]


brand_list = ['JNBY','LESS','CROQUIS','tjnby']

# no_code编码解码用
general_encode_dict = {
    '00':0.05, '01':0.15, '02':0.25, '03':0.35, '04':0.45, '05':0.55, '06':0.65, '07':0.75, '08':0.85, '09':0.95,
    '59':0.125,'60':0.175, '61':0.225, '62':0.275, '63':0.325, '64':0.375, '65':0.425, '66':0.475, '67':0.525,
    '10':0.04,'11':0.14,'12':0.24,'13':0.34,'14':0.44,'15':0.54,'16':0.64,
    '20':0.00,'21':0.006,'22':0.012,'23':0.018,'24':0.024,'25':0.03,'26':0.036,'27':0.042, '28':0.048, '29':0.054,
    '30':0.06, '31':0.16, '32':0.26, '33':0.36,'34':0.46, '35':0.56, '36':0.66,'37':0.76,'38':0.86
}
general_decode_dict = {value:key for key, value in general_encode_dict.items()}
# JNBY
jnby_encode_dict = {'00':1/11, '01':2/11, '02':3/11, '03':4/11, '04':5/11,
                    '05':6/11, '06':7/11, '07':8/11, '08':9/11, '09':10/11}
jnby_decode_dict = {value:key for key, value in jnby_encode_dict.items()}
# LESS
less_encode_dict = {'01':3/11, '02':4/11, '03':5/11, '04':6/11, '05':7/11, '06':8/11}
less_decode_dict = {value:key for key, value in less_encode_dict.items()}
# CROQUIS
croquis_encode_dict = {'00':3/11, '01':4/11, '02':5/11, '03':6/11, '04':7/11, '05':8/11, '06':9/11, '07':10/11,
                       '08':11/11, '59':5/11+1/44,'60':5/11+2/44, '61':6/11+1/44, '62':6/11+2/44, '63':7/11+1/44,
                       '64':7/11+2/44, '65':8/11+1/44,'66':8/11+2/44, '67':9/11+1/44}
croquis_decode_dict = {value:key for key, value in croquis_encode_dict.items()}
# tjnby
tjnby_encode_dict = {'00':1/17, '01':2/17, '02':3/17, '03':4/17, '04':5/17, '05':6/17, '06':7/17, '07':8/17, '08':9/17,
                     '09':10/17, '10':11/17,'11':12/17,'12':13/17,'13':14/17,'14':15/17,'15':16/17,'16':17/17,
                    '20':0.00,'21':0.01,'22':0.02,'23':0.03,'24':0.04,'25':0.05,'26':0.06,'27':0.07, '28':0.08,
                     '29':0.09, '30':0.1, '31':0.2, '32':0.3, '33':0.4,'34':0.5, '35':0.6, '36':0.7,'37':0.8,'38':0.9}
tjnby_decode_dict = {value:key for key, value in tjnby_encode_dict.items()}


