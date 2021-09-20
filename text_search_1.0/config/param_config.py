
# 渠道
channel_dict = {
    0:'微商城',
}

# 为区分product_name
# initial_words = ['短袖','中长袖','中袖','长袖','长款','中款','中长款','短款','小短款','超长款','套头','开襟',
#                  '戴帽领','V领','A型','H型']
with open('data/initial_words.txt','r') as f:
    initial_words = f.readlines()
initial_words = [d.strip() for d in initial_words if d.strip() != '']

# 品牌
brand_list = ['jnby', 'less', 'croquis', 'tjnby', 'jnbyh', 'pomme', 'apn', 'samo', 'lasumin']
# 召回字段
# ,'product_name' 改用process_product_name
recall_cols = ['brand','brand_name','pro_brand','big_class','small_class','product_name']
# 排序字段
rank_cols = ['year','abbr_year','big_season', 'season', 'band', 'replace_band',
            'pro_category','scmatertype3','scmatertype1','scmatertype2',
            'coldeepshallow','colclass','clr_description','color_name',
            'style_label_two','integral_line','tg','tl','tx','topic',]
            # 'mall_title','bomname',
# 分词和倒排索引需要删除的词
bad_words = ['【','】','—','--',' ','/','-','）','（','(',')','','unknown']
# 删除词加入停用词
bad_words += [line.strip() for line in open('data/stopwords.txt', 'r', encoding='utf-8').readlines()]


# 微商城品牌包含的品牌
brand_mapping = {
    'jnby' : ['jnby'], 'less': ['less'], 'croquis': ['croquis'], 'tjnby': ['tjnby'],
    'pomme': ['pomme'], 'apn': ['apn'], 'jnbyh': ['jnbyh'],
    'samo': ['croquis', 'jnby', 'less', 'tjnby', 'pomme', 'apn'], # 'jnbyh',
    'lasumin': ['croquis', 'jnby', 'less', 'tjnby', 'pomme', 'apn'] # 'jnbyh',
}
# attr_df用到的字段
attr_cols = ['m_productalias_id', 'm_product_id','intscode', 'name', 'skc', 'no', 'pro_brand',
           'pro_year', 'pro_band', 'pro_big_season', 'pro_small_season',
           'pro_big_type', 'pro_small_type', 'value', 'code', 'scmatertype1',
           'scmatertype2', 'scmatertype3', 'bomcode', 'bomname', 'pro_category',
           'colors_code', 'colors', 'clr_description', 'colclass', 'colsystem',
           'coldeepshallow', 'pro_topic', 'style_label_two', 'design_empid',
           'integral_line', 'tg', 'tl', 'tx', 'pricelist', 'imageurl']
# skc_df用到的字段
skc_cols = ['name', 'skc', 'product_name', 'brand', 'brand_name', 'year',
           'big_season', 'season', 'big_class', 'small_class', 'band', 'topic',
           'price', 'mall_title', 'color_code', 'color_name','mall_image',
            'after_band','sort_col']
# mall_title替换词
special_words = ['江南布衣','速写','蓬马','JNBY','CROQUIS',
                 'HOME','jnbybyJNBY','LESS','pomme','less','APN73',
                 '*','(',')',',','/','[',']','【','】','（','）','，','-','—']

# 加入商品属性词进入词典
word_cols = ['name','brand','brand_name','pro_brand','year','abbr_year',
            'big_season', 'season', 'band', 'replace_band',
            'big_class','small_class', 'product_name',
            'pro_category','scmatertype3','scmatertype1','scmatertype2',
            'color_name','clr_description','colclass','colsystem','coldeepshallow',
            'style_label_two','integral_line','tg','tl','tx']
            # 'topic','bomname','mall_title','product_name',