# Introduction
     该项目用于
        1. 定时检查是否有新商品存入数据库, 更新离线商品信息表
        2. 定时检查是否有新图片可获取, 更新离线存放图片集
        3. 更新微商城商品详情表和图
        4. 更新BOX素材库look表和图
        5. 更新订货会look表和图
        6. 更新监控和任务触发
        7. 每周刷存图, 将新加入图片去背景存放到根目录为Matted_Images的同目录下 (未启用)
# Content
    1_update_attr_set.py
        更新Dataset/SKC/attr.parquet
        更新Dataset/SKC/all_attr.parquet
    2_update_image_set.py
        更新Dataset/SKC中单品图
        2_ckpt.json记录图片更新信息
    3_update_wx_spu_set.py
        更新Dataset/WX_SPU/wx_spu_df.parquet
        更新Dataset/WX_SPU中轮播图
        3_ckpt.json记录图片更新信息
    4_update_box_look_set.py
        更新Dataset/BOX_SKC/box_skc_df.parquet
        更新Dataset/BOX_SKC中搭配图
        4_ckpt.json记录图片更新信息
    5_update_order_look_set.py
        更新Dataset/ORDER_SKC/order_skc_df.parquet
    6_task_trigger.py
        当前任务
            outfit_compat: [0_update_data.py, 3_y_torch_gennet.py]
            pro_retrival: [test.py]
            pro_sim: [test.py]
        监控1~5更新的信息, 触发下游任务
        日志记录更新操作
# Downstream Project Relation
    1. outfit_compat_1.0
        更新一定SKC图片执行
            0_update_data.py
            3_y_torch_gennet.py
        手动训练模型
    2. pro_retrival_2.0
        更新一定SKC图片执行gen_feat_dict
        更新一定WX_SPU图片执行gen_feat_dict
        手动训练模型
    3. new_pro_sim
        更新一定SKC图片执行