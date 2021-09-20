# Introduction
     该项目用于
        1. 构造分类模型数据集
        2. 分类模型训练
        3. 分类模型生成skc图像特征矩阵及分品牌skc图像相似度
# Content
    0_gen_dataset.py
        构造数据集, 拆分train|valid|test数据集
    1_train_one_label_clf.py
        指定参数训练单标签分类模型
    1_train_multi_label_clf.py
        指定参数训练多标签分类模型
    1_train_multi_task_clf.py
        指定参数训练多任务分类模型
    2_gen_feat_n_sim.py
        指定参数生成skc图像特征矩阵及分品牌skc图像相似度
