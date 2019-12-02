# coding=utf-8
test_pub_path = '../data/sna_data/sna_valid_pub.json'
test_author_path = '../data/sna_data/sna_valid_author_raw.json'
test_example_evaluation_data_path = '../data/sna_data/sna_valid_example_evaluation_scratch.json'

train_pub_path = '../data/train/train_pub.json'
train_author_path = '../data/train/train_author.json'


origin_data = '../my_data/origin_data.txt'  # 文章原始数据
test_origin_data = '../my_data/origin_test_data.txt'  # 文章原始数据-测试
true_origin_data = '../my_data/origin_true_data.txt'  # 文章原始数据-评测


author_feature = '../my_data/author_feature.txt'  # 文章作者特征
author_feature_map_pkl = '../my_data/author_feature_map.pkl'  # 特征映射表


keyword_feature = '../my_data/keyword_feature.txt'  # 关键词特征
keyword_feature_map_pkl = '../my_data/keyword_feature_map.pkl'  # 特征映射表


content_data = '../my_data/content.txt'  # 内容文件
content_test_data = '../my_data/content_test.txt'  # 测试内容文件


# pre_train_embedding = '../my_data/pre_train_embedding.txt'  # 词向量
pre_train_embedding = '../my_data/pre_train_embedding_v2.txt'  # 词向量


# pre_train_paper_embedding = '../my_data/train_paper_embedding.txt'  # 预训练文章
pre_train_paper_embedding = '../my_data/train_paper_embedding_v2.txt'  # 预训练文章
pre_train_test_paper_embedding = '../my_data/train_paper_test_embedding_v2.txt'  # 预训练测试文章
pre_train_true_paper_embedding = '../my_data/train_paper_true_embedding_v2.txt'


trained_paper_embedding = '../my_data/trained_paper_embedding.txt'  # 训练完成的文章
trained_test_paper_embedding = '../my_data/trained_test_paper_embedding.txt'  # 训练完成的文章
trained_true_paper_embedding = '../my_data/trained_true_paper_embedding.txt'

cluster_model = '../my_data/cluster_model.pkl'  # 聚类模型

true_pub_path = '../data/sna_test_data/test_pub_sna.json'
true_author_path = '../data/sna_test_data/sna_test_author_raw.json'
true_example_evaluation_data_path = '../data/sna_test_data/sna_valid_example_evaluation_scratch.json'

max_sentence_len = 1000
embedding_dim = 100
author_feature_map_len = 1000000
keyword_feature_map_len = 700000
train_paper_size = 203184
