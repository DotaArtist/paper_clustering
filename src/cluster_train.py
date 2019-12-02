#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import json
import config
import copy
import numpy as np
import dill as pickle
from utils import data_utils
from sklearn.cluster import KMeans

train_emb_dict = data_utils.load_doc_emb(config.trained_paper_embedding)
# test_emb_dict = data_utils.load_doc_emb(config.trained_test_paper_embedding)
true_emb_dict = data_utils.load_doc_emb(config.trained_true_paper_embedding)

# test_pub = json.load(open(config.test_pub_path, mode='r', encoding='utf-8'))
# test_author = json.load(open(config.test_author_path, mode='r', encoding='utf-8'))

true_pub = json.load(open(config.true_pub_path, mode='r', encoding='utf-8'))
true_author = json.load(open(config.true_author_path, mode='r', encoding='utf-8'))

train_pub = json.load(open(config.train_pub_path, mode='r', encoding='utf-8'))
train_author = json.load(open(config.train_author_path, mode='r', encoding='utf-8'))

train_paper_set = set(train_pub.keys())
train_paper_list = list(train_paper_set)

# test_paper_set = set(test_pub.keys())
# test_paper_list = list(test_paper_set)

true_paper_set = set(true_pub.keys())
true_paper_list = list(true_paper_set)

train_paper_matrix = []
for i in train_paper_list:
    train_paper_matrix.append(train_emb_dict[i])
train_paper_matrix = np.array(train_paper_matrix)

# test_paper_matrix = []
# for i in test_paper_list:
#     test_paper_matrix.append(test_emb_dict[i])
# test_paper_matrix = np.array(test_paper_matrix)

true_paper_matrix = []
for i in true_paper_list:
    true_paper_matrix.append(true_emb_dict[i])
true_paper_matrix = np.array(true_paper_matrix)

model = KMeans(n_clusters=8, random_state=0)
model.fit(train_paper_matrix)
print(model.inertia_/len(train_paper_list))

# pickle.dump(model, open(config.cluster_model, mode='wb'))

# test_predict = model.predict(test_paper_matrix)
# test_cluster_dict = dict(zip(test_paper_list, list(test_predict)))

true_predict = model.predict(true_paper_matrix)
true_cluster_dict = dict(zip(true_paper_list, list(true_predict)))

# 测试分组
test_output = copy.deepcopy(true_author)
for name in true_author.keys():
    paper_list = true_author[name]

    new_paper_list = []
    new_paper_dict = dict()

    for paper in paper_list:
        cluster_id = true_cluster_dict[paper]

        if cluster_id not in new_paper_dict.keys():
            new_paper_dict[cluster_id] = []
            new_paper_dict[cluster_id].append(paper)
        else:
            new_paper_dict[cluster_id].append(paper)

    for cluster_id in new_paper_dict.keys():
        new_paper_list.append(new_paper_dict[cluster_id])

    test_output[name] = new_paper_list

json.dump(test_output, open("./result.json", mode='w'))
