#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import sys
sys.path.append("../")
import json
import random
import config
import numpy as np
import dill as pickle
from utils import data_utils
from sklearn.neighbors import NearestNeighbors


def nec_sample(seq, probabilities):
    index = 0
    if len(seq) == len(probabilities):
        x = random.uniform(0, 1)
        cumprob = 0.0
        item = None
        for item, item_pro in zip(seq, probabilities):
            cumprob += item_pro
            if x < cumprob:
                return item, index

            index += 1
        return item, index
    else:
        raise ValueError('length not equal')


def softmax(x):
    return x/np.sum(x, axis=0)


def merge_paper_content(_dict):
    tittle, content, keywords = _dict['title'], _dict['abstract'], " ".join(_dict['keywords'])
    return " ".join([tittle, content, keywords])


def main():
    tfidf_model = pickle.load(open('../my_data/tf_idf_model.pkl', mode="rb"))

    test_pub = json.load(open(config.test_pub_path, mode='r', encoding='utf-8'))
    test_author = json.load(open(config.test_author_path, mode='r', encoding='utf-8'))

    train_pub = json.load(open(config.train_pub_path, mode='r', encoding='utf-8'))
    train_author = json.load(open(config.train_author_path, mode='r', encoding='utf-8'))

    feature_list = list(tfidf_model.get_feature_names())

    train_paper_set = set(train_pub.keys())
    train_paper_list = list(train_paper_set)

    # 随机采样三元组
    # counter = 0
    # with open("triple_train.txt", mode="w", encoding="utf-8") as f1:
    #     for author in train_author.keys():
    #         for collection in train_author[author]:
    #             pos_set = set(train_author[author][collection])
    #             neg_set = train_paper_set - pos_set
    #             #
    #             for paper in pos_set:
    #                 for i in range(2):
    #                     pos_sample = random.sample(pos_set, 1)
    #
    #                     for j in range(4):
    #                         neg_sample = random.sample(neg_set, 1)
    #                         f1.writelines("{}\t{}\t{}\n".format(paper, pos_sample[0], neg_sample[0]))
    #                         counter += 1
    #                         print(counter)

    # sub step learning + 负采样
    trained_model = data_utils.load_doc_emb(config.trained_paper_embedding)

    all_paper_matrix = []
    for i in train_paper_list:
        all_paper_matrix.append(trained_model[i])
    all_paper_matrix = np.array(all_paper_matrix)

    total_nbrs = NearestNeighbors(n_neighbors=len(train_paper_list), algorithm='auto', metric="cosine").fit(all_paper_matrix)

    counter = 0
    write_counter = 0
    sample_num = 2
    pos_sample_num = 2
    neg_sample_num = 2

    train_paper_list_size = len(train_paper_list)

    # with open("nec_triple_train_{}.txt".format(sys.argv[1]), mode="w", encoding="utf-8", buffering=0) as f1:
    with open("nec_triple_train_{}.txt".format("local"), mode="w", encoding="utf-8") as f1:
        for author in train_author.keys():
            for collection in train_author[author]:
                pos_list = train_author[author][collection]
                neg_set = train_paper_set - set(pos_list)

                # 添加正负标签
                tag_label_list = [0 for i in range(train_paper_list_size)]
                for i in pos_list:
                    tag_label_list[train_paper_list.index(i)] = 1

                for pos_paper in pos_list:
                    print("===counter{}: ===write counter:{}".format(str(counter), str(write_counter)))
                    counter += 1
                    # 随机采样
                    random_seed = random.uniform(0, 1)

                    if random_seed > 0.9 and counter >= 42236:
                        index = train_paper_list.index(pos_paper)
                        distances, indices = total_nbrs.kneighbors(all_paper_matrix[index:index + 1])

                        indices_tmp = indices[0].tolist()  # 最近的节点id
                        distances_tmp = distances[0].tolist()  # 最近的节点距离
                        sort_distances_tmp = sorted(zip(indices_tmp, distances_tmp), key=lambda x: x[0], reverse=False) # 按照全量排序

                        # tag为0的概率值置为0
                        new_distance = [np.exp(1 - i[1]) if j == 1 else 0 for i, j in zip(sort_distances_tmp, tag_label_list)]  # 全部
                        new_distance[index] = np.exp(0.005)

                        pos_prob_tmp = softmax(new_distance)

                        for i in range(sample_num):
                            pos_sample, pos_sample_index = nec_sample(train_paper_list, pos_prob_tmp)

                            pos_sample_distance = sort_distances_tmp[pos_sample_index][1]  # 距离越小越相似

                            # for j in range(neg_sample_num):
                                # tag为1的概率值置为0
                            neg_distance = [np.exp(1 - i[1])
                                            if j == 0 and i[1] > pos_sample_distance
                                            else 0 for i, j in zip(sort_distances_tmp, tag_label_list)]  # 大于正样本距离的

                            neg_prob_tmp = softmax(neg_distance)

                            neg_sample, neg_sample_index = nec_sample(train_paper_list, neg_prob_tmp)
                            f1.writelines("{}\t{}\t{}\n".format(pos_paper, pos_sample, neg_sample))
                            write_counter += 1


if __name__ == '__main__':
    main()
