# coding=utf-8
import re
import config
import numpy as np
from scipy import sparse

from bert_pre_train import BertPreTrain
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import dill as pickle


class FeatureWeight(object):
    def __init__(self):
        self.origin_path = config.origin_data
        self.author_feature = config.author_feature
        self.keyword_feature = config.keyword_feature
        self.pre_train_model = BertPreTrain(mode='pre_train', language='en', padding=False,
                                            embedding_dim=config.embedding_dim)

    def get_author_feature(self):
        author_feature_map = dict()
        author_feature_map_reverse = dict()
        feature_counter = 0

        author_feature_matrix = sparse.lil_matrix((config.train_paper_size, config.author_feature_map_len))

        with open('author_feature_simple.txt', mode='w', encoding='utf-8') as fa:
            with open(self.origin_path, mode='r', encoding='utf-8') as fo:
                for (line_index, line) in enumerate(fo):
                    line = line.strip('\n').split('\t')

                    paper_id = line[0]
                    name_org = line[1].split('|')
                    name_org = [i.split('@')[0] for i in name_org]  # 除去机构后缀
                    feature_list = []

                    for _name in name_org:
                        if _name not in author_feature_map.keys():
                            author_feature_map[_name] = feature_counter
                            author_feature_map_reverse[feature_counter] = _name
                            feature_list.append(str(feature_counter))

                            author_feature_matrix[line_index, feature_counter] = 1  # 赋值
                            feature_counter += 1
                        else:
                            feature_list.append(str(author_feature_map[_name]))
                            author_feature_matrix[line_index, author_feature_map[_name]] = 1  # 赋值

                    fa.writelines('{}\t{}\n'.format(paper_id, '|'.join(feature_list)))
        print(author_feature_matrix.shape)
        print(len(author_feature_map.keys()))
        pickle.dump(author_feature_matrix, open(config.author_feature_map_pkl, mode='wb'))

    def get_keyword_feature(self):
        keyword_feature_map = dict()
        keyword_feature_map_reverse = dict()
        feature_counter = 0

        keyword_feature_matrix = sparse.lil_matrix((config.train_paper_size, config.keyword_feature_map_len))

        with open(self.keyword_feature, mode='w', encoding='utf-8') as fa:
            with open(self.origin_path, mode='r', encoding='utf-8') as fo:
                for (line_index, line) in enumerate(fo):
                    line = line.strip('\n').split('\t')

                    paper_id = line[0]
                    keyword_list = re.split(r' |\|', line[4])
                    feature_list = []

                    for _keyword in keyword_list:
                        if _keyword not in keyword_feature_map.keys():
                            keyword_feature_map[_keyword] = feature_counter
                            keyword_feature_map[feature_counter] = _keyword
                            feature_list.append(str(feature_counter))

                            keyword_feature_matrix[line_index, feature_counter] = 1  # 赋值
                            feature_counter += 1
                        else:
                            feature_list.append(str(keyword_feature_map[_keyword]))
                            keyword_feature_matrix[line_index, keyword_feature_map[_keyword]] = 1  # 赋值

                    fa.writelines('{}\t{}\n'.format(paper_id, '|'.join(feature_list)))
        print(keyword_feature_matrix.shape)
        print(len(keyword_feature_map.keys()))
        pickle.dump(keyword_feature_map, open('keyword_feature_map_dict.pkl', mode='wb'))
        pickle.dump(keyword_feature_matrix, open(config.keyword_feature_map_pkl, mode='wb'))

    def get_content_feature(self):
        """glove word embedding + idf"""
        # with open(config.pre_train_paper_embedding, mode='w', encoding='utf-8') as ft: # 训练
        # with open(config.pre_train_test_paper_embedding, mode='w', encoding='utf-8') as ft: # 测试
        with open(config.pre_train_true_paper_embedding, mode='w', encoding='utf-8') as ft:
            counter = 0

            # with open(self.origin_path, mode='r', encoding='utf-8') as fo: # 训练
            # with open(config.test_origin_data, mode='r', encoding='utf-8') as fo: # 测试
            with open(config.true_origin_data, mode='r', encoding='utf-8') as fo:  # true
                total_feature_combined = []
                for (line_index, line) in enumerate(fo):
                    line = line.strip('\n').split('\t')
                    paper_id, tittle, content, keywords = line[0], line[2], line[3], line[4]

                    paper_tittle_feature = self.pre_train_model.get_output(tittle.split(' '), _show_tokens=False)
                    paper_content_feature = self.pre_train_model.get_output(content.split(' '), _show_tokens=False)
                    paper_keywords_feature = self.pre_train_model.get_output(re.split(r' |\|', keywords), _show_tokens=False)

                    # feature_combined = np.sum(
                    #     (np.sum(paper_tittle_feature, axis=0, keepdims=False),
                    #      np.sum(paper_content_feature, axis=0, keepdims=False),
                    #      np.sum(paper_keywords_feature, axis=0, keepdims=False)
                    #      ), axis=0, keepdims=False).reshape(-1)
                    feature_combined = np.concatenate(
                        (np.sum(paper_tittle_feature, axis=0, keepdims=False),
                         np.sum(paper_content_feature, axis=0, keepdims=False),
                         np.sum(paper_keywords_feature, axis=0, keepdims=False)), axis=1).reshape(-1)

                    ft.writelines('{} {}\n'.format(paper_id, ' '.join(feature_combined.astype(str).tolist())))
                    print(counter)
                    counter += 1
                    total_feature_combined.append(feature_combined)
                total_feature_combined = np.stack(total_feature_combined, axis=0)
                print('feature loaded!')

    def get_content_feature_v2(self):
        """tf-idf"""
        with open(self.origin_path, mode='r', encoding='utf-8') as fo:
            paper_id_list = []
            paper_content_list = []
            for (line_index, line) in enumerate(fo):
                line = line.strip('\n').split('\t')
                paper_id, tittle, content, keywords = line[0], line[2], line[3], line[4]
                paper_id_list.append(paper_id)
                paper_content_list.append(' '.join([tittle, content, keywords]))

            # tf idf
            tf_idf = TfidfVectorizer(analyzer=lambda s: re.split(r" |\||\t", str(s)), min_df=5)
            tf_idf.fit(paper_content_list)
            pickle.dump(tf_idf, open('../my_data/tf_idf_model.pkl', mode="wb"))

            x_train = tf_idf.transform(paper_content_list)
            print(x_train.shape)


if __name__ == '__main__':
    a = FeatureWeight()
    a.get_content_feature()
