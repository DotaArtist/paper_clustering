#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""emd + triple loss train"""

__author__ = 'yp'

import os
import config
import numpy as np
from utils import eval_utils
from utils import data_utils
from keras import backend as K
from keras.optimizers import Adam
from scipy.spatial import distance
from keras.layers import Dense, Input, Lambda
from keras.models import Model, model_from_json

"""
global metric learning model
"""
MODEL_DIR = "../model_v2"

# EMB_DIM = 50
EMB_DIM = 300
ORIGIN_MODEL_NAME = "1127_emb150_200w"
NEW_MODEL_NAME = "step_1127_emb150_200w"

# ORIGIN_MODEL_NAME = "ban_1127_emb150_200w"
# NEW_MODEL_NAME = "ban_1127_emb150_200w"


def l2Norm(x):
    return K.l2_normalize(x, axis=-1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    trip_loss = K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0, 0]) - K.square(y_pred[:, 1, 0]) + margin))
    # d = tf.Print(trip_loss, [trip_loss], "loss")
    return trip_loss


def accuracy(_, y_pred):
    return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])


class GlobalTripletModel:

    def __init__(self, data_scale):
        self.data_scale = data_scale
        # self.train_triplets_dir = "../my_data/150w_triple_train.txt"
        self.train_triplets_dir = "../my_data/nec_train.txt"
        self.test_triplets_dir = "../my_data/14w_triple_train.txt"
        self.train_triplet_files_num = 360996
        self.test_triplet_files_num = 50000
        self.emb_dict = self.load_pretrain_emb()
        self.emb_test_dict = self.load_pretrain_test_emb()
        self.emb_true_dict = self.load_pretrain_true_emb()
        self.chunk_size = 32000
        self.file_obj = None

    @staticmethod
    def get_triplets_files_num(path_dir):
        files = []
        for f in os.listdir(path_dir):
            if f.startswith('anchor_embs_'):
                files.append(f)
        return len(files)

    def load_batch_triplets(self, line_string):
        anchor, pos, neg = line_string.strip().split("\t")
        X1 = self.emb_dict[anchor]
        X2 = self.emb_dict[pos]
        X3 = self.emb_dict[neg]
        return X1, X2, X3

    @staticmethod
    def load_pretrain_emb():
        return data_utils.load_doc_emb(config.pre_train_paper_embedding)

    @staticmethod
    def load_pretrain_test_emb():
        return data_utils.load_doc_emb(config.pre_train_test_paper_embedding)

    @staticmethod
    def load_pretrain_true_emb():
        return data_utils.load_doc_emb(config.pre_train_true_paper_embedding)

    def init_read(self, path):
        self.file_obj = open(path, mode="r", encoding="utf-8")

    def batch_triplets_data(self):
        X1, X2, X3 = [], [], []

        counter = 0
        for line in self.file_obj.readlines():
            x1_batch, x2_batch, x3_batch = self.load_batch_triplets(line)
            if counter == 0:
                X1, X2, X3 = [], [], []

            X1.append(x1_batch)
            X2.append(x2_batch)
            X3.append(x3_batch)
            counter += 1

            if counter == self.chunk_size:
                counter = 0
                yield np.array(X1), np.array(X2), np.array(X3)
        yield np.array(X1), np.array(X2), np.array(X3)

    def test_triplets_data(self):
        X1, X2, X3 = [], [], []

        for line in self.file_obj.readlines():
            x1_batch, x2_batch, x3_batch = self.load_batch_triplets(line)
            X1.append(x1_batch)
            X2.append(x2_batch)
            X3.append(x3_batch)
        return np.array(X1), np.array(X2), np.array(X3)

    @staticmethod
    def create_triplet_model():
        emb_anchor = Input(shape=(EMB_DIM,), name='anchor_input')
        emb_pos = Input(shape=(EMB_DIM,), name='pos_input')
        emb_neg = Input(shape=(EMB_DIM,), name='neg_input')

        # shared layers
        layer1 = Dense(128, activation='relu', name='first_emb_layer')
        layer2 = Dense(64, activation='relu', name='last_emb_layer')
        norm_layer = Lambda(l2Norm, name='norm_layer', output_shape=[64])

        encoded_emb = norm_layer(layer2(layer1(emb_anchor)))
        encoded_emb_pos = norm_layer(layer2(layer1(emb_pos)))
        encoded_emb_neg = norm_layer(layer2(layer1(emb_neg)))

        pos_dist = Lambda(euclidean_distance, name='pos_dist')([encoded_emb, encoded_emb_pos])
        neg_dist = Lambda(euclidean_distance, name='neg_dist')([encoded_emb, encoded_emb_neg])

        def cal_output_shape(input_shape):
            shape = list(input_shape[0])
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists',
            output_shape=cal_output_shape
        )([pos_dist, neg_dist])

        model = Model([emb_anchor, emb_pos, emb_neg], stacked_dists, name='triple_siamese')
        model.compile(loss=triplet_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])

        inter_layer = Model(inputs=model.get_input_at(0), outputs=model.get_layer('norm_layer').get_output_at(0))
        return model, inter_layer

    def load_triplets_model(self):
        model_dir = MODEL_DIR
        rf = open(os.path.join(model_dir, '{}.json'.format(ORIGIN_MODEL_NAME)), 'r')
        model_json = rf.read()
        rf.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(os.path.join(model_dir, '{}.h5'.format(ORIGIN_MODEL_NAME)))
        return loaded_model

    def train_triplets_model(self, step_learning=False):
        if step_learning:
            model = self.load_triplets_model()
            model.compile(loss=triplet_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])

            # read data
            self.init_read(self.train_triplets_dir)
            chuck_num = 0
            for X1, X2, X3 in self.batch_triplets_data():
                n_triplets = len(X1)
                # print(model.summary())
                X_anchor, X_pos, X_neg = X1, X2, X3
                X = {'anchor_input': X_anchor, 'pos_input': X_pos, 'neg_input': X_neg}

                # 模型训练
                print("chunk num:===={}".format(chuck_num))
                model.fit(X, np.ones((n_triplets, 2)), batch_size=64, shuffle=True, epochs=20)
                chuck_num += 1

            model_json = model.to_json()
            model_dir = MODEL_DIR
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, '{}.json'.format(NEW_MODEL_NAME)), 'w') as wf:
                wf.write(model_json)
            model.save_weights(os.path.join(model_dir, '{}.h5'.format(NEW_MODEL_NAME)))

            # val
            self.init_read(path=self.test_triplets_dir)
            test_triplets = self.test_triplets_data()
            auc_score = eval_utils.full_auc(model, test_triplets)
            print('val AUC', auc_score)

            self.init_read(path=self.test_triplets_dir)
            loaded_model = self.load_triplets_model()
            print('triplets model loaded')
            auc_score = eval_utils.full_auc(loaded_model, test_triplets)
            print('val AUC', auc_score)

        else:
            model, inter_model = self.create_triplet_model()

            # read data
            self.init_read(self.train_triplets_dir)
            chuck_num = 0
            for X1, X2, X3 in self.batch_triplets_data():
                n_triplets = len(X1)
                # print(model.summary())
                X_anchor, X_pos, X_neg = X1, X2, X3
                X = {'anchor_input': X_anchor, 'pos_input': X_pos, 'neg_input': X_neg}

                # 模型训练
                print("chunk num:===={}".format(chuck_num))
                model.fit(X, np.ones((n_triplets, 2)), batch_size=64, shuffle=True, epochs=5)
                chuck_num += 1

            model_json = model.to_json()
            model_dir = MODEL_DIR
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, '{}.json'.format(NEW_MODEL_NAME)), 'w') as wf:
                wf.write(model_json)
            model.save_weights(os.path.join(model_dir, '{}.h5'.format(NEW_MODEL_NAME)))

            # val
            self.init_read(path=self.test_triplets_dir)
            test_triplets = self.test_triplets_data()
            auc_score = eval_utils.full_auc(model, test_triplets)
            print('val AUC', auc_score)

            self.init_read(path=self.test_triplets_dir)
            loaded_model = self.load_triplets_model()
            print('triplets model loaded')
            auc_score = eval_utils.full_auc(loaded_model, test_triplets)
            print('val AUC', auc_score)

    def evaluate_triplet_model(self):
        self.init_read(path="")
        test_triplets = self.batch_triplets_data()
        loaded_model = self.load_triplets_model()
        print('triplets model loaded')
        auc_score = eval_utils.full_auc(loaded_model, test_triplets)
        print('test AUC', auc_score)


if __name__ == '__main__':
    global_model = GlobalTripletModel(data_scale=2000000)

    # paper_dict = global_model.emb_dict  # 训练
    # paper_test_dict = global_model.emb_test_dict  # 测试
    paper_true_dict = global_model.emb_true_dict

    # 训练
    # global_model.train_triplets_model(step_learning=True)  # 是否分步训练

    # 预测
    loaded_model = global_model.load_triplets_model()
    last_emb_layer_model = Model(inputs=loaded_model.get_layer('anchor_input').get_input_at(0),
                                 outputs=loaded_model.get_layer('norm_layer').get_output_at(0))

    # a = last_emb_layer_model.predict(np.array([paper_dict[random.sample(paper_dict.keys(), 1)[0]]]))
    # b = last_emb_layer_model.predict(np.array([paper_dict[random.sample(paper_dict.keys(), 1)[0]]]))
    #
    # a = last_emb_layer_model.predict(np.array([paper_dict["XUfKudWh"]]))
    # b = last_emb_layer_model.predict(np.array([paper_dict["Daaib2AJ"]]))
    counter = 0

    # with open(config.trained_paper_embedding, mode="w", encoding="utf-8") as fc:  # 训练
    # with open(config.trained_test_paper_embedding, mode="w", encoding="utf-8") as fc:  # 测试
    with open(config.trained_true_paper_embedding, mode="w", encoding="utf-8") as fc:  # 测试

        # for sample_paper in paper_dict.keys():  # 训练
        # for sample_paper in paper_test_dict.keys():  # 测试
        for sample_paper in paper_true_dict.keys():

            a = last_emb_layer_model.predict(np.array([paper_true_dict[sample_paper]])).reshape(-1)
            b = " ".join(list(a.reshape(-1).astype("str")))
            fc.writelines("{} {}\n".format(sample_paper, b))
            counter += 1
            print(counter)
