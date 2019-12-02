#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""稀疏自编码器"""

__author__ = 'yp'

import os
import time
import math
import re
import json
import config
import dill as pickle
from operator import mul
import tensorflow as tf
from functools import reduce


class FeedforwardSparseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, rho=0.01, alpha=0.0001, beta=3, activation=tf.nn.sigmoid,
                 learning_rate=0.1):
        super(FeedforwardSparseAutoEncoder, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.rho = rho  # sparse parameters
        self.alpha = alpha
        self.beta = beta

        self.activation = activation
        self.learning_rate = learning_rate

        self.X = tf.placeholder("float", shape=[None, self.n_input])

        self.W1 = self.init_weights((self.n_input, self.n_hidden))
        self.b1 = self.init_weights((1, self.n_hidden))

        self.W2 = self.init_weights((self.n_hidden, self.n_input))
        self.b2 = self.init_weights((1, self.n_input))

        self.optimizer = self.training()

        self.loss = self.get_loss(self.X)

        self.__name__ = "FeedforwardSparseAutoEncoder"

    def init_weights(self, shape):
        r = math.sqrt(6) / math.sqrt(self.n_input + self.n_hidden + 1)
        weights = tf.random_normal(shape, stddev=r)
        return tf.Variable(weights)

    def encode(self, X):
        l = tf.matmul(X, self.W1) + self.b1
        return self.activation(l)

    def decode(self, H):
        l = tf.matmul(H, self.W2) + self.b2
        return self.activation(l)

    def kl_divergence(self, rho, rho_hat):
        return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

    def regularization(self, weights):
        return tf.nn.l2_loss(weights)

    def get_loss(self, X):
        H = self.encode(X)
        rho_hat = tf.reduce_mean(H,
                                 axis=0)  # Average hidden layer over all data points in X, Page 14 in https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
        kl = self.kl_divergence(self.rho, rho_hat)
        X_ = self.decode(H)
        diff = X - X_
        cost = 0.5 * tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1)) \
               + 0.5 * self.alpha * (tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2)) \
               + self.beta * tf.reduce_sum(kl)
        return cost

    def training(self):
        var_list = [self.W1, self.W2]
        loss_ = self.get_loss(self.X)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss_, var_list)
        training_op = optimizer.apply_gradients(grads_and_vars)
        return training_op

        # train_step = tf.contrib.opt.ScipyOptimizerInterface(loss_, var_list=var_list, method='L-BFGS-B',
        #                                                     options={'maxiter': n_iter})
        # train_step.minimize(self.sess, feed_dict={X: training_data})


def merge_paper_content(_dict):
    tittle = _dict['title'] if not isinstance(_dict['title'], type(None)) else ""

    try:
        content = _dict['abstract'] if not isinstance(_dict['abstract'], type(None)) else ""
    except KeyError:
        content = ""

    try:
        keywords = " ".join(_dict['keywords']) if not isinstance(_dict['keywords'], type(None)) else ""
    except KeyError:
        keywords = ""

    return " ".join([tittle, content, keywords])


def main():
    tfidf_model = pickle.load(open('../my_data/tf_idf_model.pkl', mode="rb"))
    test_pub = json.load(open(config.test_pub_path, mode='r', encoding='utf-8'))
    test_author = json.load(open(config.test_author_path, mode='r', encoding='utf-8'))
    train_pub = json.load(open(config.train_pub_path, mode='r', encoding='utf-8'))
    train_author = json.load(open(config.train_author_path, mode='r', encoding='utf-8'))

    model = FeedforwardSparseAutoEncoder(n_input=21888, n_hidden=1000,
                                         rho=0.01, alpha=0.0001, beta=3,
                                         activation=tf.nn.sigmoid, learning_rate=0.01)
    model_dir = os.path.join('../', '_'.join([model.__name__, time.strftime("%Y%m%d%H")]))
    if os.path.exists(model_dir):
        pass
    else:
        os.mkdir(model_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    num_params = 0
    for variable in tf.trainable_variables():
        print("variable name:", variable)
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    print("model params num:", num_params)

    input_x = model.X
    loss = model.get_loss(input_x)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    epoch_num = 5

    with tf.Session(config=conf) as sess:
        sess.run(init)

        step = 0

        for i in range(epoch_num):
            merge_paper_list = []
            for author in train_author.keys():
                for collection in train_author[author]:
                    for paper in train_author[author][collection]:
                        if len(merge_paper_list) == 8:
                            a = tfidf_model.transform(merge_paper_list).A
                            _loss = sess.run([loss], feed_dict={input_x: a})
                            print("step:{} ===loss:{}".format(step, _loss))

                            merge_paper_list = []
                            step += 1
                        else:
                            merge_paper_list.append(merge_paper_content(train_pub[paper]))

            a = tfidf_model.transform(merge_paper_list).A
            _loss = sess.run([loss], feed_dict={input_x: a})
            saver.save(sess, "%s/model_epoch_%s" % (model_dir, str(i)))


if __name__ == '__main__':
    main()
