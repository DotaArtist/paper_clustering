# coding=utf-8

import config
from bert_serving.client import BertClient
from embeding_data_helper import PreTrainProcess
from config import max_sentence_len


class BertPreTrain(object):
    def __init__(self, mode='remote'):
        if mode == 'remote':
            self.model = BertClient(ip='192.168.236.14', port=5555, check_version=False)
        elif mode == 'pre_train':
            self.model = PreTrainProcess(path=config.pre_train_embedding,
                                         embedding_dim=256, sentence_len=max_sentence_len)
        else:
            self.model = BertClient(ip='127.0.0.1', port=5555, check_version=False)

    def get_output(self, sentence, _show_tokens=True):
        try:
            return self.model.encode(sentence, show_tokens=_show_tokens)
        except TypeError:
            print("sentence must be list!")


if __name__ == "__main__":
    from datetime import datetime
    model = BertPreTrain(mode='pre_train')

    aa = datetime.now()
    a = model.get_output(['输入内容文本最大长度128', '今天'], _show_tokens=False)
    bb = datetime.now()
    print((bb-aa).microseconds)
    print(a.shape)
