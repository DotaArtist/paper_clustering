# coding=utf-8
import json
import config


# todo
def transform_name(cap_name):
    """
    H.Y. Wang : h_y_wang
    """
    cap_name = [i.lower() for i in cap_name.split(' ')]
    return cap_name[1] + '_' + cap_name[0]


# todo
def restore_name(low_name):
    low_name = [i for i in low_name.split('_')]
    return low_name[1].upper() + ' ' + low_name[0].capitalize()


class DataCollection(object):
    def __init__(self):
        self.test_pub_path = config.test_pub_path
        self.test_author_path = config.test_author_path
        self.train_pub_path = config.train_pub_path
        self.train_author_path = config.train_author_path

        self.test_pub = None
        self.test_author = None
        self.train_pub = None
        self.train_author = None

    def load(self):
        self.test_pub = json.load(open(self.test_pub_path, mode='r', encoding='utf-8'))
        self.test_author = json.load(open(self.test_author_path, mode='r', encoding='utf-8'))
        self.train_pub = json.load(open(self.train_pub_path, mode='r', encoding='utf-8'))
        self.train_author = json.load(open(self.train_author_path, mode='r', encoding='utf-8'))

    @staticmethod
    def get_paper_info(paper_id, author_data):
        return author_data[paper_id]


if __name__ == '__main__':
    pass
