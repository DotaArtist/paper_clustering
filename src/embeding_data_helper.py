# coding=utf-8
import config
import numpy as np
import dill as pickle


class PreTrainProcess(object):
    def __init__(self, path=config.pre_train_embedding,
                 embedding_dim=config.embedding_dim,
                 sentence_len=config.max_sentence_len,
                 language='ch', pair_mode=False, padding=True):
        embeddings = dict()

        self.embedding_path = path
        self.embedding_dim = embedding_dim
        self.sentence_len = sentence_len
        self.pair_mode = pair_mode
        self.language = language
        self.padding = padding

        self.tfidf_model = pickle.load(open('../my_data/tf_idf_model.pkl', mode="rb"))
        self.idf_dict = dict(zip(self.tfidf_model.get_feature_names(), self.tfidf_model.idf_))

        with open(self.embedding_path, encoding='utf-8', mode='r') as f1:
            for line in f1.readlines():
                line = line.strip().split(' ')
                character = line[0]
                vector = [float(i) for i in line[1:]]

                if character not in embeddings.keys():
                    embeddings[character] = vector
        print('pre train feature loaded.')
        self.embedding_dict = embeddings

    def encode(self, sentence, **kwargs):
        if 'pair_mode' in kwargs.keys():
            if not isinstance(kwargs['pair_mode'], bool):
                raise TypeError("mode type must bool!")

        if 'pair_mode' in kwargs.keys() and kwargs['pair_mode']:
            try:
                assert isinstance(sentence, list)
            except AssertionError:
                print("sentence must be list!")
        else:
            try:
                assert isinstance(sentence, list)
                embedding_unk = [0.0 for _ in range(self.embedding_dim)]
                out_put = []

                for sentence_idx, _sentence in enumerate(sentence):
                    out_put_tmp = []

                    if self.language == 'ch':
                        _sentence = list(_sentence)
                    elif self.language == 'en':
                        _sentence = [_sentence]
                    else:
                        raise Exception("language error!")

                    for char_idx, _char in enumerate(_sentence):
                        if char_idx < self.sentence_len:
                            # idf * emb
                            out_put_tmp.append(self.idf_dict.get(_char, 0.) *
                                               np.array(self.embedding_dict.get(_char, embedding_unk)))
                            # out_put_tmp.append(self.embedding_dict.get(_char, embedding_unk))

                    if self.padding:
                        for i in range(self.sentence_len - len(out_put_tmp)):
                            out_put_tmp.append(embedding_unk)

                    out_put_tmp = np.stack(out_put_tmp, axis=0)
                    out_put.append(out_put_tmp)

                return np.stack(out_put, axis=0)

            except AssertionError:
                print("sentence must be list!")


if __name__ == '__main__':
    model = PreTrainProcess()
    a = model.encode(['asd', 'Objective: To establish a simultaneous determination method of central nervous drugs including barbitals, benzodiazepines,'])
    print(a.shape)
