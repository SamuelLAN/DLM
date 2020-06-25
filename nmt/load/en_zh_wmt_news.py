import random
from nmt.preprocess.corpus import wmt_news, um_corpus


class Loader:
    RANDOM_STATE = 42
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1

    def __init__(self, start_ratio=0.0, end_ratio=0.8, sample_rate=1.0):
        # load data from files
        zh_data, en_data = wmt_news.zh_en()

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        data = list(zip(en_data, zh_data))
        random.shuffle(data)

        # split data according to the ratio (for train set, val set and test set)
        data = self.__split_data(data, start_ratio, end_ratio)

        self.__src_data, self.__tar_data = list(zip(*data))

    @staticmethod
    def __split_data(data, start_ratio, end_ratio):
        """ split data according to the ratio """
        len_data = len(data)
        start_index = int(len_data * start_ratio)
        end_index = int(len_data * end_ratio)
        return data[start_index: end_index]

    def data(self):
        return self.__src_data, self.__tar_data
