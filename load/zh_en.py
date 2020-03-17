import random
from preprocess import wmt_news


class Loader:
    RANDOM_STATE = 42

    def __init__(self, start_ratio=0.0, end_ratio=0.8):
        # load data from files
        zh_data, en_data = wmt_news.zh_en()

        # TODO add more source of data here
        # ...

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        data = list(zip(zh_data, en_data))
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
