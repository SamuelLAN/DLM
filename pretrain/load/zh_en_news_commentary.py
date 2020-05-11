import random
from functools import reduce
from nmt.preprocess.corpus import news_commentary


class Loader:
    RANDOM_STATE = 42
    TRAIN_RATIO = 0.98

    def __init__(self, start_ratio=0.0, end_ratio=0.98, sample_rate=1.0):
        # load data from files
        data = news_commentary.zh_en()

        data = self.__split_data(data, 0., self.TRAIN_RATIO)
        data = reduce(lambda x, y: x + y, data)

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        random.shuffle(data)

        # get the train set
        data = self.__split_data(data, start_ratio, end_ratio)

        if start_ratio == 0. or sample_rate < 1.:
            # sample data if the data size is too big; low resource setting
            data = self.sample_data(data, sample_rate)

        self.__src_data, self.__tar_data = list(zip(*data))

    @staticmethod
    def __split_data(data, start_ratio, end_ratio):
        """ split data according to the ratio """
        len_data = len(data)
        start_index = int(len_data * start_ratio)
        end_index = int(len_data * end_ratio)
        return data[start_index: end_index]

    @staticmethod
    def sample_data(data, sample_rate):
        len_data = len(data)
        if sample_rate <= 1.0:
            return data[: int(len_data * sample_rate)]
        else:
            return data * int(sample_rate) + data[: int(len_data * (sample_rate - int(sample_rate)))]

    def data(self):
        return self.__src_data, self.__tar_data
