import random
from nmt.preprocess.corpus import wmt_news, europarl


class Loader:
    RANDOM_STATE = 42

    def __init__(self, start_ratio=0.0, end_ratio=0.8, sample_rate=1.0):
        # load data from files
        fr_data, en_data = wmt_news.fr_en()

        # add more data source
        # TODO check if you need the europaral data
        #   also check the combine ratio between wmt_news and europarl
        fr_data_2, en_data_2 = europarl.fr_en()
        fr_data += fr_data_2
        en_data += en_data_2

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        data = list(zip(fr_data, en_data))
        random.shuffle(data)

        # sample data if the data size is too big; low resource setting
        data = self.sample_data(data, sample_rate)

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

    @staticmethod
    def sample_data(data, sample_rate):
        len_data = len(data)
        return data[: int(len_data * sample_rate)]

    def data(self):
        return self.__src_data, self.__tar_data
