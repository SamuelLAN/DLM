import random
from nmt.preprocess.corpus import wmt_news, um_corpus


class Loader:
    RANDOM_STATE = 42
    NMT_TRAIN_RATIO = 0.9
    PRETRAIN_TRAIN_RATIO = 0.9

    def __init__(self, start_ratio=0.0, end_ratio=0.9, sample_rate=1.0):
        # load data from wmt_news
        zh_data, en_data = wmt_news.zh_en()

        # reproduce the process that nmt would go through in order to get its train set; shuffle the data
        random.seed(self.RANDOM_STATE)
        data = list(zip(zh_data, en_data))
        random.shuffle(data)

        # get the train set
        data = self.__split_data(data, 0.0, self.NMT_TRAIN_RATIO)
        
        # split dataset
        data = self.__split_data(data, start_ratio, end_ratio)

        if start_ratio == 0. or sample_rate < 1.:
            # sample data
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
