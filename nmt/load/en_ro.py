import random
from nmt.preprocess.corpus import europarl, setimes


class Loader:
    RANDOM_STATE = 42
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1

    def __init__(self, start_ratio=0.0, end_ratio=0.8, sample_rate=1.0):
        # load data from files
        ro_data, en_data = europarl.ro_en()
        ro_data_2, en_data_2 = setimes.ro_en()

        # shuffle the data
        data = list(zip(en_data + en_data_2, ro_data + ro_data_2))
        random.seed(self.RANDOM_STATE)
        random.shuffle(data)

        # split data according to the ratio (for train set, val set and test set)
        data = self.__split_data(data, start_ratio, end_ratio)

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
        return data[: int(len_data * sample_rate)]

    def data(self):
        return self.__src_data, self.__tar_data
