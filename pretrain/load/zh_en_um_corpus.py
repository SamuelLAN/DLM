import random
from nmt.preprocess.corpus import um_corpus


class Loader:
    RANDOM_STATE = 42
    NMT_TRAIN_RATIO = 0.9

    def __init__(self, start_ratio=0.0, end_ratio=0.9, data_size=None, domain='*', sample_rate=1.0):
        # get all data
        if not data_size:
            zh_data, en_data = um_corpus.zh_en(domain)

        # get data according to specific ratio
        else:
            domain_dict = {
                'education': 7,
                'laws': 5,
                'news': 15,
                'science': 6,
                'spoken': 5,
                'subtitles': 6,
                'thesis': 6,
            }
            total = list(map(lambda x: x[1], list(domain_dict.items())))
            total = float(sum(total))

            zh_data = []
            en_data = []
            for domain, val in domain_dict.items():
                tmp_zh_data, tmp_en_data = um_corpus.zh_en(domain)
                sample_size = int(val / total * int(data_size))
                zh_data += tmp_zh_data[:sample_size]
                en_data += tmp_en_data[:sample_size]

        # reproduce the process that nmt would go through in order to get its train set; shuffle the data
        random.seed(self.RANDOM_STATE)
        data = list(zip(zh_data, en_data))
        random.shuffle(data)

        # get the train set
        data = self.__split_data(data, 0.0, self.NMT_TRAIN_RATIO)        # sample data

        # split dataset
        data = self.__split_data(data, start_ratio, end_ratio)

        if start_ratio == 0. or sample_rate < 1.:
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
