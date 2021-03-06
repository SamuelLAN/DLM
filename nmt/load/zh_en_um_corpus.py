import random
from nmt.preprocess.corpus import um_corpus


class Loader:
    RANDOM_STATE = 42

    def __init__(self, start_ratio=0.0, end_ratio=0.8, data_size=None, is_test=False):
        # load data from files
        if not is_test:
            # get all data
            if not data_size:
                zh_data, en_data = um_corpus.zh_en()

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

        else:
            zh_data, en_data = um_corpus.zh_en(get_test=True)

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        data = list(zip(zh_data, en_data))
        random.shuffle(data)

        # # sample data if the data size is too big; low resource setting
        # data = self.sample_data(data, sample_rate)

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
