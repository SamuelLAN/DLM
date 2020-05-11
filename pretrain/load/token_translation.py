import random
from pretrain.preprocess.config import filtered_union_en_zh_dict_path, filtered_union_zh_en_dict_path
from lib.utils import load_json


class Loader:
    RANDOM_STATE = 42
    NMT_TRAIN_RATIO = 0.98

    def __init__(self, start_ratio=0.0, end_ratio=0.9, sample_ratio=1.0):
        # load data from wmt_news
        zh_en_dict = load_json(filtered_union_zh_en_dict_path)
        en_zh_dict = load_json(filtered_union_en_zh_dict_path)

        data = []
        for zh, val in zh_en_dict.items():
            if not val or 'translation' not in val or not val['translation']:
                continue

            for en in val['translation']:
                data.append([zh, en])
                data.append([en, zh])

        for en, val in en_zh_dict.items():
            if not val or 'translation' not in val or not val['translation']:
                continue

            for zh in val['translation']:
                data.append([zh, en])
                data.append([en, zh])

        # TODO remove duplicate

        # reproduce the process that nmt would go through in order to get its train set; shuffle the data
        random.seed(self.RANDOM_STATE)
        random.shuffle(data)

        # sample data
        data = self.sample_data(data, sample_ratio)

        # split dataset
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
        if sample_rate <= 1.0:
            return data[: int(len_data * sample_rate)]
        else:
            return data * int(sample_rate) + data[: int(len_data * (sample_rate - int(sample_rate)))]

    def data(self):
        return self.__src_data, self.__tar_data


