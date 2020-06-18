import os
import math
import random
from functools import reduce
from nmt.preprocess.corpus import news_commentary, wmt_news, um_corpus
from lib.preprocess import utils
from lib.utils import create_dir, write_pkl, load_pkl, load_json
from pretrain.preprocess.config import data_dir
from pretrain.preprocess.config import filtered_pos_union_zh_en_dict_path


class GenTokenizer:
    RANDOM_STATE = 42
    NMT_TRAIN_RATIO_NEWS_COMMENTARY = 0.98
    NMT_TRAIN_RATIO_WMT_NEWS = 0.9
    PRETRAIN_TRAIN_RATIO = 0.98
    BATCH_SIZE_PER_FILE = 128

    def __init__(self, data_params={}, tokenizer_pl=[], _tokenizer_dir='only_news_commentary'):
        # initialize variables
        self.__data_params = data_params
        self.__tokenizer_pl = tokenizer_pl
        self.tokenizer_dir = f'{_tokenizer_dir}_{self.__data_params["vocab_size"]}'
        self.tokenizer_path = os.path.join(create_dir(data_dir, 'tokenizer', self.tokenizer_dir), 'tokenizer.pkl')

        if os.path.isfile(self.tokenizer_path):
            return

        data = self.__load_from_news_commentary()
        data += self.__load_from_wmt_news()
        data += self.__load_from_um_corpus()
        data += self.__load_from_dict()

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        random.shuffle(data)

        # get tokenizer
        self.__tokenizer_src, self.__tokenizer_tar = list(zip(*data))
        self.get_tokenizer()

    def __load_from_news_commentary(self):
        data = news_commentary.zh_en()
        data = self.__split_data(data, 0., self.NMT_TRAIN_RATIO_NEWS_COMMENTARY)
        return reduce(lambda x, y: x + y, data)

    def __load_from_wmt_news(self):
        zh_data, en_data = wmt_news.zh_en()
        wmt_news_data = list(zip(zh_data, en_data))
        return self.__split_data(wmt_news_data, 0.0, self.NMT_TRAIN_RATIO_WMT_NEWS)

    @staticmethod
    def __load_from_um_corpus():
        zh_data, en_data = um_corpus.zh_en(get_test=False)
        return list(zip(zh_data, en_data))

    @staticmethod
    def __load_from_dict():
        # load data from files
        # zh_en_dict = load_json(filtered_pos_union_en_zh_dict_path)
        zh_en_dict = load_json(filtered_pos_union_zh_en_dict_path)
        zh_en_list = list(filter(lambda x: 'translation' in x[1] and x[1]['translation'], zh_en_dict.items()))
        zh_en_list = list(map(lambda x: [[x[0]] * len(x[1]['translation']), x[1]['translation']], zh_en_list))
        # data = reduce(lambda x, y: [x[0] + y[0], x[1] + y[1]], zh_en_list)

        zh_data = []
        en_data = []
        length = len(zh_en_list)
        for i, val in enumerate(zh_en_list):
            if i % 50 == 0:
                progress = float(i + 1) / length * 100.
                print('\rprogress: %.2f%% ' % progress, end='')

            zh_data += val[0]
            en_data += val[1]

        return list(zip(zh_data, en_data))

    def get_tokenizer(self):
        print('\nstart training tokenizer ... ')

        self.__tokenizer = utils.pipeline(
            self.__tokenizer_pl, self.__tokenizer_src, self.__tokenizer_tar, self.__data_params,
        )

        del self.__tokenizer_src
        del self.__tokenizer_tar

        print('finish training tokenizer')

        # saving the tokenizer to file
        write_pkl(self.tokenizer_path, self.__tokenizer)

        return self.__tokenizer

    @staticmethod
    def __split_data(data, start_ratio, end_ratio):
        """ split data according to the ratio """
        len_data = len(data)
        start_index = int(len_data * start_ratio)
        end_index = int(len_data * end_ratio)
        return data[start_index: end_index]


from pretrain.models.transformer_cdlm_translate import Model

tokenizer_dir = f'news_commentary_wmt_news_um_corpus_zh_en_dict'

GenTokenizer(
    data_params=Model.data_params,
    tokenizer_pl=Model.tokenizer_pl,
    _tokenizer_dir=tokenizer_dir,
)
