import os
import math
import random
from functools import reduce
from nmt.preprocess.corpus import news_commentary
from lib.preprocess import utils
from lib.utils import create_dir, write_pkl, load_pkl
from pretrain.preprocess.config import data_dir


class GenTokenizer:
    RANDOM_STATE = 42
    NMT_TRAIN_RATIO = 0.98
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

            # load data from files
        data = news_commentary.zh_en()

        data = self.__split_data(data, 0., self.NMT_TRAIN_RATIO)
        data = reduce(lambda x, y: x + y, data)

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        random.shuffle(data)

        # get tokenizer
        self.__tokenizer_src, self.__tokenizer_tar = list(zip(*data))
        self.get_tokenizer()

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

tokenizer_dir = f'only_news_commentary'

GenTokenizer(
    data_params=Model.data_params,
    tokenizer_pl=Model.tokenizer_pl,
    _tokenizer_dir=tokenizer_dir,
)
