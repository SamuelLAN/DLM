import os
import math
import random
from functools import reduce
from nmt.preprocess.corpus import news_commentary
from lib.preprocess import utils
from lib.utils import create_dir, write_pkl, load_pkl
from pretrain.preprocess.config import data_dir


class GenData:
    RANDOM_STATE = 42
    NMT_TRAIN_RATIO = 0.98
    PRETRAIN_TRAIN_RATIO = 0.98
    BATCH_SIZE_PER_FILE = 12

    buffer_size = 1000

    def __init__(self, start_ratio=0.0, end_ratio=0.98, _sample_rate=1.0, data_params={}, pretrain_params={},
                 tokenizer_pl=[], encoder_pl=[], _tokenizer_dir='cdlm', _dataset='cdlm'):
        # initialize variables
        self.__data_params = data_params
        self.__pretrain_params = pretrain_params
        self.__tokenizer_pl = tokenizer_pl
        self.__encoder_pl = encoder_pl
        self.__sample_rate = _sample_rate

        self.__running = True
        self.__cur_index = 0
        self.__data = []

        self.__tokenizer_path = os.path.join(create_dir(data_dir, 'tokenizer', _tokenizer_dir), 'tokenizer.pkl')
        self.__processed_dir_path = create_dir(data_dir, 'preprocessed', _dataset)

        # load data from files
        data = news_commentary.zh_en()

        data = self.__split_data(data, 0., self.NMT_TRAIN_RATIO)
        data = reduce(lambda x, y: x + y, data)

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        random.shuffle(data)

        # get tokenizer
        if os.path.isfile(self.__tokenizer_path):
            self.__tokenizer = load_pkl(self.__tokenizer_path)
        else:
            self.__tokenizer_src, self.__tokenizer_tar = list(zip(*data))
            self.get_tokenizer()

        # get the data set (train or validation or test)
        data = self.__split_data(data, start_ratio, end_ratio)

        self.gen_preprocessed_data(data, self.BATCH_SIZE_PER_FILE)

    def get_tokenizer(self):
        print('\nstart training tokenizer ... ')

        self.__tokenizer = utils.pipeline(
            self.__tokenizer_pl, self.__tokenizer_src, self.__tokenizer_tar, self.__data_params,
        )

        del self.__tokenizer_src
        del self.__tokenizer_tar

        print('finish training tokenizer')

        # saving the tokenizer to file
        write_pkl(self.__tokenizer_path, self.__tokenizer)

        return self.__tokenizer

    def gen_preprocessed_data(self, data, batch_size):
        length = len(data)
        num_batch = int(math.ceil(length / batch_size))
        steps = int(num_batch * self.__sample_rate)

        print(f'\nstart generating preprocessed data ({steps} files) ... ')

        for i in range(steps):
            # show progress
            if i % 10 == 0:
                progress = float(i + 1) / steps * 100.
                print('\rprogress: %.2f%% ' % progress, end='')

            # get a batch
            index_of_batch = i % num_batch
            index_start = int(index_of_batch * batch_size)
            index_end = index_start + batch_size
            batch_src, batch_tar = list(zip(*data[index_start: index_end]))

            # preprocess data
            batch_x, batch_y, batch_lan_x, batch_lan_y, batch_pos_y = utils.pipeline(
                self.__encoder_pl, batch_src, batch_tar, {**self.__data_params, 'tokenizer': self.__tokenizer},
                verbose=i ==0
            )

            # save data to file
            file_path = os.path.join(self.__processed_dir_path, f'batch_{i}.pkl')
            write_pkl(file_path, [batch_x, batch_y, batch_lan_x, batch_lan_y, batch_pos_y])

        print('finish generating preprocessed data ')

    @staticmethod
    def __split_data(data, start_ratio, end_ratio):
        """ split data according to the ratio """
        len_data = len(data)
        start_index = int(len_data * start_ratio)
        end_index = int(len_data * end_ratio)
        return data[start_index: end_index]


from pretrain.models.transformer_cdlm_translate import Model

is_train = True
sample_rate = 10.0
train_ratio = 0.98
dataset = f'cdlm_translate_train_{sample_rate}' if is_train else f'cdlm_translate_test'
tokenizer_dir = f'cdlm_translate_train_{sample_rate}'

GenData(
    start_ratio=0.0 if is_train else train_ratio,
    end_ratio=train_ratio if is_train else 1.0,
    _sample_rate=sample_rate if is_train else 1.0,
    data_params=Model.data_params,
    pretrain_params=Model.pretrain_params,
    tokenizer_pl=Model.tokenizer_pl,
    encoder_pl=Model.encode_pl,
    _tokenizer_dir=dataset,
    _dataset=dataset
)
