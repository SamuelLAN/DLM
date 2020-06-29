import os
import math
import random
from pretrain.load import ro_en
from lib.preprocess import utils
from lib.utils import create_dir, write_pkl, load_pkl
from pretrain.preprocess.config import data_dir


class GenData:
    RANDOM_STATE = 42
    NMT_TRAIN_RATIO = 0.98
    PRETRAIN_TRAIN_RATIO = 0.98
    BATCH_SIZE_PER_FILE = 128

    def __init__(self, _is_train, _sample_rate=1.0, data_params={},
                 tokenizer_pl=[], encoder_pl=[], _tokenizer_dir='cdlm', _dataset='cdlm'):
        # initialize variables
        self.__data_params = data_params
        self.__tokenizer_pl = tokenizer_pl
        self.__encoder_pl = encoder_pl
        self.__sample_rate = _sample_rate

        self.__tokenizer_path = os.path.join(create_dir(data_dir, 'tokenizer', _tokenizer_dir), 'tokenizer.pkl')
        self.__processed_dir_path = create_dir(data_dir, 'preprocessed', _dataset)

        # initialize wmt news loader
        start_ratio = 0.0 if _is_train else ro_en.Loader.PRETRAIN_TRAIN_RATIO
        end_ratio = ro_en.Loader.PRETRAIN_TRAIN_RATIO if _is_train else 1.0
        ro_en_loader = ro_en.Loader(start_ratio, end_ratio, 0.125)

        # load the data
        ro_data, en_data = ro_en_loader.data()
        data = list(zip(ro_data, en_data))

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        random.shuffle(data)

        # get tokenizer
        if os.path.isfile(self.__tokenizer_path):
            self.__tokenizer = load_pkl(self.__tokenizer_path)
        else:
            self.__tokenizer_src, self.__tokenizer_tar = list(zip(*data))
            self.get_tokenizer()

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
                verbose=i == 0
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


from pretrain.models.transformer_cdlm_translate_ro_en import Model

dataset = f'ro_en_cdlm_translate_6w_size_10_sr_80k_voc_test'
tokenizer_dir = f'zh_en_ro_wmt_16_19_20_80000'

GenData(
    _is_train=False,
    _sample_rate=1.0,
    data_params=Model.data_params,
    tokenizer_pl=Model.tokenizer_pl,
    encoder_pl=Model.encode_pl,
    _tokenizer_dir=tokenizer_dir,
    _dataset=dataset
)
