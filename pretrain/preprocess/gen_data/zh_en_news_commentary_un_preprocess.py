import os
import math
import random
from functools import reduce
from nmt.preprocess.corpus import news_commentary
from lib.utils import create_dir, write_pkl
from pretrain.preprocess.config import data_dir


class GenData:
    RANDOM_STATE = 42
    NMT_TRAIN_RATIO = 0.98
    PRETRAIN_TRAIN_RATIO = 0.98
    BATCH_SIZE_PER_FILE = 128

    def __init__(self, start_ratio=0.0, end_ratio=0.98, _dataset='cdlm'):
        # initialize variables
        self.__processed_dir_path = create_dir(data_dir, 'un_preprocessed', _dataset)

        # load data from files
        data = news_commentary.zh_en()

        data = self.__split_data(data, 0., self.NMT_TRAIN_RATIO)

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        random.shuffle(data)

        # get the data set (train or validation or test)
        data = self.__split_data(data, start_ratio, end_ratio)

        data = reduce(lambda x, y: x + y, data)

        self.gen_data(data, self.BATCH_SIZE_PER_FILE)

    def gen_data(self, data, batch_size):
        length = len(data)
        num_batch = int(math.ceil(length / batch_size))

        print(f'\nstart generating preprocessed data ({num_batch} files) ... ')

        for i in range(num_batch):
            # show progress
            if i % 10 == 0:
                progress = float(i + 1) / num_batch * 100.
                print('\rprogress: %.2f%% ' % progress, end='')

            # get a batch
            index_of_batch = i % num_batch
            index_start = int(index_of_batch * batch_size)
            index_end = index_start + batch_size
            batch_src, batch_tar = list(zip(*data[index_start: index_end]))

            # save data to file
            file_path = os.path.join(self.__processed_dir_path, f'batch_{i}.pkl')
            write_pkl(file_path, [batch_src, batch_tar])

        print('finish generating preprocessed data ')

    @staticmethod
    def __split_data(data, start_ratio, end_ratio):
        """ split data according to the ratio """
        len_data = len(data)
        start_index = int(len_data * start_ratio)
        end_index = int(len_data * end_ratio)
        return data[start_index: end_index]


is_train = False
train_ratio = GenData.PRETRAIN_TRAIN_RATIO
dataset = f'news_commentary_{"train" if is_train else "test"}'

GenData(
    start_ratio=0.0 if is_train else train_ratio,
    end_ratio=train_ratio if is_train else 1.0,
    _dataset=dataset
)
