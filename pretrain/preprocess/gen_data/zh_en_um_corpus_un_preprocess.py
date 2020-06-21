import os
import math
import random
from nmt.preprocess.corpus import um_corpus
from lib.utils import create_dir, write_pkl
from pretrain.preprocess.config import data_dir


class GenData:
    RANDOM_STATE = 42
    BATCH_SIZE_PER_FILE = 128

    def __init__(self, _dataset='cdlm'):
        # initialize variables
        self.__processed_dir_path = create_dir(data_dir, 'un_preprocessed', _dataset)

        zh_data, en_data = um_corpus.zh_en(get_test=False)
        data = list(zip(zh_data, en_data))

        # shuffle the data
        random.seed(self.RANDOM_STATE)
        random.shuffle(data)

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


GenData('zh_en_um_corpus')
