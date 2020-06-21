import os
import math
import random
from pretrain.load import ro_en
from lib.utils import create_dir, write_pkl
from pretrain.preprocess.config import data_dir


class GenData:
    RANDOM_STATE = 42
    BATCH_SIZE_PER_FILE = 128

    def __init__(self, _is_train, _dataset='cdlm'):
        # initialize variables
        self.__processed_dir_path = create_dir(data_dir, 'un_preprocessed', _dataset)

        # initialize wmt news loader
        start_ratio = 0.0 if _is_train else ro_en.Loader.PRETRAIN_TRAIN_RATIO
        end_ratio = ro_en.Loader.PRETRAIN_TRAIN_RATIO if _is_train else 1.0
        ro_en_loader = ro_en.Loader(start_ratio, end_ratio)

        # load the data
        zh_data, en_data = ro_en_loader.data()
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


is_train = False
dataset = f'ro_en_{"train" if is_train else "test"}'

GenData(
    is_train,
    _dataset=dataset
)
