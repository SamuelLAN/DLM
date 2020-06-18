import os
import time
import random
import threading
import numpy as np
from lib.preprocess import utils
from lib.utils import create_dir, get_file_path, load_pkl
from pretrain.preprocess.config import data_dir


class Loader:
    RANDOM_STATE = 42
    buffer_size = 6400
    queue_size = 1280
    data_size_per_file = 128

    def __init__(self, tokenizer_dir, un_preprocess_dirs,
                 data_params={}, pretrain_params={}, encoder_pl=[]):
        # initialize variables
        self.__data_params = data_params
        self.__pretrain_params = pretrain_params
        self.__encoder_pl = encoder_pl
        self.__dirs = un_preprocess_dirs

        self.__running = True
        self.__cur_index = 0
        self.__data = []
        self.__file_list = []

        self.__tokenizer = load_pkl(get_file_path(data_dir, 'tokenizer', tokenizer_dir, 'tokenizer.pkl'))

        # get the list of all files
        for dir_name in self.__dirs:
            _dir_path = create_dir(data_dir, 'un_preprocessed', dir_name)
            self.__file_list += list(map(lambda x: os.path.join(_dir_path, x), os.listdir(_dir_path)))
        self.__len_files = len(self.__file_list)

        random.seed(self.RANDOM_STATE)
        random.shuffle(self.__file_list)

        self.start()

    def start(self):
        thread = threading.Thread(target=self.__load)
        thread.start()
        print('Start thread for loading data ')

    def stop(self):
        self.__running = False

    def __load(self):
        data_queue = []
        max_queue_size = min(self.size(), self.queue_size)
        max_buffer_size = min(self.size(), self.buffer_size)

        while self.__running:
            while len(data_queue) < max_queue_size:
                file_path = self.__file_list[self.__cur_index]
                self.__cur_index = (self.__cur_index + 1) % self.__len_files

                batch_src, batch_tar = load_pkl(file_path)

                # preprocess data
                batch_x, batch_y, batch_lan_x, batch_lan_y, batch_pos_y = utils.pipeline(
                    self.__encoder_pl, batch_src, batch_tar, {**self.__data_params, 'tokenizer': self.__tokenizer},
                    verbose=False
                )

                data_queue += list(zip(batch_x, batch_y, batch_lan_x, batch_lan_y, batch_pos_y))

            if len(self.__data) < max_buffer_size:
                random.seed(42)
                random.shuffle(data_queue)

                self.__data += data_queue
                data_queue = []

            time.sleep(0.1)

        print('Stop thread for loading data ')

    def size(self):
        return self.__len_files * self.data_size_per_file

    def generator(self, pos_emb_fn, batch_size=16):
        while True:
            X, Y = self.batch_example(pos_emb_fn, batch_size)
            self.__data = self.__data[batch_size:]
            yield X, Y

    def batch_example(self, pos_emb_fn, batch_size=16):
        batch_x, batch_lan_x, batch_y, batch_lan_y, batch_pos_y = self.batch_data(pos_emb_fn, batch_size)

        X = (batch_x, batch_lan_x, batch_y[:, :-1], batch_lan_y[:, :-1], batch_pos_y[:, :-1])
        Y = batch_y[:, 1:]
        return X, Y

    def batch_data(self, pos_emb_fn, batch_size=16):
        while len(self.__data) < batch_size:
            time.sleep(0.3)

        data = self.__data[: batch_size]
        batch_x, batch_y, batch_lan_x, batch_lan_y, batch_pos_y = zip(*data)
        batch_pos_y = pos_emb_fn(batch_pos_y)
        return np.array(batch_x), np.array(batch_lan_x), np.array(batch_y), np.array(batch_lan_y), np.array(batch_pos_y)

    def show_statistics(self, pos_emb_fn=None):
        while len(self.__data) < 16:
            time.sleep(0.3)

        data = self.__data[: 16]
        batch_x, batch_y, batch_lan_x, batch_lan_y, batch_pos_y = zip(*data)

        stats = {
            'dirs': self.__dirs,
            'size': self.size(),
            'x.shape': np.array(batch_x).shape,
            'y.shape': np.array(batch_y).shape,
            'lan_x.shape': np.array(batch_lan_x).shape,
            'lan_y.shape': np.array(batch_lan_y).shape,
            'pos_y.shape': np.array(batch_pos_y).shape,
            'pos_emb_y.shape': pos_emb_fn(np.array(batch_pos_y)).shape if
            not isinstance(pos_emb_fn, type(None)) else '',
        }

        print(f'\n----------------------------------------')
        for k, v in stats.items():
            print(f'{k}: {v}')

        return stats

