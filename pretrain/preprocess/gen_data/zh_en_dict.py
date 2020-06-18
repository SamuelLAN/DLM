import os
import math
import random
from functools import reduce
from nmt.preprocess.corpus import news_commentary
from lib.preprocess import utils
from lib.utils import create_dir, write_pkl, load_pkl, load_json
from pretrain.preprocess.config import data_dir
from pretrain.preprocess.config import filtered_pos_union_en_zh_dict_path, filtered_pos_union_zh_en_dict_path


class GenData:
    RANDOM_STATE = 42
    NMT_TRAIN_RATIO = 0.98
    PRETRAIN_TRAIN_RATIO = 0.98
    BATCH_SIZE_PER_FILE = 128

    def __init__(self, start_ratio=0.0, end_ratio=0.98, _sample_rate=1.0, data_params={},
                 tokenizer_pl=[], encoder_pl=[], _tokenizer_dir='cdlm', _dataset='cdlm'):
        # initialize variables
        self.__data_params = data_params
        self.__tokenizer_pl = tokenizer_pl
        self.__encoder_pl = encoder_pl
        self.__sample_rate = _sample_rate

        self.__tokenizer_path = os.path.join(create_dir(data_dir, 'tokenizer', _tokenizer_dir), 'tokenizer.pkl')
        self.__processed_dir_path = create_dir(data_dir, 'preprocessed', _dataset)

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

        data = list(zip(zh_data, en_data))

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
            batch_x, batch_y, _, _ = utils.pipeline(
                self.__encoder_pl, batch_src, batch_tar, {**self.__data_params, 'tokenizer': self.__tokenizer},
                verbose=i == 0
            )

            # save data to file
            file_path = os.path.join(self.__processed_dir_path, f'batch_{i}.pkl')
            write_pkl(file_path, [batch_x, batch_y])

        print('finish generating preprocessed data ')

    @staticmethod
    def __split_data(data, start_ratio, end_ratio):
        """ split data according to the ratio """
        len_data = len(data)
        start_index = int(len_data * start_ratio)
        end_index = int(len_data * end_ratio)
        return data[start_index: end_index]


from nmt.models.transformer_baseline import Model

is_train = False
sample_rate = 1.0
train_ratio = 0.995
dataset = f'word_translate_zh_en_train_{sample_rate}_tokenizer_all' if is_train else f'word_translate_zh_en_test_tokenizer_all'
# tokenizer_dir = f'word_translate_train'
# tokenizer_dir = f'only_news_commentary_80000'
tokenizer_dir = f'news_commentary_wmt_news_um_corpus_zh_en_dict_90000'

GenData(
    start_ratio=0.0 if is_train else train_ratio,
    end_ratio=train_ratio if is_train else 1.0,
    _sample_rate=sample_rate if is_train else 1.0,
    data_params=Model.data_params,
    tokenizer_pl=Model.tokenizer_pl,
    encoder_pl=Model.encode_pipeline,
    _tokenizer_dir=tokenizer_dir,
    _dataset=dataset
)
