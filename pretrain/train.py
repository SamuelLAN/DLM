import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import os
import time
from pretrain.models.transformer_mlm import Model
from lib.preprocess import utils
from lib.utils import cache, read_cache, create_dir_in_root, md5
from pretrain.load.zh_en import Loader


class Train:

    def __init__(self, use_cache=True):
        # read data from cache ;
        #    if no cache, then load the data and preprocess it, then store it to cache
        cache_name = f'pretrain_preprocessed_data_{md5(Model.data_params)}.pkl'
        data = read_cache(cache_name) if use_cache else None
        if not isinstance(data, type(None)):
            self.__train_x, \
            self.__train_y, \
            self.__train_lan_x, \
            self.__train_lan_y, \
            self.__test_x, \
            self.__test_y, \
            self.__test_lan_x, \
            self.__test_lan_y, \
            self.__tokenizer, \
            self.__vocab_size = data

        else:
            self.__load_data()
            self.__preprocess()

            cache(cache_name, [
                self.__train_x,
                self.__train_y,
                self.__train_lan_x,
                self.__train_lan_y,
                self.__test_x,
                self.__test_y,
                self.__test_lan_x,
                self.__test_lan_y,
                self.__tokenizer,
                self.__vocab_size,
            ])

        print(f'vocab_size: {self.__vocab_size}\n')
        print(f'train_x.shape: {self.__train_x.shape}\ntrain_y.shape: {self.__train_y.shape}')
        print(f'train_lan_x.shape: {self.__train_lan_x.shape}\ntrain_lan_y.shape: {self.__train_lan_y.shape}')
        print(f'test_x.shape: {self.__test_x.shape}\ntest_y.shape: {self.__test_y.shape}')
        print(f'test_lan_x.shape: {self.__test_lan_x.shape}\ntest_lan_y.shape: {self.__test_lan_y.shape}')

    def __load_data(self):
        """ load the data """
        print('\nLoading data ...')

        # load the data
        tokenizer_loader = Loader(0.0, 1.0, Model.data_params['sample_ratio'], Model.data_params['sample_um_ratio'])
        train_loader = Loader(0.0, 0.9, Model.data_params['sample_ratio'], Model.data_params['sample_um_ratio'])
        test_loader = Loader(0.9, 1.0, Model.data_params['sample_ratio'], Model.data_params['sample_um_ratio'])

        # get data for tokenizer; if load from exist model, then do not need to regenerate the tokenizer
        load_model_params = Model.checkpoint_params['load_model']
        if not load_model_params:
            self.__train_tokenizer_src, self.__train_tokenizer_tar = tokenizer_loader.data()

        # get data
        self.__train_src, self.__train_tar = train_loader.data()
        self.__test_src, self.__test_tar = test_loader.data()

        print('\nFinish loading ')

    def __preprocess(self):
        """ preprocess the data to list of list token idx """
        print('\nProcessing data ... ')

        # get tokenizer
        load_model_params = Model.checkpoint_params['load_model']
        if not load_model_params:
            self.__tokenizer = utils.pipeline(
                Model.tokenizer_pl, self.__train_tokenizer_src, self.__train_tokenizer_tar, Model.data_params,
            )
            del self.__train_tokenizer_src
            del self.__train_tokenizer_tar

        # load tokenizer from cache
        else:
            tokenizer_path = create_dir_in_root('runtime', 'tokenizer',
                                                load_model_params[0], load_model_params[1], 'tokenizer.pkl')
            self.__tokenizer = read_cache(tokenizer_path)

        # preprocess train data
        self.__train_x, self.__train_y, self.__train_lan_x, self.__train_lan_y = utils.pipeline(
            Model.MLM_pl,
            self.__train_src,
            self.__train_tar,
            {**Model.data_params, 'tokenizer': self.__tokenizer},
        )

        # preprocess test data
        self.__test_x, self.__test_y, self.__test_lan_x, self.__test_lan_y = utils.pipeline(
            Model.MLM_pl,
            self.__test_src,
            self.__test_tar,
            {**Model.data_params, 'tokenizer': self.__tokenizer},
        )

        # get vocabulary size
        self.__vocab_size = self.__tokenizer.vocab_size

        # release storage
        del self.__train_src
        del self.__train_tar
        del self.__test_src
        del self.__test_tar

        print('\nFinish preprocessing ')

    def train(self):
        print('\nBuilding model ({}) ...'.format(Model.TIME))
        self.model = Model(self.__vocab_size, self.__vocab_size)

        print('\nTraining model ...')
        start_time = time.time()
        self.model.train(
            train_x=(self.__train_x, self.__train_lan_x, self.__train_y[:, :-1], self.__train_lan_y[:, :-1]),
            train_y=self.__train_y[:, 1:],
            val_x=(self.__test_x, self.__test_lan_x, self.__test_y[:, :-1], self.__test_lan_y[:, :-1]),
            val_y=self.__test_y[:, 1:]
        )
        self.__train_time = time.time() - start_time
        print('\nFinish training')

    def test(self, load_model=False):
        """ test BLEU here """
        if load_model:
            self.model = Model(self.__vocab_size, self.__vocab_size, finish_train=True)
            self.model.train(
                train_x=(self.__train_x, self.__train_lan_x, self.__train_y[:, :-1], self.__train_lan_y[:, :-1]),
                train_y=self.__train_y[:, 1:],
            )
            self.__train_time = 0.

        print('\nTesting model ...')

        # train_examples = self.show_examples(self.__train_x, self.__train_y, 5)
        # test_examples = self.show_examples(self.__test_x, self.__test_y, 5)

        # print('\nTrain examples: {}\n\nTest examples: {}'.format(train_examples, test_examples))

        # print('\n\nCalculating bleu ...')

        start_train_time = time.time()
        # train_loss = self.model.calculate_loss_for_encoded(self.__train_src_encode, self.__train_tar_encode, 'train')
        # train_bleu = 1.0
        start_test_time = time.time()
        # test_loss = self.model.calculate_loss_for_encoded(self.__test_src_encode, self.__test_tar_encode, 'test')
        self.__test_train_time = start_test_time - start_train_time
        self.__test_test_time = time.time() - start_test_time

        print('\nFinish testing')

        # self.log({
        #     'train_loss': train_loss,
        #     'train_bleu': train_bleu,
        #     'test_loss': test_loss,
        #     'test_bleu': test_bleu,
        #     'train_examples': train_examples,
        #     'test_examples': test_examples,
        # })

    # def show_examples(self, src_encoded_data, tar_encoded_data, example_num):
    #     pred = self.model.translate_list_token_idx(src_encoded_data[:example_num], self.__tar_tokenizer)
    #     examples = ''
    #     for i in range(example_num):
    #         src_lan = self.model.decode_src_data(src_encoded_data[i:i + 1], self.__src_tokenizer)[0]
    #         tar_lan = self.model.decode_tar_data(tar_encoded_data[i:i + 1], self.__tar_tokenizer)[0]
    #         examples += 'src_lan: {}\ntar_lan: {}\ntranslation: {}\n\n'.format(src_lan, tar_lan, pred[i])
    #     return examples

    def log(self, kwargs):
        string = '\n'.join(list(map(lambda x: '{}: {}'.format(x[0], x[1]), list(kwargs.items()))))
        data = (self.model.name, self.model.TIME, string,
                self.model.data_params, self.model.model_params, self.model.train_params,
                self.__train_time, self.__test_train_time, self.__test_test_time)

        string = '\n---------------------------------------------------' \
                 '\nmodel_name: {}\nmodel_time: {}\n{}\n' \
                 'data_params: {}\nmodel_params: {}\ntrain_params: {}\n' \
                 'train_time: {}\ntest_train_time: {}\ntest_test_time: {}\n\n'.format(*data)

        print(string)

        with open(os.path.join(create_dir_in_root('runtime', 'log'), '{}.log'.format(self.model.name)), 'ab') as f:
            f.write(string.encode('utf-8'))


o_train = Train(use_cache=True)
o_train.train()
o_train.test(load_model=False)
