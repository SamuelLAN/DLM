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
from lib.utils import cache, read_cache, create_dir_in_root, md5, get_relative_file_path
from pretrain.load.zh_en_wmt_news import Loader as Loader_wmt_news


class Train:
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader_wmt_news

    def __init__(self, use_cache=True):
        # read data from cache ;
        #    if no cache, then load the data and preprocess it, then store it to cache
        cache_name = f'pre{self.TRAIN_NAME}_preprocessed_data_{md5(self.M.data_params)}.pkl'
        data = read_cache(cache_name) if use_cache else None
        if not isinstance(data, type(None)):
            self.train_x, \
            self.train_y, \
            self.train_lan_x, \
            self.train_lan_y, \
            self.test_x, \
            self.test_y, \
            self.test_lan_x, \
            self.test_lan_y, \
            self.tokenizer, \
            self.vocab_size = data

        else:
            self.load_data()
            self.preprocess_tokenizer()
            self.preprocess()

            cache(cache_name, [
                self.train_x,
                self.train_y,
                self.train_lan_x,
                self.train_lan_y,
                self.test_x,
                self.test_y,
                self.test_lan_x,
                self.test_lan_y,
                self.tokenizer,
                self.vocab_size,
            ])

        print(f'vocab_size: {self.vocab_size}\n')
        print(f'train_x.shape: {self.train_x.shape}\ntrain_y.shape: {self.train_y.shape}')
        print(f'train_lan_x.shape: {self.train_lan_x.shape}\ntrain_lan_y.shape: {self.train_lan_y.shape}')
        print(f'test_x.shape: {self.test_x.shape}\ntest_y.shape: {self.test_y.shape}')
        print(f'test_lan_x.shape: {self.test_lan_x.shape}\ntest_lan_y.shape: {self.test_lan_y.shape}')

    def load_data(self):
        """ load the data """
        print('\nLoading data ...')

        # load the data
        tokenizer_loader = self.Loader(0.0, 1.0, self.M.data_params['sample_ratio'])
        train_loader = self.Loader(0.0, self.Loader.PRETRAIN_TRAIN_RATIO, self.M.data_params['sample_ratio'])
        test_loader = self.Loader(self.Loader.PRETRAIN_TRAIN_RATIO, 1.0, self.M.data_params['sample_ratio'])

        # get data for tokenizer; if load from exist model, then do not need to regenerate the tokenizer
        load_model_params = self.M.checkpoint_params['load_model']
        if not load_model_params:
            self.train_tokenizer_src, self.train_tokenizer_tar = tokenizer_loader.data()

        # get data
        self.train_src, self.train_tar = train_loader.data()
        self.test_src, self.test_tar = test_loader.data()

        print('\nFinish loading ')

    def preprocess_tokenizer(self):
        print('\nProcessing tokenizer ... ')

        # get tokenizer
        load_model_params = self.M.checkpoint_params['load_model']
        if not load_model_params:
            self.tokenizer = utils.pipeline(
                self.M.tokenizer_pl, self.train_tokenizer_src, self.train_tokenizer_tar, self.M.data_params,
            )
            del self.train_tokenizer_src
            del self.train_tokenizer_tar

        # load tokenizer from cache
        else:
            tokenizer_path = get_relative_file_path('runtime', 'tokenizer',
                                                    load_model_params[0], load_model_params[1], 'tokenizer.pkl')
            self.tokenizer = read_cache(tokenizer_path)

    def preprocess(self):
        """ preprocess the data to list of list token idx """
        print('\nProcessing data ... ')

        # preprocess train data
        self.train_x, self.train_y, self.train_lan_x, self.train_lan_y = utils.pipeline(
            self.M.encode_pl,
            self.train_src,
            self.train_tar,
            {**self.M.data_params, 'tokenizer': self.tokenizer},
        )

        # preprocess test data
        self.test_x, self.test_y, self.test_lan_x, self.test_lan_y = utils.pipeline(
            self.M.encode_pl,
            self.test_src,
            self.test_tar,
            {**self.M.data_params, 'tokenizer': self.tokenizer},
        )

        # get vocabulary size
        self.vocab_size = self.tokenizer.vocab_size

        # release storage
        del self.train_src
        del self.train_tar
        del self.test_src
        del self.test_tar

        print('\nFinish preprocessing ')

    def train(self):
        print('\nBuilding model ({}) ...'.format(self.M.TIME))
        self.model = self.M(self.vocab_size, self.vocab_size)

        # save tokenizer
        cache(os.path.join(self.model.tokenizer_dir, 'tokenizer.pkl'), self.tokenizer)

        print('\nTraining model ...')
        start_time = time.time()
        self.model.train(
            train_x=(self.train_x, self.train_lan_x, self.train_y[:, :-1], self.train_lan_y[:, :-1]),
            train_y=self.train_y[:, 1:],
            val_x=(self.test_x, self.test_lan_x, self.test_y[:, :-1], self.test_lan_y[:, :-1]),
            val_y=self.test_y[:, 1:]
        )
        self.train_time = time.time() - start_time
        print('\nFinish training')

    def test(self, load_model=False):
        """ test BLEU here """
        if load_model:
            self.model = self.M(self.vocab_size, self.vocab_size, finish_train=True)
            self.model.train(
                train_x=(self.train_x, self.train_lan_x, self.train_y[:, :-1], self.train_lan_y[:, :-1]),
                train_y=self.train_y[:, 1:],
            )
            self.train_time = 0.

        print('\nTesting model ...')

        train_examples = self.show_examples(5, self.train_x, self.train_lan_x, self.train_y, self.train_lan_y)
        test_examples = self.show_examples(5, self.test_x, self.test_lan_x, self.test_y, self.test_lan_y)
        print('\nTrain examples: {}\n\nTest examples: {}'.format(train_examples, test_examples))

        print('\n\nCalculating metrics ...')

        start_train_time = time.time()
        train_loss, train_acc, train_ppl = self.model.evaluate_metrics_for_encoded(
            'train', self.train_x[:2000], self.train_lan_x[:2000], self.train_y[:2000], self.train_lan_y[:2000]
        )
        start_test_time = time.time()
        test_loss, test_acc, test_ppl = self.model.evaluate_metrics_for_encoded(
            'test', self.test_x, self.test_lan_x, self.test_y, self.test_lan_y
        )
        self.test_train_time = start_test_time - start_train_time
        self.test_test_time = time.time() - start_test_time

        print('\nFinish testing')

        self.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_ppl': train_ppl,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_ppl': test_ppl,
            'train_examples': train_examples,
            'test_examples': test_examples,
        })

    def show_examples(self, example_num, *args):
        pred = self.model.eval_example_for_pretrain(*[v[:example_num] for v in args])

        decoded_x = self.model.decode_encoded_data(self.model.decode_pl, args[0][:example_num], self.tokenizer)
        decoded_y = self.model.decode_encoded_data(self.model.decode_pl, args[2][:example_num], self.tokenizer)
        decoded_pred = self.model.decode_encoded_data(self.model.decode_pl, pred, self.tokenizer)

        return '\n\n'.join(['x: {}\nreference: {}\noutput: {}'.format(decoded_x[i], decoded_y[i], decoded_pred[i])
                            for i in range(example_num)])

    def log(self, kwargs):
        string = '\n'.join(list(map(lambda x: '{}: {}'.format(x[0], x[1]), list(kwargs.items()))))
        data = (self.model.name, self.model.TIME, string,
                self.model.pretrain_params, self.model.data_params, self.model.model_params, self.model.train_params,
                self.train_time, self.test_train_time, self.test_test_time)

        string = '\n---------------------------------------------------' \
                 '\nmodel_name: {}\nmodel_time: {}\n{}\n\n' \
                 'pretrain_params: {}\ndata_params: {}\nmodel_params: {}\ntrain_params: {}\n' \
                 'train_time: {}\ntest_train_time: {}\ntest_test_time: {}\n\n'.format(*data)

        print(string)

        with open(os.path.join(create_dir_in_root('runtime', 'log'), '{}.log'.format(self.model.name)), 'ab') as f:
            f.write(string.encode('utf-8'))
