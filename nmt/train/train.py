import tensorflow as tf
import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_root_dir)
sys.path.append(root_dir)

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
from nmt.models.other_lans.transformer_zh_en import Model
from lib.preprocess import utils
from lib.utils import cache, read_cache, create_dir_in_root, md5
# from nmt.load.zh_en_news_commentary import Loader
from nmt.load.zh_en_wmt_news import Loader

Model.name = 'transformer_nmt_baseline'
Model.checkpoint_params['load_model'] = ['baseline', 'wmt-news']


class Train:

    def __init__(self, use_cache=True):
        # read data from cache ;
        #    if no cache, then load the data and preprocess it, then store it to cache
        cache_name = f'nmt_preprocessed_data_{md5(Model.data_params)}.pkl'
        data = read_cache(cache_name) if use_cache else None
        if not isinstance(data, type(None)):
            self.__train_src, \
            self.__train_tar, \
            self.__train_src_encode, \
            self.__train_tar_encode, \
            self.__test_src, \
            self.__test_tar, \
            self.__test_src_encode, \
            self.__test_tar_encode, \
            self.__src_tokenizer, \
            self.__tar_tokenizer, \
            self.__src_vocab_size, \
            self.__tar_vocab_size = data

        else:
            self.__load_data()
            self.__preprocess()

            cache(cache_name, [
                self.__train_src,
                self.__train_tar,
                self.__train_src_encode,
                self.__train_tar_encode,
                self.__test_src,
                self.__test_tar,
                self.__test_src_encode,
                self.__test_tar_encode,
                self.__src_tokenizer,
                self.__tar_tokenizer,
                self.__src_vocab_size,
                self.__tar_vocab_size,
            ])

        print('src_vocab_size: {}\ntar_vocab_size: {}'.format(self.__src_vocab_size, self.__tar_vocab_size))
        print('train_size: {}\ntest_size: {}'.format(len(self.__train_src), len(self.__test_src)))
        print('train_x.shape: {}\ntrain_y.shape: {}'.format(
            self.__train_src_encode.shape, self.__train_tar_encode.shape))
        print('test_x.shape: {}\ntest_y.shape: {}'.format(
            self.__test_src_encode.shape, self.__test_tar_encode.shape))

    def __load_data(self):
        """ load the data """
        print('\nLoading data ...')

        # load the data
        train_loader = Loader(0.0, Loader.TRAIN_RATIO, Model.data_params['sample_rate'])
        test_loader = Loader(Loader.TRAIN_RATIO, 1.0, Model.data_params['sample_rate'])

        # load data
        self.__train_src, self.__train_tar = train_loader.data()
        self.__test_src, self.__test_tar = test_loader.data()

        print('\nFinish loading ')

    def __preprocess(self):
        """ preprocess the data to list of list token idx """
        print('\nProcessing data ... ')

        load_model_params = Model.checkpoint_params['load_model']
        if load_model_params:
            tokenizer_path = create_dir_in_root('runtime', 'tokenizer',
                                                load_model_params[0], load_model_params[1], 'tokenizer.pkl')
            self.__src_tokenizer = self.__tar_tokenizer = read_cache(tokenizer_path)

            self.__train_src_encode, self.__train_tar_encode, _, _ = utils.pipeline(
                Model.encode_pipeline,
                self.__train_src,
                self.__train_tar, {
                    **Model.data_params,
                    'tokenizer': self.__src_tokenizer,
                    'src_tokenizer': self.__src_tokenizer,
                    'tar_tokenizer': self.__tar_tokenizer,
                })

        else:
            self.__train_src_encode, self.__train_tar_encode, self.__src_tokenizer, self.__tar_tokenizer = utils.pipeline(
                Model.preprocess_pipeline,
                self.__train_src,
                self.__train_tar,
                Model.data_params,
            )

        params = {
            **Model.data_params,
            'tokenizer': self.__src_tokenizer,
            'src_tokenizer': self.__src_tokenizer,
            'tar_tokenizer': self.__tar_tokenizer,
        }

        self.__test_src_encode, self.__test_tar_encode, _, _ = utils.pipeline(Model.encode_pipeline,
                                                                              self.__test_src, self.__test_tar, params)

        # get vocabulary size
        self.__src_vocab_size = self.__src_tokenizer.vocab_size
        self.__tar_vocab_size = self.__tar_tokenizer.vocab_size

        print('\nFinish preprocessing ')

    def train(self):
        print('\nBuilding model ({}) ...'.format(Model.TIME))
        self.model = Model(self.__src_vocab_size, self.__tar_vocab_size)

        # save tokenizer before training
        cache(os.path.join(self.model.tokenizer_dir, 'tokenizer.pkl'), self.__src_tokenizer)

        print('\nTraining model ...')
        start_time = time.time()
        self.model.train((self.__train_src_encode, self.__train_tar_encode[:, :-1]), self.__train_tar_encode[:, 1:],
                         (self.__test_src_encode, self.__test_tar_encode[:, :-1]), self.__test_tar_encode[:, 1:])
        self.__train_time = time.time() - start_time
        print('\nFinish training')

    def test(self, load_model=False):
        """ test BLEU here """
        if load_model:
            self.model = Model(self.__src_vocab_size, self.__tar_vocab_size, finish_train=True)
            self.model.train((self.__train_src_encode, self.__train_tar_encode[:, :-1]), self.__train_tar_encode[:, 1:])
            self.__train_time = 0.

        print('\nTesting model ...')

        train_examples = self.show_examples(self.__train_src_encode, self.__train_tar_encode, 5)
        test_examples = self.show_examples(self.__test_src_encode, self.__test_tar_encode, 5)

        print('\nTrain examples: {}\n\nTest examples: {}'.format(train_examples, test_examples))

        print('\n\nCalculating bleu ...')

        start_train_time = time.time()
        train_loss = self.model.calculate_loss_for_encoded(self.__train_src_encode, self.__train_tar_encode, 'train')
        train_bleu = self.model.calculate_bleu_for_encoded(self.__train_src_encode[:2000],
                                                           self.__train_tar_encode[:2000], 'train')
        # train_bleu = 1.0
        start_test_time = time.time()
        test_loss = self.model.calculate_loss_for_encoded(self.__test_src_encode, self.__test_tar_encode, 'test')
        test_bleu = self.model.calculate_bleu_for_encoded(self.__test_src_encode, self.__test_tar_encode, 'test')
        test_precision = self.model.calculate_precision_for_encoded(self.__test_src_encode, self.__src_tokenizer)
        self.__test_train_time = start_test_time - start_train_time
        self.__test_test_time = time.time() - start_test_time

        print('\nFinish testing')

        model_list = os.listdir(self.model.model_dir)
        if model_list:
            model_list.sort(reverse=True)
            best_model = model_list[0]
        else:
            best_model = ''

        shape_of_data = {
            'train_size': len(self.__train_src),
            'test_size': len(self.__test_src),
            'train_x_shape': self.__train_src_encode.shape,
            'train_y_shape': self.__train_tar_encode.shape,
            'test_x_shape': self.__test_src_encode.shape,
            'test_y_shape': self.__test_tar_encode.shape,
        }

        self.log({
            'train_loss': train_loss,
            'train_bleu': train_bleu,
            'test_loss': test_loss,
            'test_bleu': test_bleu,
            'test_precision': test_precision,
            'train_examples': train_examples,
            'test_examples': test_examples,
            'real_vocab_size': self.__src_tokenizer.vocab_size,
            'early_stop_at': best_model,
            'shape_of_data': shape_of_data,
            'initialize_from': self.model.checkpoint_params['load_model'],
        })

    def show_examples(self, src_encoded_data, tar_encoded_data, example_num):
        pred = self.model.translate_list_token_idx(src_encoded_data[:example_num], self.__tar_tokenizer)
        examples = ''
        for i in range(example_num):
            src_lan = self.model.decode_src_data(src_encoded_data[i:i + 1], self.__src_tokenizer)[0]
            tar_lan = self.model.decode_tar_data(tar_encoded_data[i:i + 1], self.__tar_tokenizer)[0]
            examples += 'src_lan: {}\ntar_lan: {}\ntranslation: {}\n\n'.format(src_lan, tar_lan, pred[i])
        return examples

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
# o_train.train()
o_train.test(load_model=True)
