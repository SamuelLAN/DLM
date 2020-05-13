import os, time
from pretrain.train.train_base import Train as TrainBase
from pretrain.models.transformer_cdlm_translate import Model
from pretrain.load.zh_en_wmt_news import Loader
from lib.preprocess import utils
from lib.utils import cache, read_cache, get_relative_file_path, md5

Model.name = 'transformer_CDLM_translate_wmt_news'


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader

    def __init__(self, use_cache=True):
        # read data from cache ;
        #    if no cache, then load the data and preprocess it, then store it to cache
        cache_name = f'pre{Train.TRAIN_NAME}_preprocessed_data_{md5(self.M.data_params)}.pkl'
        data = read_cache(cache_name) if use_cache else None
        if not isinstance(data, type(None)):
            self.__train_x, \
            self.__train_y, \
            self.__train_lan_x, \
            self.__train_lan_y, \
            self.__train_pos_y, \
            self.__test_x, \
            self.__test_y, \
            self.__test_lan_x, \
            self.__test_lan_y, \
            self.__test_pos_y, \
            self.__tokenizer, \
            self.__vocab_size = data

        else:
            self.load_data()
            self.preprocess()

            cache(cache_name, [
                self.__train_x,
                self.__train_y,
                self.__train_lan_x,
                self.__train_lan_y,
                self.__train_pos_y,
                self.__test_x,
                self.__test_y,
                self.__test_lan_x,
                self.__test_lan_y,
                self.__test_pos_y,
                self.__tokenizer,
                self.__vocab_size,
            ])

        print(f'vocab_size: {self.__vocab_size}\n')
        print(f'train_x.shape: {self.__train_x.shape}\ntrain_y.shape: {self.__train_y.shape}')
        print(f'train_lan_x.shape: {self.__train_lan_x.shape}\ntrain_lan_y.shape: {self.__train_lan_y.shape}')
        print(f'train_pos_y.shape: {self.__train_pos_y.shape}')
        print(f'test_x.shape: {self.__test_x.shape}\ntest_y.shape: {self.__test_y.shape}')
        print(f'test_lan_x.shape: {self.__test_lan_x.shape}\ntest_lan_y.shape: {self.__test_lan_y.shape}')
        print(f'test_pos_y.shape: {self.__test_pos_y.shape}')

    def preprocess(self):
        """ preprocess the data to list of list token idx """
        print('\nProcessing data ... ')

        # get tokenizer
        load_model_params = self.M.checkpoint_params['load_model']
        if not load_model_params:
            self.__tokenizer = utils.pipeline(
                self.M.tokenizer_pl, self.__train_tokenizer_src, self.__train_tokenizer_tar, self.M.data_params,
            )
            del self.__train_tokenizer_src
            del self.__train_tokenizer_tar

        # load tokenizer from cache
        else:
            tokenizer_path = get_relative_file_path('runtime', 'tokenizer',
                                                    load_model_params[0], load_model_params[1], 'tokenizer.pkl')
            self.__tokenizer = read_cache(tokenizer_path)

        # preprocess train data
        self.__train_x, self.__train_y, self.__train_lan_x, self.__train_lan_y, self.__train_pos_y = utils.pipeline(
            self.M.encode_pl,
            self.__train_src,
            self.__train_tar,
            {**self.M.data_params, 'tokenizer': self.__tokenizer},
        )

        # preprocess test data
        self.__test_x, self.__test_y, self.__test_lan_x, self.__test_lan_y, self.__test_pos_y = utils.pipeline(
            self.M.encode_pl,
            self.__test_src,
            self.__test_tar,
            {**self.M.data_params, 'tokenizer': self.__tokenizer},
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
        print('\nBuilding model ({}) ...'.format(self.M.TIME))
        self.model = self.M(self.__vocab_size, self.__vocab_size)

        # save tokenizer
        cache(os.path.join(self.model.tokenizer_dir, 'tokenizer.pkl'), self.__tokenizer)

        self.__train_pos_y = self.model.pos_emb(self.__train_pos_y)
        self.__test_pos_y = self.model.pos_emb(self.__test_pos_y)

        print('\nTraining model ...')
        start_time = time.time()
        self.model.train(
            train_x=(self.__train_x, self.__train_lan_x,
                     self.__train_y[:, :-1], self.__train_lan_y[:, :-1], self.__train_pos_y[:, :-1]),
            train_y=self.__train_y[:, 1:],
            val_x=(self.__test_x, self.__test_lan_x,
                   self.__test_y[:, :-1], self.__test_lan_y[:, :-1], self.__test_pos_y[:, :-1]),
            val_y=self.__test_y[:, 1:]
        )
        self.__train_time = time.time() - start_time
        print('\nFinish training')

    def test(self, load_model=False):
        """ test BLEU here """
        if load_model:
            self.model = self.M(self.__vocab_size, self.__vocab_size, finish_train=True)

            self.__train_pos_y = self.model.pos_emb(self.__train_pos_y)
            self.__test_pos_y = self.model.pos_emb(self.__test_pos_y)

            self.model.train(
                train_x=(self.__train_x, self.__train_lan_x,
                         self.__train_y[:, :-1], self.__train_lan_y[:, :-1], self.__train_pos_y[:, :-1]),
                train_y=self.__train_y[:, 1:],
            )
            self.__train_time = 0.

        print('\nTesting model ...')

        train_examples = self.show_examples(5, self.__train_x, self.__train_lan_x,
                                            self.__train_y, self.__train_lan_y, self.__train_pos_y)
        test_examples = self.show_examples(5, self.__test_x, self.__test_lan_x,
                                           self.__test_y, self.__test_lan_y, self.__test_pos_y)
        print('\nTrain examples: {}\n\nTest examples: {}'.format(train_examples, test_examples))

        print('\n\nCalculating metrics ...')

        start_train_time = time.time()
        train_loss, train_acc, train_ppl = self.model.evaluate_metrics_for_encoded(
            'train', self.__train_x[:2000], self.__train_lan_x[:2000],
            self.__train_y[:2000], self.__train_lan_y[:2000], self.__test_pos_y[:2000]
        )
        start_test_time = time.time()
        test_loss, test_acc, test_ppl = self.model.evaluate_metrics_for_encoded(
            'test', self.__test_x, self.__test_lan_x, self.__test_y, self.__test_lan_y, self.__test_pos_y
        )
        self.__test_train_time = start_test_time - start_train_time
        self.__test_test_time = time.time() - start_test_time

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
