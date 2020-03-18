import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
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

from models.transformer_for_nmt import Model
from lib.preprocess import utils
from lib.utils import cache, read_cache
from load.zh_en import Loader


class Train:

    def __init__(self, use_cache=True):
        # read data from cache ;
        #    if no cache, then load the data and preprocess it, then store it to cache
        cache_name = 'preprocessed_data.pkl'
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
        train_loader = Loader(0.0, 0.9)
        test_loader = Loader(0.9, 1.0)

        # load data
        self.__train_src, self.__train_tar = train_loader.data()
        self.__test_src, self.__test_tar = test_loader.data()

        print('\nFinish loading ')

    def __preprocess(self):
        """ preprocess the data to list of list token idx """
        print('\nProcessing data ... ')

        self.__train_src_encode, self.__train_tar_encode, self.__src_tokenizer, self.__tar_tokenizer = utils.pipeline(
            Model.preprocess_pipeline,
            self.__train_src,
            self.__train_tar,
            Model.data_params,
        )

        params = {
            **Model.data_params,
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

        print('\nTraining model ...')
        self.model.train(self.__train_src_encode, self.__train_tar_encode)
        print('\nFinish training')

    def test(self, load_model=False):
        """ test BLEU here """
        if load_model:
            self.model = Model(self.__src_vocab_size, self.__tar_vocab_size, finish_train=True)
            self.model.train(self.__train_src_encode, self.__train_tar_encode)

        print('\nTesting model ...')

        print('\nTranslate examples:')
        example_num = 10

        pred = self.model.translate_list_token_idx(self.__train_src_encode[:example_num], self.__tar_tokenizer)
        for i in range(example_num):
            src_lan = self.model.decode_src_data(self.__train_src_encode[i:i + 1], self.__src_tokenizer)[0]
            tar_lan = self.model.decode_tar_data(self.__train_tar_encode[i:i + 1], self.__tar_tokenizer)[0]
            print('\nsrc_lan: {}\ntar_lan: {}\ntranslation: {}'.format(src_lan, tar_lan, pred[i]))

        print('\nCalculating bleu ...')

        train_bleu = self.model.calculate_bleu_for_encoded(self.__train_src_encode, self.__train_tar_encode, 'train')
        test_bleu = self.model.calculate_bleu_for_encoded(self.__test_src_encode, self.__test_tar_encode, 'test')

        print('\nFinish testing')


o_train = Train(False)
o_train.train()
o_train.test()
