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

    def __init__(self):
        cache_name = 'preprocessed_data.pkl'
        data = read_cache(cache_name)
        if not isinstance(data, type(None)):
            self.__train_src, self.__train_tar, self.__train_src_encode, self.__train_tar_encode, \
            self.__val_src, self.__val_src_encode, self.__val_tar, self.__val_tar_encode, \
            self.__test_src, self.__test_tar, self.__test_src_encode, \
            self.__src_tokenizer, self.__tar_tokenizer, self.__src_vocab_size, self.__tar_vocab_size = data

        else:
            self.__load_data()
            self.__preprocess()

            cache(cache_name, [
                self.__train_src,
                self.__train_tar,
                self.__train_src_encode,
                self.__train_tar_encode,
                self.__val_src,
                self.__val_src_encode,
                self.__val_tar,
                self.__val_tar_encode,
                self.__test_src,
                self.__test_tar,
                self.__test_src_encode,
                self.__src_tokenizer,
                self.__tar_tokenizer,
                self.__src_vocab_size,
                self.__tar_vocab_size,
            ])

        print('src_vocab_size: {}\ntar_vocab_size: {}'.format(self.__src_vocab_size, self.__tar_vocab_size))
        print('train_size: {}\nval_size: {}\ntest_size: {}'.format(
            len(self.__train_src), len(self.__val_src), len(self.__test_src)))

    def __load_data(self):
        """ load the data """
        print('\nLoading data ...')

        # load the data
        train_loader = Loader(0.0, 0.8)
        val_loader = Loader(0.8, 0.9)
        test_loader = Loader(0.9, 1.0)

        # load data
        self.__train_src, self.__train_tar = train_loader.data()
        self.__val_src, self.__val_tar = val_loader.data()
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

        self.__val_src_encode, self.__val_tar_encode, _, _ = utils.pipeline(
            Model.encode_pipeline,
            self.__val_src,
            self.__val_tar,
            {
                **Model.data_params,
                'src_tokenizer': self.__src_tokenizer,
                'tar_tokenizer': self.__tar_tokenizer,
            }
        )

        self.__test_src_encode = utils.pipeline(Model.encode_pipeline_for_src, self.__test_src, None, {
            **Model.data_params,
            'src_tokenizer': self.__src_tokenizer,
        })

        # get vocabulary size
        self.__src_vocab_size = self.__src_tokenizer.vocab_size
        self.__tar_vocab_size = self.__tar_tokenizer.vocab_size

        print('\nFinish preprocessing ')

    def train(self):
        print('\nBuilding model ...')
        # build model
        self.model = Model(self.__src_vocab_size, self.__tar_vocab_size)

        print('\nTraining model ...')
        # train
        self.model.train(self.__train_src_encode, self.__train_tar_encode, self.__val_src_encode, self.__val_tar_encode)

        print('\nFinish training')

    def test(self):
        """ test BLEU here """
        print('\nTesting model ...')

        train_bleu = self.model.calculate_bleu_for_encoded(self.__train_src_encode, self.__tar_tokenizer,
                                                           self.__train_tar, 'train')
        val_bleu = self.model.calculate_bleu_for_encoded(self.__val_src_encode, self.__tar_tokenizer,
                                                         self.__val_tar, 'val')
        test_bleu = self.model.calculate_bleu_for_encoded(self.__test_src_encode, self.__tar_tokenizer,
                                                          self.__test_tar, 'test')

        print('\nFinish testing')


o_train = Train()
o_train.train()
o_train.test()
