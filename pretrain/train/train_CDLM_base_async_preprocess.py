import os
import time
from pretrain.train.train_base import Train as TrainBase
from pretrain.models.transformer_cdlm_translate import Model
from pretrain.load.async_loader import Loader as LoaderVal
from pretrain.load.async_preprocess_loader import Loader as LoaderTrain
from pretrain.preprocess.config import data_dir
from lib.utils import load_pkl, get_file_path

Model.name = 'transformer_CDLM_translate_wmt_news'


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    LoaderTrain = LoaderTrain
    LoaderVal = LoaderVal

    train_preprocess_dirs = ['news_commentary_train']
    val_preprocess_dirs = ['news_commentary_test_v2']
    tokenizer_dir = 'only_news_commentary_80000'

    def __init__(self):
        # load the data
        self.train_loader = self.LoaderTrain(self.tokenizer_dir, self.train_preprocess_dirs,
                                             self.M.data_params, self.M.pretrain_params, self.M.encode_pl)
        self.val_loader = self.LoaderVal(*self.val_preprocess_dirs)

        # get the generator of the dataset
        self.train_data = self.train_loader.generator(self.M.pos_emb, self.M.train_params['batch_size'])
        self.val_data = self.val_loader.generator(self.M.pos_emb, self.M.train_params['batch_size'])

        # get the data size
        self.train_size = self.train_loader.size()
        self.val_size = self.val_loader.size()

        # get an example of a batch
        self.train_example_x, self.train_example_y = self.train_loader.batch_example(self.M.pos_emb)
        self.val_example_x, self.val_example_y = self.train_loader.batch_example(self.M.pos_emb)
        self.train_batch = self.train_loader.batch_data(self.M.pos_emb)
        self.val_batch = self.val_loader.batch_data(self.M.pos_emb)

        # load the tokenizer
        self.tokenizer = load_pkl(get_file_path(data_dir, 'tokenizer', self.tokenizer_dir, 'tokenizer.pkl'))
        self.vocab_size = self.tokenizer.vocab_size

        # show some statistics for dataset
        print(f'vocab_size: {self.vocab_size}\n')
        self.train_stats = self.train_loader.show_statistics(self.M.pos_emb)
        self.val_stats = self.val_loader.show_statistics(self.M.pos_emb)

    def train(self):
        print('\nBuilding model ({}) ...'.format(self.M.TIME))
        self.model = self.M(self.vocab_size, self.vocab_size)

        print('\nTraining model ...')
        start_time = time.time()
        self.model.train(
            train_x=self.train_data,
            train_y=None,
            val_x=self.val_data,
            val_y=None,
            train_size=self.train_size,
            val_size=self.val_size,
            train_example_x=self.train_example_x,
            train_example_y=self.train_example_y
        )
        self.train_time = time.time() - start_time
        print('\nFinish training')

    def test(self, load_model=False):
        """ test BLEU here """
        if load_model:
            self.model = self.M(self.vocab_size, self.vocab_size, finish_train=True)
            self.model.train(train_x=self.train_example_x, train_y=self.train_example_y)
            self.train_time = 0.

        print('\nTesting model ...')

        train_examples = self.show_examples(5, *self.train_batch)
        test_examples = self.show_examples(5, *self.val_batch)
        print('\nTrain examples: {}\n\nTest examples: {}'.format(train_examples, test_examples))

        print('\n\nCalculating metrics ...')

        start_train_time = time.time()
        train_loss, train_acc, train_ppl = self.model.evaluate_metrics_for_encoded(
            'train', *self.train_loader.batch_data(self.M.pos_emb, 2000)
        )
        start_test_time = time.time()
        test_loss, test_acc, test_ppl = self.model.evaluate_metrics_for_encoded(
            'test', *self.val_loader.batch_data(self.M.pos_emb, self.val_size)
        )
        self.test_train_time = start_test_time - start_train_time
        self.test_test_time = time.time() - start_test_time

        print('\nFinish testing')

        model_list = os.listdir(self.model.model_dir)
        if model_list:
            model_list.sort(reverse=True)
            best_model = model_list[0]
        else:
            best_model = ''

        self.log({
            'train_stats': self.train_stats,
            'val_stats': self.val_stats,
            'vocab_size': self.vocab_size,
            'tokenizer_dir': self.tokenizer_dir,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_ppl': train_ppl,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_ppl': test_ppl,
            'train_examples': train_examples,
            'test_examples': test_examples,
            'early_stop_at': best_model,
            'initialize_from': self.model.checkpoint_params['load_model'],
        })

    def end(self):
        self.train_loader.stop()
        self.val_loader.stop()
