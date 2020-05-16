import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_root_dir)
sys.path.append(root_dir)

from pretrain.preprocess.config import Ids

Ids.multi_task = True
Ids.cdlm_tasks = ['translation', 'pos', 'ner', 'synonym', 'def']

from pretrain.models.transformer_cdlm_fully_share import Model
from pretrain.train.train_CDLM_fully_share_wmt_news import Train as TrainBase
from pretrain.load.zh_en_news_commentary import Loader as Loader_news_news_commentary
from pretrain.load.zh_en_wmt_news import Loader
from lib.utils import cache, read_cache, md5

Model.name = 'transformer_CDLM_fully_share_news_commentary'
Model.sample_params = {
    'translation': 1.2,
    'pos': 0.2,
    'ner': 0.3,
    'synonym': 0.2,
    'definition': 0.1,
}
Model.data_params['over_sample_rate'] = Model.sample_params


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader
    Loader2 = Loader_news_news_commentary

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
            self.train_pos_y, \
            self.test_x, \
            self.test_y, \
            self.test_lan_x, \
            self.test_lan_y, \
            self.test_pos_y, \
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
                self.train_pos_y,
                self.test_x,
                self.test_y,
                self.test_lan_x,
                self.test_lan_y,
                self.test_pos_y,
                self.tokenizer,
                self.vocab_size,
            ])

        print(f'vocab_size: {self.vocab_size}\n')
        print(f'train_x.shape: {self.train_x.shape}\ntrain_y.shape: {self.train_y.shape}')
        print(f'train_lan_x.shape: {self.train_lan_x.shape}\ntrain_lan_y.shape: {self.train_lan_y.shape}')
        print(f'train_pos_y.shape: {self.train_pos_y.shape}')
        print(f'test_x.shape: {self.test_x.shape}\ntest_y.shape: {self.test_y.shape}')
        print(f'test_lan_x.shape: {self.test_lan_x.shape}\ntest_lan_y.shape: {self.test_lan_y.shape}')
        print(f'test_pos_y.shape: {self.test_pos_y.shape}')

    def load_data(self):
        """ load the data """
        print('\nLoading data (combined) ...')

        # load the data
        tokenizer_loader = self.Loader(0.0, 1.0, self.M.data_params['sample_ratio'])
        train_loader = self.Loader(0.0, self.Loader.PRETRAIN_TRAIN_RATIO, self.M.data_params['sample_ratio'])
        test_loader = self.Loader(self.Loader.PRETRAIN_TRAIN_RATIO, 1.0, self.M.data_params['sample_ratio'])

        # load the data
        tokenizer_loader_2 = self.Loader2(0.0, 1.0, self.M.data_params['sample_ratio'])
        train_loader_2 = self.Loader2(0.0, self.Loader2.PRETRAIN_TRAIN_RATIO, self.M.data_params['sample_ratio'])
        test_loader_2 = self.Loader2(self.Loader2.PRETRAIN_TRAIN_RATIO, 1.0, self.M.data_params['sample_ratio'])

        # get data for tokenizer; if load from exist model, then do not need to regenerate the tokenizer
        load_model_params = self.M.checkpoint_params['load_model']
        if not load_model_params:
            train_tokenizer_src, train_tokenizer_tar = tokenizer_loader.data()
            train_tokenizer_src_2, train_tokenizer_tar_2 = tokenizer_loader_2.data()

            self.train_tokenizer_src = train_tokenizer_src + train_tokenizer_src_2
            self.train_tokenizer_tar = train_tokenizer_tar + train_tokenizer_tar_2

        # get data
        train_src, train_tar = train_loader.data()
        test_src, test_tar = test_loader.data()

        train_src_2, train_tar_2 = train_loader_2.data()
        test_src_2, test_tar_2 = test_loader_2.data()

        # combine data
        self.train_src = train_src + train_src_2
        self.train_tar = train_tar + train_tar_2
        self.test_src = test_src + test_src_2
        self.test_tar = test_tar + test_tar_2

        print('\nFinish loading (combined) ')


o_train = Train(use_cache=True)
o_train.train()
o_train.test(load_model=False)
