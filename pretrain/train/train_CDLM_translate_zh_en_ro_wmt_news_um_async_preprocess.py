import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_root_dir)
sys.path.append(root_dir)

from pretrain.train.train_CDLM_base_async_preprocess import Train as TrainBase
from pretrain.models.transformer_cdlm_translate_zh_en_ro import Model
from pretrain.load.async_loader import Loader as LoaderVal
from pretrain.load.async_preprocess_loader import Loader as LoaderTrain

Model.name = 'transformer_CDLM_translate_zh_en_ro_wmt_news_um_async_preprocess'
# Model.checkpoint_params['load_model'] = ['transformer_CDLM_translate_news_commentary_async_preprocess', '2020_06_14_13_28_17']


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    LoaderTrain = LoaderTrain
    LoaderVal = LoaderVal

    train_preprocess_dirs = ['zh_en_wmt_news_news_commentary_um_corpus_train', 'ro_en_train']
    val_preprocess_dirs = ['zh_en_wmt_news_news_commentary_cdlm_translate_test', 'ro_en_cdlm_translate_ro_6k_test']
    tokenizer_dir = 'zh_en_ro_news_commentary_wmt_news_um_corpus_dict_90000'


o_train = Train()
o_train.train()
o_train.test(load_model=False)
o_train.end()
