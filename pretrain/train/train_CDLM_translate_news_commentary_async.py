import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_root_dir)
sys.path.append(root_dir)

from pretrain.train.train_CDLM_base_async import Train as TrainBase
from pretrain.models.transformer_cdlm_translate import Model
from pretrain.load.async_loader import Loader

Model.name = 'transformer_CDLM_translate_news_commentary_3.0_async'


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader

    train_preprocess_dirs = ['cdlm_translate_train_3.0']
    val_preprocess_dirs = ['cdlm_translate_test']
    tokenizer_dir = 'cdlm_translate_train_3.0'


o_train = Train()
o_train.train()
o_train.test(load_model=False)
o_train.end()
