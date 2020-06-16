import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_root_dir)
sys.path.append(root_dir)

from nmt.models.transformer_baseline_ro_en import Model
from nmt.load.ro_en import Loader
from nmt.train.train_base import Train as TrainBase

Model.name = 'transformer_nmt_baseline_ro_en'
# Model.checkpoint_params['load_model'] = ['transformer_for_nmt_share_emb_zh_word_level_wmt_news', '2020_04_25_12_59_02']


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader


o_train = Train(use_cache=True)
o_train.train()
o_train.test(load_model=False)
