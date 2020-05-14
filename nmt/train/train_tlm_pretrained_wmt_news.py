import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_sub_root_dir)
sys.path.append(sub_root_dir)
sys.path.append(root_dir)

from nmt.models.transformer_tlm_pretrained import Model
from nmt.load.zh_en_wmt_news import Loader
from nmt.train.train_base import Train as TrainBase

Model.name = 'transformer_nmt_TLM_pretrained_wmt_news'
Model.checkpoint_params['load_model'] = ['transformer_TLM_wmt_news', '2020_05_14_02_12_46']


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader


o_train = Train(use_cache=True)
o_train.train()
o_train.test(load_model=False)
