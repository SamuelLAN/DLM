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

from nmt.models.transformer_cdlm_fully_share_pretrained import Model
from nmt.load.zh_en_wmt_news import Loader
from nmt.train.train_base import Train as TrainBase

Model.name = 'transformer_nmt_CDLM_fully_share_pretrained_wmt_news'
Model.checkpoint_params['load_model'] = ['transformer_CDLM_fully_share_wmt_news', '2020_05_16_06_34_50']
Model.checkpoint_params['load_model'] = [Model.name, '2020_05_17_01_47_41']


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader


o_train = Train(use_cache=True)
# o_train.train()
o_train.test(load_model=True)
