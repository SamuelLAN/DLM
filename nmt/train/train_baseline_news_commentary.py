import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_root_dir)
sys.path.append(root_dir)

# from nmt.models.transformer_baseline import Model
from nmt.models.other_lans.transformer_zh_en import Model
from nmt.load.zh_en_news_commentary import Loader
from nmt.train.train_base import Train as TrainBase

# Model.name = 'transformer_nmt_baseline_news_commentary'
Model.checkpoint_params['load_model'] = ['transformer_for_nmt_share_emb_zh_word_level_wmt_news', 'epoch_30']


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader


o_train = Train(use_cache=True)
# o_train.train()
o_train.test(load_model=True)
