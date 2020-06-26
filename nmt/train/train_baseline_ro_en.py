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
Model.checkpoint_params['load_model'] = ['transformer_nmt_baseline_wmt_news_with_zh_en_ro_60k_pretrained', '2020_06_26_02_45_02']


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader
    tokenizer_dir = 'zh_en_ro_news_commentary_wmt_news_um_corpus_dict_90000'


o_train = Train(use_cache=True)
o_train.train()
o_train.test(load_model=False)
