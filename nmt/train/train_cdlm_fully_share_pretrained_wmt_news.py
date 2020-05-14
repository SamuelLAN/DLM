import os
from pretrain.preprocess.config import Ids

Ids.multi_task = True
Ids.cdlm_tasks = ['translation', 'pos', 'ner', 'synonym', 'def']

from nmt.models.transformer_cdlm_fully_share_pretrained import Model
from nmt.load.zh_en_wmt_news import Loader
from nmt.train.train_base import Train as TrainBase

Model.name = 'transformer_nmt_CDLM_fully_share_pretrained_wmt_news'


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader


o_train = Train(use_cache=True)
o_train.train()
o_train.test(load_model=False)
