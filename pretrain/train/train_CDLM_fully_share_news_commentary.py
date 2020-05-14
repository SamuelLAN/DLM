import os
from pretrain.preprocess.config import Ids

Ids.multi_task = True
Ids.cdlm_tasks = ['translation', 'pos', 'ner', 'synonym', 'def']

from pretrain.models.transformer_cdlm_fully_share import Model
from pretrain.train.train_CDLM_fully_share_wmt_news import Train as TrainBase
from pretrain.load.zh_en_news_commentary import Loader


Model.name = 'transformer_CDLM_fully_share_news_commentary'


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader


o_train = Train(use_cache=True)
o_train.train()
o_train.test(load_model=False)
