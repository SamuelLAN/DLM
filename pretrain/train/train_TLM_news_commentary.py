import os
from pretrain.train.train_base import Train as TrainBase
from pretrain.models.transformer_tlm import Model
from pretrain.load.zh_en_news_commentary import Loader

Model.name = 'transformer_TLM_news_commentary'


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader


o_train = Train(use_cache=True)
o_train.train()
o_train.test(load_model=False)
