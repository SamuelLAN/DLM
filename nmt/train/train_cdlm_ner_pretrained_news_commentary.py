import os
from nmt.models.transformer_cdlm_ner_pretrained import Model
from nmt.load.zh_en_news_commentary import Loader
from nmt.train.train_base import Train as TrainBase

Model.name = 'transformer_nmt_CDLM_ner_pretrained_news_commentary'


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader


o_train = Train(use_cache=True)
o_train.train()
o_train.test(load_model=False)
