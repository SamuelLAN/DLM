import os
import random
import numpy as np
from pretrain.preprocess.config import Ids

Ids.multi_task = True
Ids.cdlm_tasks = ['translation', 'pos', 'ner', 'synonym', 'def']

from pretrain.models.transformer_cdlm_fully_share import Model
from pretrain.train.train_CDLM_base import Train as TrainBase
from pretrain.load.zh_en_wmt_news import Loader
from lib.preprocess import utils

Model.name = 'transformer_CDLM_fully_share_wmt_news'


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader

    def preprocess(self):
        """ preprocess the data to list of list token idx """
        print('\nProcessing data ... ')

        # process before CDLM
        train_src_preprocessed, train_tar_preprocessed = utils.pipeline(
            self.M.before_encode_pl, self.train_src, self.train_tar, self.M.data_params,
        )

        test_src_preprocessed, test_tar_preprocessed = utils.pipeline(
            self.M.before_encode_pl, self.test_src, self.test_tar, self.M.data_params,
        )

        del self.train_src
        del self.train_tar
        del self.test_src
        del self.test_tar

        # preprocess CDLM_translate
        train_x_t, train_y_t, train_lan_x_t, train_lan_y_t, train_pos_y_t = utils.pipeline(
            self.M.translate_encode_pl, train_src_preprocessed, train_tar_preprocessed,
            {**self.M.data_params, 'tokenizer': self.tokenizer},
        )

        test_x_t, test_y_t, test_lan_x_t, test_lan_y_t, test_pos_y_t = utils.pipeline(
            self.M.translate_encode_pl, test_src_preprocessed, test_tar_preprocessed,
            {**self.M.data_params, 'tokenizer': self.tokenizer},
        )

        # preprocess CDLM_pos
        train_x_pos, train_y_pos, train_lan_x_pos, train_lan_y_pos, train_pos_y_pos = \
            utils.pipeline(self.M.pos_encode_pl, train_src_preprocessed, train_tar_preprocessed,
                           {**self.M.data_params, 'tokenizer': self.tokenizer})

        test_x_pos, test_y_pos, test_lan_x_pos, test_lan_y_pos, test_pos_y_pos = \
            utils.pipeline(self.M.pos_encode_pl, test_src_preprocessed, test_tar_preprocessed,
                           {**self.M.data_params, 'tokenizer': self.tokenizer})

        # preprocess CDLM_ner
        train_x_ner, train_y_ner, train_lan_x_ner, train_lan_y_ner, train_pos_y_ner = \
            utils.pipeline(self.M.ner_encode_pl, train_src_preprocessed, train_tar_preprocessed,
                           {**self.M.data_params, 'tokenizer': self.tokenizer})

        test_x_ner, test_y_ner, test_lan_x_ner, test_lan_y_ner, test_pos_y_ner = \
            utils.pipeline(self.M.ner_encode_pl, test_src_preprocessed, test_tar_preprocessed,
                           {**self.M.data_params, 'tokenizer': self.tokenizer})

        # preprocess CDLM_synonym
        train_x_syn, train_y_syn, train_lan_x_syn, train_lan_y_syn, train_pos_y_syn = \
            utils.pipeline(self.M.synonym_encode_pl, train_src_preprocessed, train_tar_preprocessed,
                           {**self.M.data_params, 'tokenizer': self.tokenizer})

        test_x_syn, test_y_syn, test_lan_x_syn, test_lan_y_syn, test_pos_y_syn = \
            utils.pipeline(self.M.synonym_encode_pl, test_src_preprocessed, test_tar_preprocessed,
                           {**self.M.data_params, 'tokenizer': self.tokenizer})

        # preprocess CDLM_definition
        train_x_def, train_y_def, train_lan_x_def, train_lan_y_def, train_pos_y_def = \
            utils.pipeline(self.M.def_encode_pl, train_src_preprocessed, train_tar_preprocessed,
                           {**self.M.data_params, 'tokenizer': self.tokenizer})

        test_x_def, test_y_def, test_lan_x_def, test_lan_y_def, test_pos_y_def = \
            utils.pipeline(self.M.def_encode_pl, test_src_preprocessed, test_tar_preprocessed,
                           {**self.M.data_params, 'tokenizer': self.tokenizer})

        # release some storage
        del train_src_preprocessed
        del train_tar_preprocessed
        del test_src_preprocessed
        del test_tar_preprocessed

        # merge data
        self.train_x = np.vstack([train_x_t, train_x_pos, train_x_ner, train_x_syn, train_x_def])
        self.train_y = np.vstack([train_y_t, train_y_pos, train_y_ner, train_y_syn, train_y_def])
        self.train_lan_x = np.vstack(
            [train_lan_x_t, train_lan_x_pos, train_lan_x_ner, train_lan_x_syn, train_lan_x_def])
        self.train_lan_y = np.vstack(
            [train_lan_y_t, train_lan_y_pos, train_lan_y_ner, train_lan_y_syn, train_lan_y_def])
        self.train_pos_y = np.vstack(
            [train_pos_y_t, train_pos_y_pos, train_pos_y_ner, train_pos_y_syn, train_pos_y_def])

        self.test_x = np.vstack([test_x_t, test_x_pos, test_x_ner, test_x_syn, test_x_def])
        self.test_y = np.vstack([test_y_t, test_y_pos, test_y_ner, test_y_syn, test_y_def])
        self.test_lan_x = np.vstack([test_lan_x_t, test_lan_x_pos, test_lan_x_ner, test_lan_x_syn, test_lan_x_def])
        self.test_lan_y = np.vstack([test_lan_y_t, test_lan_y_pos, test_lan_y_ner, test_lan_y_syn, test_lan_y_def])
        self.test_pos_y = np.vstack([test_pos_y_t, test_pos_y_pos, test_pos_y_ner, test_pos_y_syn, test_pos_y_def])

        # shuffle data
        train_data = list(zip(self.train_x, self.train_y, self.train_lan_x, self.train_lan_y, self.train_pos_y))
        test_data = list(zip(self.test_x, self.test_y, self.test_lan_x, self.test_lan_y, self.test_pos_y))

        random.seed(42)
        random.shuffle(train_data)
        random.seed(42)
        random.shuffle(test_data)

        self.train_x, self.train_y, self.train_lan_x, self.train_lan_y, self.train_pos_y = list(zip(*train_data))
        self.test_x, self.test_y, self.test_lan_x, self.test_lan_y, self.test_pos_y = list(zip(*test_data))

        # convert to array
        def convert_arr(*args):
            return list(map(lambda x: np.array(x), args))

        self.train_x, self.train_y, self.train_lan_x, self.train_lan_y, self.train_pos_y = convert_arr(
            self.train_x, self.train_y, self.train_lan_x, self.train_lan_y, self.train_pos_y
        )
        self.test_x, self.test_y, self.test_lan_x, self.test_lan_y, self.test_pos_y = convert_arr(
            self.test_x, self.test_y, self.test_lan_x, self.test_lan_y, self.test_pos_y
        )

        # get vocabulary size
        self.vocab_size = self.tokenizer.vocab_size

        print('\nFinish preprocessing ')


o_train = Train(use_cache=True)
o_train.train()
o_train.test(load_model=False)
