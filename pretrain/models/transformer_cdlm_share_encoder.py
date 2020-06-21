from lib.tf_learning_rate.warmup_then_down import CustomSchedule
from nmt.preprocess.inputs import noise_pl, tfds_share_pl, zh_en
from pretrain.preprocess.inputs import CDLM_translation_v2 as CDLM_translation
from pretrain.preprocess.inputs import CDLM_pos, CDLM_ner, CDLM_synonym, CDLM_definition
from pretrain.preprocess.inputs.sampling import sample_pl
from pretrain.preprocess.inputs.pl import CDLM_encode, sent_2_tokens
from pretrain.preprocess.inputs.decode import decode_pl as d_pl
from lib.tf_metrics.pretrain import tf_accuracy, tf_perplexity
from pretrain.models.transformer_cdlm_translate import Model as BaseModel
from pretrain.preprocess.config import Ids
from lib.tf_models.transformer_lan_soft_pos import Transformer
from lib.tf_callback.board import Board
from lib.tf_callback.saver import Saver
from lib import utils
import os
import math
import tensorflow as tf
import types

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_CDLM_fully_share'

    pretrain_params = {
        'keep_origin_rate': 0.2,
        # 'TLM_ratio': 0.7,
        'max_ratio': 0.3,
        'max_num': 4,
    }

    sample_params = {
        'translation': 3.0,
        'pos': 3.0,
        'ner': 3.0,
        'synonym': 2.0,
        'definition': 0.5,
    }

    data_params = {
        **BaseModel.data_params,
        'vocab_size': 80000,  # approximate
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'max_src_ground_seq_len': 24,
        'max_tar_ground_seq_len': 24,
        'sample_ratio': 1.0,  # sample "sample_rate" percentage of data into dataset; > 0
        'over_sample_rate': sample_params,
        'input_incr': Ids.end_cdlm_def + Ids.pos_ids + Ids.ner_ids + 1,  # <start>, <end>, <pad>, <mask>
        'class_incr': Ids.end_cdlm_def + Ids.pos_ids + Ids.ner_ids + 1,  # <start>, <end>, <pad>, <mask>
    }

    preprocess_pl = zh_en.seg_zh_by_jieba_pipeline + noise_pl.remove_noise
    tokenizer_pl = preprocess_pl + tfds_share_pl.train_tokenizer

    before_encode_pl = preprocess_pl + sent_2_tokens

    translate_encode_pl = sample_pl(sample_params['translation']) + CDLM_translation.combine_pl(
        **pretrain_params) + CDLM_encode
    pos_encode_pl = sample_pl(sample_params['pos']) + CDLM_pos.combine_pl(**pretrain_params) + CDLM_encode
    ner_encode_pl = sample_pl(sample_params['ner']) + CDLM_ner.combine_pl(**pretrain_params) + CDLM_encode
    synonym_encode_pl = sample_pl(sample_params['synonym']) + CDLM_synonym.combine_pl(
        **pretrain_params) + CDLM_encode
    def_encode_pl = sample_pl(sample_params['definition']) + CDLM_definition.combine_pl(
        **pretrain_params) + CDLM_encode

    decode_pl = d_pl('multi', True)

    model_params = {
        **BaseModel.model_params,
        'emb_dim': 128,
        'dim_model': 128,
        'ff_units': 128,
        'num_layers': 6,
        'num_heads': 8,
        'max_pe_input': data_params['max_src_seq_len'],
        'max_pe_target': data_params['max_src_ground_seq_len'],
        'drop_rate': 0.1,
        'share_emb': True,
        'share_final': False,
        'lan_vocab_size': 4,
    }

    train_params = {
        **BaseModel.train_params,
        'learning_rate': 1e-4,
        # 'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 16,
        'epoch': 800,
        'early_stop': 20,
    }

    compile_params = {
        **BaseModel.compile_params,
        'optimizer': tfv1.train.AdamOptimizer(learning_rate=train_params['learning_rate']),
        'label_smooth': True,
        'metrics': [tf_accuracy, tf_perplexity],
    }

    monitor_params = {
        'monitor': 'val_tf_accuracy',
        'mode': 'max',  # for the "name" monitor, the "min" is best;
        'early_stop': train_params['early_stop'],
        'start_train_monitor': 'tf_accuracy',
        'start_train_monitor_value': 0.01,
        'start_train_monitor_mode': 'max',
    }

    checkpoint_params = {
        'load_model': [],
        'load_model_word_translate': [],  # [name, time]
        'load_model_cdlm_translate': [],  # [name, time]
        'load_model_cdlm_ner': [],  # [name, time]
        'load_model_cdlm_pos': [],  # [name, time]
        'load_model_cdlm_synonym': [],  # [name, time]
        # 'load_model': ['transformer_for_MLM_zh_en', '2020_04_26_15_19_16'],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['monitor']
    }

    evaluate_dict = {

    }

    def create_dir(self):
        # create tensorboard path
        self.tb_dir_word_translate = utils.create_dir_in_root('runtime', 'tensorboard', self.name, 'word_translate',
                                                              self.TIME)
        self.tb_dir_cdlm_translate = utils.create_dir_in_root('runtime', 'tensorboard', self.name, 'cdlm_translate',
                                                              self.TIME)
        self.tb_dir_cdlm_ner = utils.create_dir_in_root('runtime', 'tensorboard', self.name, 'cdlm_ner', self.TIME)
        self.tb_dir_cdlm_pos = utils.create_dir_in_root('runtime', 'tensorboard', self.name, 'cdlm_pos', self.TIME)
        self.tb_dir_cdlm_synonym = utils.create_dir_in_root('runtime', 'tensorboard', self.name, 'cdlm_synonym',
                                                            self.TIME)

        # create model path
        self.model_dir_word_translate = utils.create_dir_in_root('runtime', 'models', self.name, 'word_translate',
                                                                 self.TIME)
        self.model_dir_cdlm_translate = utils.create_dir_in_root('runtime', 'models', self.name, 'cdlm_translate',
                                                                 self.TIME)
        self.model_dir_cdlm_ner = utils.create_dir_in_root('runtime', 'models', self.name, 'cdlm_ner', self.TIME)
        self.model_dir_cdlm_pos = utils.create_dir_in_root('runtime', 'models', self.name, 'cdlm_pos', self.TIME)
        self.model_dir_cdlm_synonym = utils.create_dir_in_root('runtime', 'models', self.name, 'cdlm_synonym',
                                                               self.TIME)

        self.checkpoint_path_word_translate = os.path.join(self.model_dir_word_translate,
                                                           self.name + self.checkpoint_params['extend_name'])
        self.checkpoint_path_cdlm_translate = os.path.join(self.model_dir_cdlm_translate,
                                                           self.name + self.checkpoint_params['extend_name'])
        self.checkpoint_path_cdlm_ner = os.path.join(self.model_dir_cdlm_ner,
                                                     self.name + self.checkpoint_params['extend_name'])
        self.checkpoint_path_cdlm_pos = os.path.join(self.model_dir_cdlm_pos,
                                                     self.name + self.checkpoint_params['extend_name'])
        self.checkpoint_path_cdlm_synonym = os.path.join(self.model_dir_cdlm_synonym,
                                                         self.name + self.checkpoint_params['extend_name'])

        self.tokenizer_dir = utils.create_dir_in_root('runtime', 'tokenizer', self.name, self.TIME)

    def build(self):
        params = {
            'num_layers': self.model_params['num_layers'],
            'd_model': self.model_params['dim_model'],
            'num_heads': self.model_params['num_heads'],
            'd_ff': self.model_params['ff_units'],
            'input_vocab_size': self.input_vocab_size + self.data_params['input_incr'],
            'target_vocab_size': self.num_classes,
            'max_pe_input': self.model_params['max_pe_input'],
            'max_pe_target': self.model_params['max_pe_target'] - 1,
            'drop_rate': self.model_params['drop_rate'],
            'share_emb': self.model_params['share_emb'],
            'share_final': self.model_params['share_final'],
            'lan_vocab_size': self.model_params['lan_vocab_size']
        }

        self.word_translate_model = Transformer(**params)

        self.cdlm_translate_model = Transformer(**{**params, 'encoder': self.word_translate_model.encoder})
        self.cdlm_ner_model = Transformer(**{**params, 'encoder': self.word_translate_model.encoder})
        self.cdlm_pos_model = Transformer(**{**params, 'encoder': self.word_translate_model.encoder})
        self.cdlm_synonym_model = Transformer(**{**params, 'encoder': self.word_translate_model.encoder})

    def __get_callbacks(self, tb_dir, checkpoint_path, model):
        # callback for tensorboard
        callback_tf_board = Board(log_dir=tb_dir, **self.tb_params)
        callback_tf_board.set_model(model)

        # callback for saving model and early stopping
        callback_saver = Saver(checkpoint_path, **self.monitor_params)
        callback_saver.set_model(model)

        return [callback_tf_board, callback_saver]

    def set_callbacks(self):
        """ if using model.fit to train model,
              then we need to set callbacks for the training process """
        self.callbacks_word_translate = self.__get_callbacks(self.tb_dir_word_translate,
                                                             self.checkpoint_path_word_translate,
                                                             self.word_translate_model)

        self.callbacks_cdlm_translate = self.__get_callbacks(self.tb_dir_cdlm_translate,
                                                             self.checkpoint_path_cdlm_translate,
                                                             self.cdlm_translate_model)

        self.callbacks_cdlm_ner = self.__get_callbacks(self.tb_dir_cdlm_ner,
                                                       self.checkpoint_path_cdlm_ner,
                                                       self.cdlm_ner_model)

        self.callbacks_cdlm_pos = self.__get_callbacks(self.tb_dir_cdlm_pos,
                                                       self.checkpoint_path_cdlm_pos,
                                                       self.cdlm_pos_model)

        self.callbacks_cdlm_synonym = self.__get_callbacks(self.tb_dir_cdlm_synonym,
                                                           self.checkpoint_path_cdlm_synonym,
                                                           self.cdlm_synonym_model)

    def compile(self):
        loss = self.loss if self.compile_params['customize_loss'] else self.compile_params['loss']
        self.word_translate_model.compile(optimizer=self.compile_params['optimizer'],
                                          loss=loss,
                                          metrics=self.compile_params['metrics'])

        self.cdlm_translate_model.compile(optimizer=self.compile_params['optimizer'],
                                          loss=loss,
                                          metrics=self.compile_params['metrics'])

        self.cdlm_ner_model.compile(optimizer=self.compile_params['optimizer'],
                                    loss=loss,
                                    metrics=self.compile_params['metrics'])

        self.cdlm_pos_model.compile(optimizer=self.compile_params['optimizer'],
                                    loss=loss,
                                    metrics=self.compile_params['metrics'])

        self.cdlm_synonym_model.compile(optimizer=self.compile_params['optimizer'],
                                        loss=loss,
                                        metrics=self.compile_params['metrics'])

    def __load_model(self, _model, model_dir='', x=None, y=None):
        # get model path
        model_path = self.__get_best_model_path(model_dir)

        # empty fit, to prevent error from occurring when loading model
        _model.fit(x, y, epochs=0) if not isinstance(x, type(None)) else None

        # load model weights
        _model.load_weights(model_path)
        print('Successfully loading weights from %s ' % model_path)

    def train_multi(self, train_x_word_translate, train_y_word_translate,
                    train_x_cdlm_translate, train_y_cdlm_translate,
                    train_x_cdlm_ner, train_y_cdlm_ner,
                    train_x_cdlm_pos, train_y_cdlm_pos,
                    train_x_cdlm_synonym, train_y_cdlm_synonym,
                    val_x_word_translate=None, val_y_word_translate=None,
                    val_x_cdlm_translate=None, val_y_cdlm_translate=None,
                    val_x_cdlm_ner=None, val_y_cdlm_ner=None,
                    val_x_cdlm_pos=None, val_y_cdlm_pos=None,
                    val_x_cdlm_synonym=None, val_y_cdlm_synonym=None,
                    train_size_word_translate=None, val_size_word_translate=None,
                    train_size_cdlm_translate=None, val_size_cdlm_translate=None,
                    train_size_cdlm_ner=None, val_size_cdlm_ner=None,
                    train_size_cdlm_pos=None, val_size_cdlm_pos=None,
                    train_size_cdlm_synonym=None, val_size_cdlm_synonym=None):
        # compile model
        self.compile()

        # # if we want to load a trained model
        # if self.checkpoint_params['load_model']:
        #     _dirs = self.checkpoint_params['load_model']
        #     model_dir = utils.create_dir_in_root(*(['runtime', 'models', _dirs[0], 'word_translate', _dirs[1]]))
        #
        #     batch_x = [v[:1] for v in train_x] if isinstance(train_x, tuple) else train_x[:1]
        #     self.load_model(model_dir, batch_x, train_y[:1])

        _batch_size = self.train_params['batch_size']
        _epochs = self.train_params['epoch']

        if not self.__finish_train:
            not_generator_word_translate = not isinstance(train_x_word_translate, types.GeneratorType)
            batch_size = _batch_size if not_generator_word_translate else None
            steps_per_epoch = None if not_generator_word_translate else int(math.ceil(train_size_word_translate / _batch_size))
            validation_steps = None if not_generator_word_translate else int(math.ceil(val_size_word_translate / _batch_size))

            not_generator = not isinstance(train_x_word_translate, types.GeneratorType)
            batch_size = _batch_size if not_generator else None
            steps_per_epoch = None if not_generator else int(math.ceil(train_size_word_translate / _batch_size))
            validation_steps = None if not_generator else int(math.ceil(val_size_word_translate / _batch_size))

            not_generator = not isinstance(train_x_word_translate, types.GeneratorType)
            batch_size = _batch_size if not_generator else None
            steps_per_epoch = None if not_generator else int(math.ceil(train_size_word_translate / _batch_size))
            validation_steps = None if not_generator else int(math.ceil(val_size_word_translate / _batch_size))

            not_generator = not isinstance(train_x_word_translate, types.GeneratorType)
            batch_size = _batch_size if not_generator else None
            steps_per_epoch = None if not_generator else int(math.ceil(train_size_word_translate / _batch_size))
            validation_steps = None if not_generator else int(math.ceil(val_size_word_translate / _batch_size))

            not_generator = not isinstance(train_x_word_translate, types.GeneratorType)
            batch_size = _batch_size if not_generator else None
            steps_per_epoch = None if not_generator else int(math.ceil(train_size_word_translate / _batch_size))
            validation_steps = None if not_generator else int(math.ceil(val_size_word_translate / _batch_size))

            for _epoch in range(int(_epochs / 5)):
                # fit model
                self.word_translate_model.fit(train_x_word_translate,
                                              train_y_word_translate,
                                              epochs=5,
                                              batch_size=batch_size,
                                              steps_per_epoch=steps_per_epoch,
                                              validation_data=(val_x_word_translate, val_y_word_translate)
                                              if not isinstance(val_x_word_translate, type(None)) else None,
                                              validation_steps=validation_steps,
                                              callbacks=self.callbacks_word_translate,
                                              verbose=2)

                self.cdlm_translate_model.fit(train_x_cdlm_translate,
                                              train_y_cdlm_translate,
                                              epochs=2,
                                              batch_size=batch_size,
                                              steps_per_epoch=steps_per_epoch,
                                              validation_data=(val_x_word_translate, val_y_word_translate)
                                              if not isinstance(val_x_word_translate, type(None)) else None,
                                              validation_steps=validation_steps,
                                              callbacks=self.callbacks_word_translate,
                                              verbose=2)

                self.cdlm_ner_model.fit(train_x_word_translate,
                                        train_y_word_translate,
                                        epochs=5,
                                        batch_size=batch_size,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=(val_x_word_translate, val_y_word_translate)
                                        if not isinstance(val_x_word_translate, type(None)) else None,
                                        validation_steps=validation_steps,
                                        callbacks=self.callbacks_word_translate,
                                        verbose=2)

                self.cdlm_pos_model.fit(train_x_word_translate,
                                        train_y_word_translate,
                                        epochs=5,
                                        batch_size=batch_size,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=(val_x_word_translate, val_y_word_translate)
                                        if not isinstance(val_x_word_translate, type(None)) else None,
                                        validation_steps=validation_steps,
                                        callbacks=self.callbacks_word_translate,
                                        verbose=2)

                self.cdlm_synonym_model.fit(train_x_word_translate,
                                            train_y_word_translate,
                                            epochs=5,
                                            batch_size=batch_size,
                                            steps_per_epoch=steps_per_epoch,
                                            validation_data=(val_x_word_translate, val_y_word_translate)
                                            if not isinstance(val_x_word_translate, type(None)) else None,
                                            validation_steps=validation_steps,
                                            callbacks=self.callbacks_word_translate,
                                            verbose=2)

            # load the best model so that it could be tested
            self.load_model()

            self.__finish_train = True
