import os
import time
import math
import numpy as np
from lib import utils
from lib.tf_learning_rate.warmup_then_down import CustomSchedule
from lib.tf_callback.board import Board
from lib.tf_callback.saver import Saver
from lib.tf_models.transformer import Transformer
import tensorflow as tf

keras = tf.keras
tfv1 = tf.compat.v1


class BaseModel:
    name = 'transformer'

    data_params = {
        'src_vocab_size': 2 ** 13,  # approximate
        'tar_vocab_size': 2 ** 13,  # approximate
        'max_src_seq_len': 50,
        'max_tar_seq_len': 60,
        'incr': 3,
    }

    model_params = {
        'drop_rate': 0.1,
    }

    train_params = {
        'learning_rate': 3e-3,
        # 'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 64,
        'epoch': 300,
        'early_stop': 30,
    }

    tb_params = {
        'histogram_freq': 0,
        'update_freq': 'epoch',
        'write_grads': False,
        'write_graph': True,
    }

    compile_params = {
        'optimizer': tfv1.train.AdamOptimizer(learning_rate=train_params['learning_rate']),
        'loss': keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none'),
        'customize_loss': True,
        'metrics': [],
    }

    monitor_params = {
        'name': 'loss',
        'mode': 'min',  # for the "name" monitor, the "min" is best;
        'for_start': 'loss',
        'for_start_value': 3.,
        'for_start_mode': 'min',
    }

    checkpoint_params = {
        'load_model': [],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['name']
    }

    TIME = str(time.strftime('%Y_%m_%d_%H_%M_%S'))

    @property
    def tar_start_token_idx(self):
        return self.target_vocab_size + 1

    @property
    def tar_end_token_idx(self):
        return self.target_vocab_size + 2

    def __init__(self, input_vocab_size, target_vocab_size, name='', finish_train=False):
        self.name = name if name else self.name
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.__finish_train = finish_train

        # create directories for tensorboard files and model files
        self.__create_dir()

        # build models
        self.build()

        # for using model.fit, set callbacks for the training process
        self.set_callbacks()

    def __create_dir(self):
        # create tensorboard path
        self.__tb_dir = utils.create_dir_in_root('runtime', 'tensorboard', self.name, self.TIME)

        # create model path
        self.__model_dir = utils.create_dir_in_root('runtime', 'models', self.name, self.TIME)
        self.__checkpoint_path = os.path.join(self.__model_dir, self.name + self.checkpoint_params['extend_name'])

    def build(self):
        self.model = Transformer(
            num_layers=self.model_params['num_layers'],
            d_model=self.model_params['dim_model'],
            num_heads=self.model_params['num_heads'],
            d_ff=self.model_params['ff_units'],
            input_vocab_size=self.input_vocab_size + self.data_params['incr'],
            target_vocab_size=self.target_vocab_size + self.data_params['incr'],
            max_pe_input=self.model_params['max_pe_input'],
            max_pe_target=self.model_params['max_pe_target'],
            drop_rate=self.model_params['drop_rate'],
        )

    def set_callbacks(self):
        """ if using model.fit to train model,
              then we need to set callbacks for the training process """
        # callback for tensorboard
        callback_tf_board = Board(log_dir=self.__tb_dir,
                                  histogram_freq=self.tb_params['histogram_freq'],
                                  write_grads=self.tb_params['write_grads'],
                                  write_graph=self.tb_params['write_graph'],
                                  write_images=False,
                                  profile_batch=0,
                                  update_freq=self.tb_params['update_freq'])
        callback_tf_board.set_model(self.model)

        # callback for saving model and early stopping
        callback_saver = Saver(self.__checkpoint_path,
                               self.monitor_params['name'],
                               self.monitor_params['mode'],
                               self.train_params['early_stop'],
                               self.monitor_params['for_start'],
                               self.monitor_params['for_start_value'],
                               self.monitor_params['for_start_mode'])
        callback_saver.set_model(self.model)

        self.callbacks = [callback_tf_board, callback_saver]

    def __compile(self):
        loss = self.loss if self.compile_params['customize_loss'] else self.compile_params['loss']
        self.model.compile(optimizer=self.compile_params['optimizer'],
                           loss=loss,
                           metrics=self.compile_params['metrics'])

    @staticmethod
    def __get_best_model_path(model_dir):
        """ get the best model within model_dir """
        file_list = os.listdir(model_dir)
        file_list.sort()
        return os.path.join(model_dir, file_list[-1])

    def load_model(self, model_dir='', x=None, y=None):
        # get model path
        model_dir = model_dir if model_dir else self.__model_dir
        model_path = self.__get_best_model_path(model_dir)

        # empty fit, to prevent error from occurring when loading model
        self.model.fit(x, y, epochs=0) if not isinstance(x, type(None)) else None

        # load model weights
        self.model.load_weights(model_path)
        print('Successfully loading weights from %s ' % model_path)

    def train(self, train_x, train_y, val_x=None, val_y=None):
        # compile model
        self.__compile()

        # if we want to load a trained model
        if self.checkpoint_params['load_model']:
            model_dir = utils.create_dir_in_root(*(['runtime', 'models'] + self.checkpoint_params['load_model']))
            self.load_model(model_dir, [train_x[0][:1], train_x[1][:1]], train_y[:1])

        if not self.__finish_train:
            # fit model
            self.model.fit([train_x, train_y], train_y,
                           epochs=self.train_params['epoch'],
                           batch_size=self.train_params['batch_size'],
                           validation_data=([val_x, val_y], val_y) if not isinstance(val_x, type(None)) else None,
                           callbacks=self.callbacks,
                           verbose=2)

            # load the best model so that it could be tested
            self.load_model()

            self.__finish_train = True

    def loss(self, y_true, y_pred, from_logits=True, label_smoothing=0):
        # reshape y_true to (batch_size, max_tar_seq_len)
        y_true = tf.reshape(y_true, [-1, y_pred.shape[1]])

        # calculate the padding mask
        mask = tf.cast(tf.math.not_equal(y_true, 0), y_pred.dtype)

        # calculate the loss
        loss_ = self.compile_params['loss'](y_true, y_pred)

        # remove the padding part's loss by timing the mask
        loss_ *= mask

        # calculate the mean loss
        return tf.reduce_mean(loss_)

    def evaluate_encoded(self, list_of_list_src_token_idx):
        """ translate list of list encoded token idx; the results are also encoded """
        return self.model.evaluate_list_of_list_token_idx(list_of_list_src_token_idx,
                                                          self.tar_start_token_idx,
                                                          self.tar_end_token_idx,
                                                          self.data_params['max_tar_seq_len'])
