import os
import time
import math
import numpy as np
from lib import utils
from lib.tf_learning_rate.warmup_then_down import CustomSchedule
from lib.tf_callback.board import Board
from lib.tf_callback.saver import Saver
from lib.tf_models.transformer import Transformer
from lib.metrics import metrics
import tensorflow as tf

keras = tf.keras
tfv1 = tf.compat.v1


class BaseModel:
    name = 'transformer'

    data_params = {
        'vocab_size': 2 ** 13,  # approximate
        'src_vocab_size': 2 ** 13,  # approximate
        'tar_vocab_size': 2 ** 13,  # approximate
        'max_src_seq_len': 50,
        'max_tar_seq_len': 60,
        'input_incr': 3,  # <start>, <end>, <pad>
        'class_incr': 3,  # <start>, <end>, <pad>
    }

    model_params = {
        'drop_rate': 0.1,
        'top_k': 3,
        'get_random': False,
        'share_emb': True,
        'share_final': False,
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
        'loss': keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
        'customize_loss': True,
        'label_smooth': True,
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
        self.num_classes = self.target_vocab_size + self.data_params['class_incr']
        self.__finish_train = finish_train

        # create directories for tensorboard files and model files
        self.__create_dir()

        # build models
        self.build()

        # for using model.fit, set callbacks for the training process
        self.set_callbacks()

        self.__global_step = tfv1.train.get_or_create_global_step()

    def __create_dir(self):
        # create tensorboard path
        self.tb_dir = utils.create_dir_in_root('runtime', 'tensorboard', self.name, self.TIME)

        # create model path
        self.model_dir = utils.create_dir_in_root('runtime', 'models', self.name, self.TIME)
        self.checkpoint_path = os.path.join(self.model_dir, self.name + self.checkpoint_params['extend_name'])

        self.tokenizer_dir = utils.create_dir_in_root('runtime', 'tokenizer', self.name, self.TIME)

    def build(self):
        self.model = Transformer(
            num_layers=self.model_params['num_layers'],
            d_model=self.model_params['dim_model'],
            num_heads=self.model_params['num_heads'],
            d_ff=self.model_params['ff_units'],
            input_vocab_size=self.input_vocab_size + self.data_params['input_incr'],
            target_vocab_size=self.num_classes,
            max_pe_input=self.model_params['max_pe_input'],
            max_pe_target=self.model_params['max_pe_target'] - 1,
            drop_rate=self.model_params['drop_rate'],
            share_emb=self.model_params['share_emb'],
            share_final=self.model_params['share_final'],
        )

    def set_callbacks(self):
        """ if using model.fit to train model,
              then we need to set callbacks for the training process """
        # callback for tensorboard
        callback_tf_board = Board(log_dir=self.tb_dir,
                                  histogram_freq=self.tb_params['histogram_freq'],
                                  write_grads=self.tb_params['write_grads'],
                                  write_graph=self.tb_params['write_graph'],
                                  write_images=False,
                                  profile_batch=0,
                                  update_freq=self.tb_params['update_freq'])
        callback_tf_board.set_model(self.model)

        # callback for saving model and early stopping
        callback_saver = Saver(self.checkpoint_path,
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
        model_dir = model_dir if model_dir else self.model_dir
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
            batch_x = [v[:1] for v in train_x] if isinstance(train_x, tuple) else train_x[:1]
            self.load_model(model_dir, batch_x, train_y[:1])

        if not self.__finish_train:
            # fit model
            self.model.fit(train_x, train_y,
                           epochs=self.train_params['epoch'],
                           batch_size=self.train_params['batch_size'],
                           validation_data=(val_x, val_y) if not isinstance(val_x, type(None)) else None,
                           callbacks=self.callbacks,
                           verbose=2)

            # load the best model so that it could be tested
            self.load_model()

            self.__finish_train = True

    def loss(self, y_true, y_pred, from_logits=True, label_smoothing=0):
        # reshape y_true to (batch_size, max_tar_seq_len)
        y_true = tf.reshape(y_true, [-1, y_pred.shape[1]])
        mask = tf.expand_dims(tf.cast(tf.logical_not(tf.equal(y_true, 0)), y_pred.dtype), axis=-1)
        y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), y_pred.shape[-1]), y_pred.dtype)

        epison = 0.0001

        # # label smoothing
        if self.compile_params['label_smooth']:
            label_smoothing = 0.1
            num_classes = tf.cast(self.num_classes, y_pred.dtype)
            y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        # loss = - (y_true * (1 - y_pred) * tf.math.log(y_pred + epison) + (1 - y_true) * y_pred * tf.math.log(1 - y_pred + epison))
        loss = - (y_true * tf.math.log(y_pred + epison) + (1 - y_true) * tf.math.log(1 - y_pred + epison))

        loss *= mask
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        return loss

    def loss2(self, y_true, y_pred, from_logits=False, label_smoothing=0):
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
        batch_size = self.train_params['batch_size']
        steps = int(np.ceil(len(list_of_list_src_token_idx) / batch_size))

        # evaluate in batch so that OOM would not happen
        predictions = []
        for step in range(steps):
            progress = float(step + 1) / steps * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

            batch_x = list_of_list_src_token_idx[step * batch_size: (step + 1) * batch_size]
            predictions += self.model.evaluate_list_of_list_token_idx(batch_x,
                                                                      self.tar_start_token_idx,
                                                                      self.tar_end_token_idx,
                                                                      self.data_params['max_tar_seq_len'])
        return predictions

    def evaluate_encoded_beam_search(self, list_of_list_src_token_idx):
        """ translate list of list encoded token idx; the results are also encoded """
        return self.model.beam_search_list_of_list_token_idx(list_of_list_src_token_idx,
                                                             self.tar_start_token_idx,
                                                             self.tar_end_token_idx,
                                                             self.data_params['max_tar_seq_len'],
                                                             self.model_params['top_k'],
                                                             self.model_params['get_random'])

    def calculate_loss_for_encoded(self, src_encode_data, tar_encode_data, dataset=''):
        """ evaluate the Loss according to the encoded src language data (list_of_list_token_idx)
                                and the target reference (list of sentences) """
        print('\nstart calculating loss for {} ...'.format(dataset))
        batch_size = self.train_params['batch_size']
        steps = int(np.ceil(len(tar_encode_data) / batch_size))

        # evaluate in batch so that OOM would not happen
        loss = []
        for step in range(steps):
            progress = float(step + 1) / steps * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

            batch_x = src_encode_data[step * batch_size: (step + 1) * batch_size]
            batch_y = tar_encode_data[step * batch_size: (step + 1) * batch_size]

            batch_input = [batch_x, batch_y[:, :-1]]
            batch_output = batch_y[:, 1:]

            predictions = self.model(batch_input, training=False)
            loss.append(self.loss(batch_output, predictions))

        loss = np.mean(loss)
        print('{} loss: {}'.format(dataset, loss))
        return loss

    def evaluate_metrics_for_encoded(self, dataset, *args):
        acc_list = []
        ppl_list = []
        loss_list = []

        print('\nstart calculating metrics for {} ...'.format(dataset))
        batch_size = self.train_params['batch_size']
        steps = int(np.ceil(len(args[2]) / batch_size))

        # evaluate in batch so that OOM would not happen
        for step in range(steps):
            progress = float(step + 1) / steps * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

            start = step * batch_size
            end = start + batch_size

            batch_input = [v[start: end] for v in args[:2]] + [v[start: end, :-1] for v in args[2:]]
            batch_output = args[2][start: end, 1:]

            predictions = self.model(batch_input, training=False)

            acc_list.append(metrics.accuracy(batch_output, predictions))
            ppl_list.append(metrics.perplexity(batch_output, predictions))
            loss_list.append(self.loss(batch_output, predictions))

        acc = np.mean(acc_list)
        ppl = np.mean(ppl_list)
        loss = np.mean(loss_list)

        print('{} loss: {}, acc: {}, ppl: {}'.format(dataset, loss, acc, ppl))
        return loss, acc, ppl

    def eval_example_for_pretrain(self, *args):
        _input = list(args[:2]) + [v[:, :-1] for v in args[2:]]
        _output = args[2][:, 1:]
        predictions = self.model(_input, training=False)
        return np.argmax(predictions, axis=-1)

    @staticmethod
    def decode_encoded_data(_decode_pl, predictions, _tokenizer, verbose=False):
        from lib.preprocess.utils import pipeline
        return pipeline(_decode_pl, predictions, None, {'tokenizer': _tokenizer}, verbose=verbose)
