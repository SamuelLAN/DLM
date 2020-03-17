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
    model_params = {
       'drop_rate': 0.1,
    }

    train_params = {
        # 'learning_rate': 3e-3,
        'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 64,
        'epoch': 300,
        'early_stop': 30,
        'loss': keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none'),
    }

    compile_params = {
        'optimizer': tfv1.train.AdamOptimizer(learning_rate=train_params['learning_rate']),
        'loss': keras.losses.categorical_crossentropy,
        'metrics': [],
    }

    tb_params = {
        'histogram_freq': 0,
        'update_freq': 'epoch',
        'write_grads': False,
        'write_graph': True,
    }

    monitor_params = {
        'name': 'val_loss',
        'mode': 'min',  # for the "name" monitor, the "min" is best;
        'for_start': None,
        'for_start_value': None,
    }

    checkpoint_params = {
        'load_model': [],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['name']
    }

    evaluate_dict = {

    }

    TIME = time.strftime('%Y_%m_%d_%H_%M_%S')

    def __init__(self, input_vocab_size, target_vocab_size, name='transformer'):
        self.name = name
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.finish_train = False

        # create directories for tensorboard files and model files
        self.__create_dir()

        # set memory growth
        self.__init_gpu_config()

        # build models
        self.build()

        # for using model.fit, set callbacks for the training process
        self.set_callbacks()

    @staticmethod
    def __init_gpu_config():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

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
            input_vocab_size=self.input_vocab_size,
            target_vocab_size=self.target_vocab_size,
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
                               self.monitor_params['for_start_value'])
        callback_saver.set_model(self.model)

        self.callbacks = [callback_tf_board, callback_saver]

    def __compile(self):
        self.model.compile(optimizer=self.compile_params['optimizer'],
                           loss=self.compile_params['loss'],
                           metrics=self.compile_params['metrics'])

    @staticmethod
    def __get_best_model_path(model_dir):
        """ get the best model within model_dir """
        file_list = os.listdir(model_dir)
        file_list.sort(key=lambda x: -x)
        return os.path.join(model_dir, file_list[0])

    def load_model(self, model_dir='', x=None, y=None):
        # get model path
        model_dir = model_dir if model_dir else self.__model_dir
        model_path = self.__get_best_model_path(model_dir)

        # empty fit, to prevent error from occurring when loading model
        self.model.fit(x, y, epochs=0) if not isinstance(x, type(None)) else None

        # load model weights
        self.model.load_weights(model_path)
        print('Successfully loading weights from %s ' % model_path)

    def train(self, train_x, train_y, val_x, val_y):
        # compile model
        self.__compile()

        # if we want to load a trained model
        if self.checkpoint_params['load_model']:
            model_dir = utils.create_dir_in_root(*(['runtime', 'models'] + self.checkpoint_params['load_model']))
            self.load_model(model_dir)

        # fit model
        self.model.fit(train_x, train_y,
                       epochs=self.train_params['epoch'],
                       batch_size=self.train_params['batch_size'],
                       validation_data=(val_x, val_y),
                       callbacks=self.callbacks,
                       verbose=2)

        # load the best model so that it could be tested
        self.load_model()

        self.finish_train = True

    def train_in_eager(self, train_x, train_y, val_x, val_y):
        pass

    def loss(self):
        pass

    # def evaluate(self, sentence):
    #     pass
    #
    # def predict(self, x):
    #     """ predict x in batch """
    #     # record the predictions
    #     pred_list = []
    #
    #     # temporary variables
    #     batch_size = self.train_params['batch_size']
    #     steps = int(math.ceil(len(x) * 1.0 / batch_size))
    #
    #     # traverse all data
    #     for step in range(steps):
    #         if isinstance(x, list) or isinstance(x, tuple):
    #             tmp_x = [v[step * batch_size: (step + 1) * batch_size] for v in x]
    #         else:
    #             tmp_x = x[step * batch_size: (step + 1) * batch_size]
    #         pred_list.append(self.model.predict(tmp_x))
    #
    #     return np.vstack(pred_list)
    #
    # def test(self, x, y, dataset=''):
    #     if not self.evaluate_dict:
    #         return {}
    #
    #     # predict
    #     pred = self.predict(x)
    #
    #     # evaluate
    #     result_dict = {}
    #     for key, func in self.evaluate_dict.items():
    #         result_dict[key] = func(y, pred)
    #
    #     # show results
    #     print('\n-----------------------------------------')
    #     for key, value in result_dict.items():
    #         print('{} {}: {}'.format(dataset, key, value))
    #     return result_dict
