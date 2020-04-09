from nltk.translate.bleu_score import corpus_bleu
from models.base_model import BaseModel
from preprocess.nmt_inputs import jr_en, tfds_pl
from lib.preprocess import utils
from lib.utils import create_dir_in_root


class Model(BaseModel):
    name = 'transformer_for_nmt_torch'

    preprocess_pipeline = jr_en.seg_jr_by_mecab_pipeline + tfds_pl.train_tokenizer_pipeline + \
                          tfds_pl.encode_pipeline
    # for test
    encode_pipeline = jr_en.seg_jr_by_mecab_pipeline + tfds_pl.encode_pipeline
    encode_pipeline_for_src = jr_en.seg_jr_by_mecab_pipeline + tfds_pl.encode_pipeline_for_src
    encode_pipeline_for_tar = tfds_pl.encode_pipeline_for_src
    decode_pipeline_for_src = tfds_pl.decode_pipeline + jr_en.remove_space_pipeline
    decode_pipeline_for_tar = tfds_pl.decode_pipeline

    data_params = {
        'src_vocab_size': 15000,  # approximate
        'tar_vocab_size': 2 ** 13,  # approximate
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'sample_rate': 1.0,  # sample "sample_rate" percentage of data into dataset; range from 0 ~ 1
        'incr': 3,
    }

    model_params = {
        'emb_dim': 128,
        'dim_model': 1024,
        'ff_units': 1024,
        'num_layers': 6,
        'num_heads': 8,
        'max_pe_input': data_params['max_src_seq_len'],
        'max_pe_target': data_params['max_tar_seq_len'],
        'drop_rate': 0.1,
        'use_beam_search': False,
        'top_k': 5,
        'get_random': False,
    }

    train_params = {
        **BaseModel.train_params,
        'learning_rate': 2e-5,
        # 'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 64,
        'epoch': 100,
        'early_stop': 20,
    }

    checkpoint_params = {
        'load_model': [],  # [name, time]
    }

    def __init__(self, input_vocab_size, target_vocab_size, name='', finish_train=False):
        self.name = name if name else self.name
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.__finish_train = finish_train

        # create directories for tensorboard files and model files
        self.__create_dir()

        # build models
        self.build()

    def build(self):
        """
        Build the model to self.model
            no returns
        """
        # TODO build the pytorch model here
        #   the "model" should provide the following functions:
        #       __call__(inputs, training=None)
        #           :params: inputs is ( encoder_input, decoder_input )
        #           :return: decoder_output
        #           (It would be used in the "self.calculate_loss_for_encoded" functions during the test process)
        #       evaluate_list_of_list_token_idx
        #           (for generating the translation, which means iterate the predictions to do translations.
        #               Details are in the /lib/tf_models/transformer.)
        #       beam_search_list_of_list_token_idx (optional)
        #
        #   for the params of the model, please use the above data_params, model_params and train_params
        #
        # initialize the model instance
        # self.model = pytorch Model
        # ...
        pass

    def load_model(self, model_dir='', x=None, y=None):
        """
        load the model to self.model
            no returns
        """
        # TODO
        #   given model_dir, you could load the best model to self.model in the model dir
        #   if model_dir is empty, then you could use self.model_dir to load the model
        # ...
        pass

    def loss(self, y_true, y_pred, from_logits=False, label_smoothing=0):
        """
        Calculate the loss
          (The test process will use this function)
        TODO
            you should provide this function no matter you use it in training or not;
            because the test process would call this function
        :return: loss (float)
        """
        pass

    def train(self, train_x, train_y, val_x=None, val_y=None):
        """
        TODO
            you need to override this train functions

        :params: train_x (tuple): a tuple with two elements, each one is a np.array
                                    (encoder_input, decoder_input)
                                    e.g., (train_src_encode, train_tar_encode[:, :-1])
        :params: train_y (np.array): decoder_output (ground_truth)
                                    train_tar_encode[:, 1:]
        :params: val_x (tuple): Use for validation, could be None.
                                If None, then no validation; if it is not None, it would be the same as train_X
        :params: val_y (np.array): Use for validation, could be None.
                                If None, then no validation; if it is not None, it would be the same as train_y

        no returns
        """

        # if we want to load a trained model
        if self.checkpoint_params['load_model']:
            model_dir = create_dir_in_root(*(['runtime', 'models'] + self.checkpoint_params['load_model']))
            self.load_model(model_dir)

        if not self.__finish_train:
            # TODO
            #   you should write down the train process here
            #   This process should also provide the following functions:
            #       calculate the loss
            #       update model weights
            #       early stop
            #       validation
            #       something like tensorboard (for saving files for tensorboard, you should save it to self.tb_dir)
            #       print out the results to console for each epoch
            #       save the best model to self.model_dir
            # ...
            #
            # for step in range(steps)
            # ...

            # load the best model so that it could be tested
            self.load_model()

            self.__finish_train = True

    def translate_sentences(self, list_of_src_sentences, src_tokenizer, tar_tokenizer):
        """ translate list of sentences and decode the results """
        encoded_data = utils.pipeline(self.encode_pipeline_for_src, list_of_src_sentences, None, {
            'src_tokenizer': src_tokenizer,
            'max_src_seq_len': self.data_params['max_src_seq_len'],
        })

        pred_encoded = self.evaluate(encoded_data)
        return self.decode_tar_data(pred_encoded, tar_tokenizer)

    def translate_list_token_idx(self, list_of_list_of_src_token_idx, tar_tokenizer):
        """ translate the src list token idx to target language sentences """
        pred_encoded = self.evaluate(list_of_list_of_src_token_idx)
        return self.decode_tar_data(pred_encoded, tar_tokenizer)

    def decode_src_data(self, encoded_data, tokenizer, to_sentence=True):
        """ decode the list of list token idx to sentences """
        end_index = None if to_sentence else -2
        return utils.pipeline(self.decode_pipeline_for_src[:end_index], encoded_data, None, {'tokenizer': tokenizer},
                              False)

    def decode_tar_data(self, encoded_data, tokenizer, to_sentence=True):
        """ decode the list of list token idx to sentences """
        end_index = None if to_sentence else -1
        return utils.pipeline(self.decode_pipeline_for_tar[:end_index], encoded_data, None, {'tokenizer': tokenizer},
                              False)

    def calculate_bleu_for_encoded(self, src_encode_data, tar_encode_data, dataset=''):
        """ evaluate the BLEU according to the encoded src language data (list_of_list_token_idx)
                                and the target reference (list of sentences) """
        print('\nstart translating {} ...'.format(dataset))
        pred_encoded_data = self.evaluate(src_encode_data)

        pred_encoded_data = utils.convert_list_of_list_token_idx_2_string(pred_encoded_data)
        tar_encode_data = utils.convert_list_of_list_token_idx_2_string(tar_encode_data)

        print('calculating bleu ...')
        bleu = corpus_bleu(tar_encode_data, pred_encoded_data)

        print('{} bleu: {}'.format(dataset, bleu))
        return bleu

    def evaluate(self, list_of_list_src_token_idx):
        if self.model_params['use_beam_search']:
            return self.evaluate_encoded_beam_search(list_of_list_src_token_idx)
        return self.evaluate_encoded(list_of_list_src_token_idx)
