from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import tensorflow as tf
from lib.tf_learning_rate.warmup_then_down import CustomSchedule
from models.base_model import BaseModel
from preprocess import zh_en, tfds_pl, tfds_share_pl
from lib.preprocess import utils

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_for_nmt_share_emb_zh_word_level'

    preprocess_pipeline = zh_en.seg_zh_by_jieba_pipeline + tfds_pl.remove_noise_pipeline + \
                          tfds_share_pl.train_tokenizer + tfds_share_pl.encode_pipeline
    # for test
    encode_pipeline = zh_en.seg_zh_by_jieba_pipeline + tfds_pl.remove_noise_pipeline + tfds_share_pl.encode_pipeline
    encode_pipeline_for_src = zh_en.seg_zh_by_jieba_pipeline + tfds_pl.remove_noise_pipeline + \
                              tfds_share_pl.encode_pipeline_for_src
    encode_pipeline_for_tar = tfds_share_pl.encode_pipeline_for_src
    decode_pipeline_for_src = tfds_share_pl.decode_pipeline + zh_en.remove_space_pipeline
    decode_pipeline_for_tar = tfds_share_pl.decode_pipeline

    data_params = {
        'vocab_size': 35000,  # approximate
        # 'src_vocab_size': 16000,  # approximate
        # 'tar_vocab_size': 16000,  # approximate
        'max_src_seq_len': 80,
        'max_tar_seq_len': 80,
        'sample_rate': 0.05,  # sample "sample_rate" percentage of data into dataset; range from 0 ~ 1
        'incr': 3,
    }

    model_params = {
        'emb_dim': 128,
        'dim_model': 128,
        'ff_units': 128,
        'num_layers': 6,
        'num_heads': 8,
        'max_pe_input': data_params['max_src_seq_len'],
        'max_pe_target': data_params['max_tar_seq_len'],
        'drop_rate': 0.1,
        'share_emb': True,
        'use_beam_search': False,
        'top_k': 5,
        'get_random': False,
    }

    train_params = {
        **BaseModel.train_params,
        # 'learning_rate': 8e-5,
        'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 64,
        'epoch': 500,
        'early_stop': 50,
    }

    compile_params = {
        **BaseModel.compile_params,
        'optimizer': keras.optimizers.Adam(learning_rate=train_params['learning_rate'], beta_1=0.9, beta_2=0.98,
                                           epsilon=1e-9),
        # 'optimizer': tfv1.train.AdamOptimizer(learning_rate=train_params['learning_rate']),
        'metrics': [],
    }

    monitor_params = {
        **BaseModel.monitor_params,
        'name': 'val_loss',
        'mode': 'min',  # for the "name" monitor, the "min" is best;
        'for_start': 'loss',
        'for_start_value': 1.5,
        'for_start_mode': 'min',
    }

    checkpoint_params = {
        'load_model': [],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['name']
    }

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
        tar_encode_data = utils.remove_some_token_idx(tar_encode_data, [0])

        pred_encoded_data = utils.convert_list_of_list_token_idx_2_string(pred_encoded_data)
        tar_encode_data = utils.convert_list_of_list_token_idx_2_string(tar_encode_data)
        tar_encode_data = list(map(lambda x: [x], tar_encode_data))

        print('calculating bleu ...')
        bleu = corpus_bleu(tar_encode_data, pred_encoded_data)

        print('{} bleu: {}'.format(dataset, bleu))
        return bleu

    def evaluate(self, list_of_list_src_token_idx):
        if self.model_params['use_beam_search']:
            return self.evaluate_encoded_beam_search(list_of_list_src_token_idx)
        return self.evaluate_encoded(list_of_list_src_token_idx)
