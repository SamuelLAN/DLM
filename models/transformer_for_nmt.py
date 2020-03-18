from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import tensorflow as tf
from lib.tf_learning_rate.warmup_then_down import CustomSchedule
from models.base_model import BaseModel
from preprocess import zh_en
from lib.preprocess import utils

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_for_nmt'

    preprocess_pipeline = zh_en.seg_zh_by_jieba_pipeline + zh_en.train_subword_tokenizer_pipeline + \
                          zh_en.encode_with_tfds_tokenizer_pipeline
    # for test
    encode_pipeline = zh_en.encode_with_tfds_tokenizer_pipeline
    encode_pipeline_for_src = zh_en.seg_zh_by_jieba_pipeline + zh_en.encode_with_tfds_tokenizer_pipeline_for_src
    encode_pipeline_for_tar = zh_en.encode_with_tfds_tokenizer_pipeline_for_src
    decode_pipeline_for_src = zh_en.decode_with_tfds_tokenizer_pipeline + zh_en.remove_zh_space_pipeline
    decode_pipeline_for_tar = zh_en.decode_with_tfds_tokenizer_pipeline

    data_params = {
        'src_vocab_size': 2 ** 13,  # approximate
        'tar_vocab_size': 2 ** 13,  # approximate
        'max_src_seq_len': 50,
        'max_tar_seq_len': 60,
        'incr': 3,
    }

    model_params = {
        'emb_dim': 128,
        'dim_model': 128,
        'ff_units': 128,
        'num_layers': 6,
        'num_heads': 8,
        'max_pe_input': 50,
        'max_pe_target': 60,
        'drop_rate': 0.1,
    }

    train_params = {
        **BaseModel.train_params,
        'learning_rate': 3e-3,
        # 'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 64,
        'epoch': 300,
        'early_stop': 30,
    }

    compile_params = {
        **BaseModel.compile_params,
        'optimizer': tfv1.train.AdamOptimizer(learning_rate=train_params['learning_rate']),
        'metrics': [],
    }

    monitor_params = {
        **BaseModel.monitor_params,
        'name': 'val_loss',
        'mode': 'min',  # for the "name" monitor, the "min" is best;
    }

    checkpoint_params = {
        'load_model': [],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['name']
    }

    evaluate_dict = {

    }

    @property
    def tar_start_token_idx(self):
        return self.target_vocab_size

    @property
    def tar_end_token_idx(self):
        return self.target_vocab_size + 1

    @property
    def tar_max_seq_len(self):
        return self.data_params['max_tar_seq_len']

    def train_in_eager(self, train_x, train_y, val_x, val_y):
        pass

    def translate_sentences(self, list_of_src_sentences, src_tokenizer, tar_tokenizer):
        """ translate list of sentences and decode the results """
        encoded_data = utils.pipeline(self.encode_pipeline_for_src, list_of_src_sentences, None, {
            'src_tokenizer': src_tokenizer,
            'max_src_seq_len': self.data_params['max_src_seq_len'],
        })

        pred_encoded = self.evaluate_encoded(encoded_data)
        return self.decode_data(pred_encoded, tar_tokenizer)

    def decode_data(self, encoded_data, tokenizer):
        """ decode the list of list token idx to sentences """
        return utils.pipeline(self.decode_pipeline_for_tar, encoded_data, None, {'tokenizer': tokenizer})

    def calculate_bleu_for_encoded(self, src_encode_data, tar_tokenizer, tar_decode_data, dataset=''):
        """ evaluate the BLEU according to the encoded src language data (list_of_list_token_idx)
                                and the target reference (list of sentences) """
        pred_encoded_data = self.evaluate_encoded(src_encode_data)
        pred_decoded_data = self.decode_data(pred_encoded_data, tar_tokenizer)

        bleu = corpus_bleu(tar_decode_data, pred_decoded_data)
        print('{} bleu: {}'.format(dataset, bleu))
        return bleu
