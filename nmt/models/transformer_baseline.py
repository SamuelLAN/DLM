from nltk.translate.bleu_score import corpus_bleu
import tensorflow as tf
from nmt.models.base_model import BaseModel
from nmt.preprocess.inputs import noise_pl, tfds_share_pl, zh_en
from lib.tf_metrics.pretrain import tf_accuracy, tf_perplexity
from lib.preprocess import utils
from lib.tf_models.transformer_after_pretrain import Transformer

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_nmt_baseline'

    preprocess_pipeline = zh_en.seg_zh_by_jieba_pipeline + noise_pl.remove_noise + \
                          tfds_share_pl.train_tokenizer + tfds_share_pl.encode_pipeline
    # for test
    encode_pipeline = zh_en.seg_zh_by_jieba_pipeline + noise_pl.remove_noise + tfds_share_pl.encode_pipeline
    encode_pipeline_for_src = zh_en.seg_zh_by_jieba_pipeline + noise_pl.remove_noise_for_src + \
                              tfds_share_pl.encode_pipeline_for_src
    encode_pipeline_for_tar = noise_pl.remove_noise_for_src + tfds_share_pl.encode_pipeline_for_src
    decode_pipeline_for_src = tfds_share_pl.decode_pipeline + zh_en.remove_space_pipeline
    decode_pipeline_for_tar = tfds_share_pl.decode_pipeline

    data_params = {
        **BaseModel.data_params,
        'vocab_size': 80000,  # approximate
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'sample_rate': 1.0,  # sample "sample_rate" percentage of data into dataset; range from 0 ~ 1
        'input_incr': 4,
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
        'share_final': False,
        'use_beam_search': False,
        'top_k': 5,
        'get_random': False,
        'lan_vocab_size': 2,
    }

    train_params = {
        **BaseModel.train_params,
        'learning_rate': 1e-4,
        # 'learning_rate': 1e-3,
        # 'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 16,
        'epoch': 800,
        'early_stop': 15,
    }

    compile_params = {
        **BaseModel.compile_params,
        # 'optimizer': keras.optimizers.Adam(learning_rate=train_params['learning_rate'], beta_1=0.9, beta_2=0.98,
        #                                    epsilon=1e-9),
        'optimizer': tfv1.train.AdamOptimizer(learning_rate=train_params['learning_rate']),
        'label_smooth': True,
        'metrics': [tf_accuracy, tf_perplexity],
    }

    monitor_params = {
        **BaseModel.monitor_params,
        'name': 'val_tf_accuracy',
        'mode': 'max',  # for the "name" monitor, the "min" is best;
        'for_start': 'tf_accuracy',
        'for_start_value': 0.05,
        'for_start_mode': 'max',
    }

    checkpoint_params = {
        'load_model': [],  # [name, time] # test
        # 'load_model': ['transformer_CDLM_translate_wmt_news', '2020_05_13_04_33_50'],  # [name, time] # test
        # 'load_model': ['transformer_for_MLM_zh_en', '2020_04_23_15_16_14'],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['name']
    }

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
            lan_vocab_size=self.model_params['lan_vocab_size'],
        )

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
