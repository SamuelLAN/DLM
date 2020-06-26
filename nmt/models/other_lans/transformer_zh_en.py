from nltk.translate.bleu_score import corpus_bleu
import tensorflow as tf
from nmt.models.base_model import BaseModel
from nmt.preprocess.inputs import noise_pl, tfds_share_pl, zh_en
from lib.preprocess import utils
from lib.tf_metrics.pretrain import tf_accuracy

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_for_nmt_share_emb_zh_word_level_wmt_news'

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
        'vocab_size': 85000,  # approximate
        # 'src_vocab_size': 16000,  # approximate
        # 'tar_vocab_size': 16000,  # approximate
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'sample_rate': 1.0,  # sample "sample_rate" percentage of data into dataset; range from 0 ~ 1
        # 'sample_rate': 0.066,  # sample "sample_rate" percentage of data into dataset; range from 0 ~ 1
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
    }

    train_params = {
        **BaseModel.train_params,
        'learning_rate': 1e-4,
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
        'metrics': [tf_accuracy],
    }

    monitor_params = {
        'monitor': 'val_tf_accuracy',
        'mode': 'max',  # for the "name" monitor, the "min" is best;
        'early_stop': train_params['early_stop'],
        'start_train_monitor': 'tf_accuracy',
        'start_train_monitor_value': 0.05,
        'start_train_monitor_mode': 'max',
    }

    checkpoint_params = {
        'load_model': ["baseline", "wmt-news"],  # [name, time]
        # 'load_model': [name, '2020_04_26_20_26_51'],  # [name, time] # BLEU 21, for news-commentary
        # 'load_model': [name, '2020_04_25_12_59_02'],  # [name, time] # BLEU 46, for wmt-news
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['monitor']
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
        return self.calculate_bleu_for_pred(pred_encoded_data, tar_encode_data, dataset)

    @staticmethod
    def calculate_bleu_for_pred(pred_encoded_data, tar_encode_data, dataset=''):
        """ evaluate the BLEU according to the encoded src language data (list_of_list_token_idx)
                                and the target reference (list of sentences) """
        print('\nstart translating {} ...'.format(dataset))
        tar_encode_data = utils.remove_some_token_idx(tar_encode_data, [0])

        pred_encoded_data = utils.convert_list_of_list_token_idx_2_string(pred_encoded_data)
        tar_encode_data = utils.convert_list_of_list_token_idx_2_string(tar_encode_data)
        tar_encode_data = list(map(lambda x: [x], tar_encode_data))

        print('calculating bleu ...')
        bleu = corpus_bleu(tar_encode_data, pred_encoded_data)

        print('{} bleu: {}'.format(dataset, bleu))
        return bleu

    def calculate_precisions_for_decoded(self, pred_encoded, tar_encoded_data, tar_tokenizer,
                                         n_gram=1, dataset='test', is_zh=False):

        tar_decoded_data = self.decode_tar_data(tar_encoded_data, tar_tokenizer)
        predictions = self.decode_tar_data(pred_encoded, tar_tokenizer)

        # convert sentences to list of words
        list_of_list_tar_token = list(map(
            lambda x: list(map(
                lambda a: a.strip(),
                x.strip('.').strip('!').strip('?').strip(';').strip(',').split(' ')
            )), tar_decoded_data))

        list_of_list_pred_token = list(map(
            lambda x: list(map(
                lambda a: a.strip(),
                x.strip('.').strip('!').strip('?').strip(';').strip(',').split(' ')
            )), predictions))

        test_precision_all = self.calculate_precision_for_encoded(
            list_of_list_pred_token, list_of_list_tar_token, n_gram,
            info_key='*', dataset=dataset, is_zh=is_zh)
        test_precision_translate = self.calculate_precision_for_encoded(
            list_of_list_pred_token, list_of_list_tar_token, n_gram,
            info_key='translation', dataset=dataset, is_zh=is_zh)
        test_precision_pos = self.calculate_precision_for_encoded(
            list_of_list_pred_token, list_of_list_tar_token, n_gram,
            info_key='pos', dataset=dataset, is_zh=is_zh)
        test_precision_src_syn = self.calculate_precision_for_encoded(
            list_of_list_pred_token, list_of_list_tar_token, n_gram,
            info_key='src_synonyms', dataset=dataset, is_zh=is_zh)
        test_precision_src_def = self.calculate_precision_for_encoded(
            list_of_list_pred_token, list_of_list_tar_token, n_gram,
            info_key='src_meanings', dataset=dataset, is_zh=is_zh)
        test_precision_tar_syn = self.calculate_precision_for_encoded(
            list_of_list_pred_token, list_of_list_tar_token, n_gram,
            info_key='tar_synonyms', dataset=dataset, is_zh=is_zh)
        test_precision_tar_def = self.calculate_precision_for_encoded(
            list_of_list_pred_token, list_of_list_tar_token, n_gram,
            info_key='tar_meanings', dataset=dataset, is_zh=is_zh)

        return {
            f'{dataset}_precision_all': test_precision_all,
            f'{dataset}_precision_translate': test_precision_translate,
            f'{dataset}_precision_pos': test_precision_pos,
            f'{dataset}_precision_src_syn': test_precision_src_syn,
            f'{dataset}_precision_src_def': test_precision_src_def,
            f'{dataset}_precision_tar_syn': test_precision_tar_syn,
            f'{dataset}_precision_tar_def': test_precision_tar_def,
        }

    @staticmethod
    def calculate_precision_for_encoded(list_of_list_pred_token, list_of_list_tar_token,
                                        n_gram=1, info_key='*', dataset='test', is_zh=False):
        """ evaluate the precision according to the encoded src language data """
        from lib.preprocess.utils import stem
        from pretrain.preprocess.dictionary.map_dict import zh_word, en_word, zh_phrase, en_phrase, n_grams

        # get map function
        if n_gram == 1:
            map_token = zh_word if is_zh else en_word
        else:
            map_token = zh_phrase if is_zh else en_phrase

        if n_gram > 1:
            list_of_list_tar_token = n_grams(list_of_list_tar_token, n_gram)
            list_of_list_pred_token = n_grams(list_of_list_pred_token, n_gram)

        # use the reference to map the dictionary
        list_of_list_info_for_ref = list(map(
            lambda l: list(map(
                lambda x: map_token(x, info_key), l
            )), list_of_list_tar_token
        ))

        # get stem list
        if n_gram == 1:
            list_of_list_tar_stem = list(map(lambda x: list(map(stem, x)), list_of_list_tar_token))
            list_of_list_pred_stem = list(map(lambda x: list(map(stem, x)), list_of_list_pred_token))

        else:
            list_of_list_tar_stem = list(map(
                lambda l: list(map(lambda g: ' '.join(list(map(stem, g))), l)),
                list_of_list_tar_token))
            list_of_list_pred_stem = list(map(
                lambda l: list(map(lambda g: ' '.join(list(map(stem, g))), l)),
                list_of_list_pred_token))

        # convert n gram list to string
        if n_gram > 1:
            list_of_list_tar_token = list(map(lambda l: list(map(lambda g: ' '.join(g), l)), list_of_list_tar_token))
            list_of_list_pred_token = list(map(lambda l: list(map(lambda g: ' '.join(g), l)), list_of_list_pred_token))

        map_ref_list = []
        map_pred_list = []

        # traverse the mapping results
        for sent_idx, list_of_info in enumerate(list_of_list_info_for_ref):

            # get correspond word list or stem list
            tar_word_list = list_of_list_tar_token[sent_idx]
            tar_stem_list = list_of_list_tar_stem[sent_idx]
            pred_word_list = list_of_list_pred_token[sent_idx]
            pred_stem_list = list_of_list_pred_stem[sent_idx]

            for gram_idx, info in enumerate(list_of_info):
                if not info:
                    continue

                tar_gram = tar_word_list[gram_idx]
                map_ref_list.append(tar_gram)

                if tar_gram in pred_word_list or tar_gram in pred_stem_list:
                    map_pred_list.append(tar_gram)
                    continue

                tar_stem = tar_stem_list[gram_idx]
                if tar_stem in pred_word_list or tar_stem in pred_stem_list:
                    map_pred_list.append(tar_stem)

        # get precision
        if len(map_ref_list) == 0:
            precision = 0
        else:
            precision = float(len(map_pred_list)) / len(map_ref_list)
        print(f'{dataset} precision in dictionary ({info_key}): {precision}')
        return precision

    def evaluate(self, list_of_list_src_token_idx, show_attention_weight=False):
        if self.model_params['use_beam_search']:
            return self.evaluate_encoded_beam_search(list_of_list_src_token_idx)
        return self.evaluate_encoded(list_of_list_src_token_idx, show_attention_weight)

    def get_attention_map(self, list_of_src_sentences, src_tokenizer, tar_tokenizer):
        """ translate list of sentences and decode the results """
        encoded_data = utils.pipeline(self.encode_pipeline_for_src, list_of_src_sentences, None, {
            'tokenizer': src_tokenizer,
            'vocab_size': src_tokenizer.vocab_size,
            'max_src_seq_len': self.data_params['max_src_seq_len'],
        })

        pred_encoded, attentions = self.evaluate(encoded_data, True)

        pred_decoded = self.decode_tar_data(pred_encoded, tar_tokenizer, False)
        src_decoded = self.decode_src_data(encoded_data, src_tokenizer, False)

        pred_decoded = pred_decoded[0]
        src_decoded = src_decoded[0]
        attentions = attentions[0]

        print('start ploting ...')

        for _layer, attention in attentions.items():
            if _layer != 'decoder_layer6_block2':
                continue
            # if _layer[-1] != '2':
            #     continue
            print(f'plotting {_layer} ... ')
            self.plot_attention_weights(attention, src_decoded, pred_decoded, _layer)

        print('finish plotting ')

        exit()
