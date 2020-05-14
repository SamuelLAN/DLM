from nltk.translate.bleu_score import corpus_bleu
import tensorflow as tf
from nmt.preprocess.inputs import noise_pl, tfds_share_pl, zh_en
from lib.tf_metrics.pretrain import tf_accuracy, tf_perplexity
from nmt.models.transformer_baseline import Model as BaseModel
from pretrain.preprocess.config import Ids, LanIds

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_nmt_CDLM_synonym_pretrained'

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
        'vocab_size': 10000,  # approximate
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'sample_rate': 1.0,  # sample "sample_rate" percentage of data into dataset; range from 0 ~ 1
        'input_incr': Ids.end_cdlm_synonym_2 + 1,
        'class_incr': Ids.end_cdlm_synonym_2 + 1,
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
        'epoch': 2,
        'early_stop': 10,
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
        'load_model': ['transformer_CDLM_synonym_wmt_news', '2020_05_13_21_41_13'],  # [name, time] # test
        # 'load_model': ['transformer_CDLM_translate_wmt_news', '2020_05_13_04_33_50'],  # [name, time] # test
        # 'load_model': ['transformer_for_MLM_zh_en', '2020_04_23_15_16_14'],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['name']
    }
