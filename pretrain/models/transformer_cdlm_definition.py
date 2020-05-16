from lib.tf_learning_rate.warmup_then_down import CustomSchedule
from nmt.preprocess.inputs import noise_pl, tfds_share_pl, zh_en
from pretrain.preprocess.inputs import CDLM_definition
from pretrain.preprocess.inputs.sampling import sample_pl
from pretrain.preprocess.inputs.pl import CDLM_encode, sent_2_tokens
from pretrain.preprocess.inputs.decode import decode_pl as d_pl
from lib.tf_metrics.pretrain import tf_accuracy, tf_perplexity
from pretrain.preprocess.config import Ids
from pretrain.models.transformer_cdlm_translate import Model as BaseModel
import tensorflow as tf
import numpy as np

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_CDLM_definition'

    pretrain_params = {
        'keep_origin_rate': 0.2,
        'TLM_ratio': 0.7,
        'max_ratio': 0.3,
        'max_num': 4,
    }

    data_params = {
        **BaseModel.data_params,
        'vocab_size': 80000,  # approximate
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'max_src_ground_seq_len': 32,
        'max_tar_ground_seq_len': 32,
        'sample_ratio': 1.0,  # sample "sample_rate" percentage of data into dataset; > 0
        'over_sample_rate': 3.0,
        'input_incr': Ids.end_cdlm_def + 1,  # <start>, <end>, <pad>, <mask>
        'class_incr': Ids.end_cdlm_def + 1,  # <start>, <end>, <pad>, <mask>
    }

    preprocess_pl = zh_en.seg_zh_by_jieba_pipeline + noise_pl.remove_noise
    tokenizer_pl = preprocess_pl + tfds_share_pl.train_tokenizer
    encode_pl = preprocess_pl + sent_2_tokens + sample_pl(data_params['over_sample_rate']) + \
                CDLM_definition.combine_pl(**pretrain_params) + CDLM_encode
    decode_pl = d_pl('')

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
        'lan_vocab_size': 2,
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
        **BaseModel.monitor_params,
        'name': 'val_tf_accuracy',
        'mode': 'max',  # for the "name" monitor, the "min" is best;
    }

    checkpoint_params = {
        'load_model': [],  # [name, time]
        # 'load_model': ['transformer_for_MLM_zh_en', '2020_04_26_15_19_16'],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['name']
    }

    evaluate_dict = {

    }
