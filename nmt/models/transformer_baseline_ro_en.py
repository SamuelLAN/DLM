import tensorflow as tf
from nmt.models.transformer_baseline import Model as BaseModel
from nmt.preprocess.inputs import noise_pl, tfds_share_pl

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_nmt_baseline_ro_en'

    preprocess_pipeline = noise_pl.remove_noise + tfds_share_pl.train_tokenizer + tfds_share_pl.encode_pipeline
    # for test
    tokenizer_pl = noise_pl.remove_noise + tfds_share_pl.train_tokenizer
    encode_pipeline = noise_pl.remove_noise + tfds_share_pl.encode_pipeline
    encode_pipeline_for_src = noise_pl.remove_noise_for_src + tfds_share_pl.encode_pipeline_for_src
    encode_pipeline_for_tar = noise_pl.remove_noise_for_src + tfds_share_pl.encode_pipeline_for_src
    decode_pipeline_for_src = tfds_share_pl.decode_pipeline
    decode_pipeline_for_tar = tfds_share_pl.decode_pipeline

    data_params = {
        **BaseModel.data_params,
        'vocab_size': 20000,  # approximate
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'sample_rate': 0.0125,  # sample "sample_rate" percentage of data into dataset; range from 0 ~ 1
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

    monitor_params = {
        **BaseModel.monitor_params,
        'early_stop': train_params['early_stop'],
    }

    checkpoint_params = {
        'load_model': [],  # [name, time] # test
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['monitor']
    }
