import tensorflow as tf
from nmt.models.other_lans.transformer_zh_en import Model as BaseModel
from nmt.preprocess.inputs import noise_pl, tfds_share_pl

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_for_nmt_share_emb_de_en_use_wmt'

    preprocess_pipeline = noise_pl.remove_noise + tfds_share_pl.train_tokenizer + tfds_share_pl.encode_pipeline
    # for test
    encode_pipeline = noise_pl.remove_noise + tfds_share_pl.encode_pipeline
    encode_pipeline_for_src = noise_pl.remove_noise_for_src + tfds_share_pl.encode_pipeline_for_src
    encode_pipeline_for_tar = noise_pl.remove_noise_for_src + tfds_share_pl.encode_pipeline_for_src
    decode_pipeline_for_src = tfds_share_pl.decode_pipeline
    decode_pipeline_for_tar = tfds_share_pl.decode_pipeline

    data_params = {
        **BaseModel.data_params,
        'vocab_size': 45000,  # approximate
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'sample_rate': 1.0,  # sample "sample_rate" percentage of data into dataset; range from 0 ~ 1
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
        'learning_rate': 1e-4,
        # 'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 16,
        'epoch': 180,
        'early_stop': 10,
    }

    compile_params = {
        **BaseModel.compile_params,
        # 'optimizer': keras.optimizers.Adam(learning_rate=train_params['learning_rate'], beta_1=0.9, beta_2=0.98,
        #                                    epsilon=1e-9),
        'optimizer': tfv1.train.AdamOptimizer(learning_rate=train_params['learning_rate']),
        'metrics': [],
    }

    monitor_params = {
        **BaseModel.monitor_params,
        'name': 'loss',
        'mode': 'min',  # for the "name" monitor, the "min" is best;
        'for_start': 'loss',
        'for_start_value': 1.5,
        'for_start_mode': 'min',
    }

    checkpoint_params = {
        'load_model': [],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['name']
    }
