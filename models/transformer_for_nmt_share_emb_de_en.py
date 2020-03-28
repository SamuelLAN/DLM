import tensorflow as tf
from lib.tf_learning_rate.warmup_then_down import CustomSchedule
from models.transformer_for_nmt_share_emb_zh_en import Model as BaseModel
from preprocess import tfds_share_pl

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_for_nmt_share_emb_de_en'

    preprocess_pipeline = tfds_share_pl.train_tokenizer + tfds_share_pl.encode_pipeline
    # for test
    encode_pipeline = tfds_share_pl.encode_pipeline
    encode_pipeline_for_src = tfds_share_pl.encode_pipeline_for_src
    encode_pipeline_for_tar = tfds_share_pl.encode_pipeline_for_src
    decode_pipeline_for_src = tfds_share_pl.decode_pipeline
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
        'batch_size': 16,
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
