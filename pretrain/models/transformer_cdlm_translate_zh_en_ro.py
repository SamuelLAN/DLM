from lib.tf_learning_rate.warmup_then_down import CustomSchedule
from nmt.models.base_model import BaseModel
from nmt.preprocess.inputs import noise_pl, tfds_share_pl
from pretrain.preprocess.inputs import CDLM_translation_zh_ro_en as CDLM_translation
from pretrain.preprocess.inputs.sampling import sample_pl
from pretrain.preprocess.inputs.pl import CDLM_encode, sent_2_tokens
from pretrain.preprocess.inputs.decode import decode_pl as d_pl
from lib.tf_models.transformer_lan_soft_pos import Transformer
from lib.tf_models.pos_embeddings import embedding as pos_embeddings
from lib.tf_metrics.pretrain import tf_accuracy, tf_perplexity
from pretrain.preprocess.config import Ids
import tensorflow as tf
import numpy as np

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_CDLM_translate_zh_en_ro'

    pretrain_params = {
        'keep_origin_rate': 0.2,
        # 'TLM_ratio': 0.7,
        'max_ratio': 0.3,
        'max_num': 4,
    }

    data_params = {
        **BaseModel.data_params,
        'vocab_size': 90000,  # approximate
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'max_src_ground_seq_len': 24,
        'max_tar_ground_seq_len': 24,
        'sample_ratio': 1.0,  # sample "sample_rate" percentage of data into dataset; > 0
        'over_sample_rate': 3.0,
        'input_incr': Ids.end_cdlm_t_2 + 1,  # <start>, <end>, <pad>, <mask>
        'class_incr': Ids.end_cdlm_t_2 + 1,  # <start>, <end>, <pad>, <mask>
    }

    preprocess_pl = noise_pl.remove_noise
    tokenizer_pl = preprocess_pl + tfds_share_pl.train_tokenizer
    # encode_pl = preprocess_pl + sent_2_tokens + sample_pl(data_params['over_sample_rate']) + CDLM_translation.combine_pl(**pretrain_params) + CDLM_encode
    encode_pl = preprocess_pl + sent_2_tokens + CDLM_translation.MLM_pl(**pretrain_params) + CDLM_encode
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
        'lan_vocab_size': 3,
    }

    train_params = {
        **BaseModel.train_params,
        'learning_rate': 1e-4,
        # 'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 16,
        'epoch': 200,
        'early_stop': 20,
    }

    compile_params = {
        **BaseModel.compile_params,
        'optimizer': tfv1.train.AdamOptimizer(learning_rate=train_params['learning_rate']),
        'label_smooth': True,
        'metrics': [tf_accuracy, tf_perplexity],
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
        'load_model': [],  # [name, time]
        # 'load_model': ['transformer_for_MLM_zh_en', '2020_04_26_15_19_16'],  # [name, time]
        'extend_name': '.{epoch:03d}-{%s:.4f}.hdf5' % monitor_params['monitor']
    }

    evaluate_dict = {

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

    @staticmethod
    def pos_emb(pos_matrix):
        _emb = pos_embeddings(Model.data_params['max_src_seq_len'] + Model.data_params['max_src_ground_seq_len'],
                              Model.model_params['dim_model'])
        return np.array([_emb[0, list(np.array(pos_vector))] for pos_vector in pos_matrix], dtype=np.float32)
