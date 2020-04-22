from lib.tf_learning_rate.warmup_then_down import CustomSchedule
from lib.tf_models.transformer import Transformer
from nmt.models.base_model import BaseModel
import tensorflow as tf

keras = tf.keras
tfv1 = tf.compat.v1


class Model(BaseModel):
    name = 'transformer_for_cross_lingual_pretrain'

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
        # 'learning_rate': 3e-3,
        'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 64,
        'epoch': 300,
        'early_stop': 30,
        'loss': keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none'),
    }

    compile_params = {
        'optimizer': tfv1.train.AdamOptimizer(learning_rate=train_params['learning_rate']),
        'loss': keras.losses.categorical_crossentropy,
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

    def build(self):
        self.model = Transformer(
            num_layers=self.model_params['num_layers'],
            d_model=self.model_params['dim_model'],
            num_heads=self.model_params['num_heads'],
            d_ff=self.model_params['ff_units'],
            input_vocab_size=self.input_vocab_size,
            target_vocab_size=self.target_vocab_size,
            max_pe_input=self.model_params['max_pe_input'],
            max_pe_target=self.model_params['max_pe_target'],
            drop_rate=self.model_params['drop_rate'],
        )

    def train_in_eager(self, train_x, train_y, val_x, val_y):
        pass

    def loss(self):
        pass
