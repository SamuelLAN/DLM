import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_root_dir)
sys.path.append(root_dir)

from pretrain.train.train_CDLM_base_async import Train as TrainBase
from pretrain.models.transformer_cdlm_translate import Model
from pretrain.load.async_loader import Loader

Model.name = 'transformer_CDLM_translate_wmt_16_19_20_sr_10_voc_8w'


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader

    train_preprocess_dirs = ['zh_en_wmt_19_20_cdlm_translate_6w_size_10_sr_80k_voc_train', 'ro_en_cdlm_translate_6w_size_10_sr_80k_voc_train']
    val_preprocess_dirs = ['zh_en_wmt_19_20_cdlm_translate_6w_size_10_sr_80k_voc_test', 'ro_en_cdlm_translate_6w_size_10_sr_80k_voc_test']
    tokenizer_dir = 'zh_en_ro_wmt_16_19_20_80000'


o_train = Train()
o_train.train()
o_train.test(load_model=False)
o_train.end()


# vocab_size: 80289
#
#
# ----------------------------------------
# dirs: ('cdlm_translate_train_10.0',)
# size: 3005440
# x.shape: (12, 60)
# y.shape: (12, 24)
# lan_x.shape: (12, 60)
# lan_y.shape: (12, 24)
# pos_y.shape: (12, 24)
# pos_emb_y.shape: (12, 24, 128)
#
# ----------------------------------------
# dirs: ('cdlm_translate_test',)
# size: 6144
# x.shape: (12, 60)
# y.shape: (12, 24)
# lan_x.shape: (12, 60)
# lan_y.shape: (12, 24)
# pos_y.shape: (12, 24)
# pos_emb_y.shape: (12, 24, 128)

# Epoch 8/50
# 2020-06-17 19:19:32.082414: W tensorflow/core/kernels/data/generator_dataset_op.cc:103] Error occurred when finalizing GeneratorDataset iterator: Cancelled: Operation was cancelled
# Save model to /home/a/github/DLM/runtime/models/transformer_CDLM_translate_news_commentary_10.0_async/2020_06_14_01_18_19/transformer_CDLM_translate_news_commentary_10.0_async.008-0.2533.hdf5
# 187840/187840 - 40397s - loss: -1.6839e+00 - tf_accuracy: 0.2465 - tf_perplexity: 2.6137 - val_loss: -1.7168e+00 - val_tf_accuracy: 0.2533 - val_tf_perplexity: 2.1004
