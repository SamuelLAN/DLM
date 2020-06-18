import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_root_dir)
sys.path.append(root_dir)

from pretrain.train.train_CDLM_base_async_preprocess import Train as TrainBase
from pretrain.models.transformer_cdlm_translate import Model
from pretrain.load.async_loader import Loader as LoaderVal
from pretrain.load.async_preprocess_loader import Loader as LoaderTrain

Model.name = 'transformer_CDLM_translate_news_commentary_async_preprocess'
Model.checkpoint_params['load_model'] = ['transformer_CDLM_translate_news_commentary_async_preprocess', '2020_06_14_13_28_17']


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    LoaderTrain = LoaderTrain
    LoaderVal = LoaderVal

    train_preprocess_dirs = ['news_commentary_train']
    val_preprocess_dirs = ['news_commentary_test_v2']
    tokenizer_dir = 'only_news_commentary_80000'


o_train = Train()
# o_train.train()
o_train.test(load_model=True)
o_train.end()

# vocab_size: 80289
# ----------------------------------------
# dirs: ['news_commentary_train']
# size: 300544
# x.shape: (12, 60)
# y.shape: (12, 24)
# lan_x.shape: (12, 60)
# lan_y.shape: (12, 24)
# pos_y.shape: (12, 24)
# pos_emb_y.shape: (12, 24, 128)
#
# ----------------------------------------
# dirs: ('news_commentary_test_v2',)
# size: 6144
# x.shape: (12, 60)
# y.shape: (12, 24)
# lan_x.shape: (12, 60)
# lan_y.shape: (12, 24)
# pos_y.shape: (12, 24)
# pos_emb_y.shape: (12, 24, 128)
#
# transformer_CDLM_translate_news_commentary_async_preprocess  2020_06_14_13_28_17
# Epoch 66/200
# 2020-06-18 03:19:59.336546: W tensorflow/core/kernels/data/generator_dataset_op.cc:103] Error occurred when finalizing GeneratorDataset iterator: Cancelled: Operation was cancelled
# Save model to D:\Github\DLM\runtime\models\transformer_CDLM_translate_news_commentary_async_preprocess\2020_06_14_13_28_17\transformer_CDLM_translate_news_commentary_async_preprocess.066-0.2189.hdf5
# 18784/18784 - 4483s - loss: -1.7611e+00 - tf_accuracy: 0.2574 - tf_perplexity: 2.0008 - val_loss: -1.4158e+00 - val_tf_accuracy: 0.2189 - val_tf_perplexity: 1.8817
