import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_root_dir)
sys.path.append(root_dir)

from pretrain.train.train_CDLM_base_async_preprocess import Train as TrainBase
from pretrain.models.transformer_cdlm_translate_zh_en_ro import Model
from pretrain.load.async_loader import Loader as LoaderVal
from pretrain.load.async_preprocess_loader import Loader as LoaderTrain

Model.name = 'transformer_CDLM_translate_zh_en_ro_wmt_news_60k_async_preprocess'
Model.checkpoint_params['load_model'] = ['transformer_CDLM_translate_zh_en_ro_wmt_news_um_async_preprocess', '2020_06_21_16_47_40']


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    LoaderTrain = LoaderTrain
    LoaderVal = LoaderVal

    train_preprocess_dirs = ['zh_en_wmt_news_news_commentary_60k_train', 'ro_en_60k_train']
    val_preprocess_dirs = ['zh_en_wmt_news_news_commentary_60k_cdlm_translate_test', 'ro_en_cdlm_translate_ro_60k_test']
    tokenizer_dir = 'zh_en_ro_news_commentary_wmt_news_um_corpus_dict_90000'


o_train = Train()
o_train.train()
o_train.test(load_model=False)
o_train.end()

# Epoch 3/200
# 2020-06-23 18:30:59.176981: W tensorflow/core/kernels/data/generator_dataset_op.cc:103] Error occurred when finalizing GeneratorDataset iterator: Cancelled: Operation was cancelled
# Save model to D:\Github\DLM\runtime\models\transformer_CDLM_translate_zh_en_ro_wmt_news_um_async_preprocess\2020_06_21_16_47_40\transformer_CDLM_translate_zh_en_ro_wmt_news_um_async_preprocess.003-0.1196.hdf5
# 185792/185792 - 70792s - loss: -9.3006e-01 - tf_accuracy: 0.1099 - tf_perplexity: 2.9565 - val_loss: -1.0084e+00 - val_tf_accuracy: 0.1196 - val_tf_perplexity: 2.9570
# Epoch 4/200
# 2020-06-24 13:29:42.836178: W tensorflow/core/kernels/data/generator_dataset_op.cc:103] Error occurred when finalizing GeneratorDataset iterator: Cancelled: Operation was cancelled
# Save model to D:\Github\DLM\runtime\models\transformer_CDLM_translate_zh_en_ro_wmt_news_um_async_preprocess\2020_06_21_16_47_40\transformer_CDLM_translate_zh_en_ro_wmt_news_um_async_preprocess.004-0.1204.hdf5
# 185792/185792 - 68329s - loss: -9.7101e-01 - tf_accuracy: 0.1149 - tf_perplexity: 3.2285 - val_loss: -1.0156e+00 - val_tf_accuracy: 0.1204 - val_tf_perplexity: 3.4276
