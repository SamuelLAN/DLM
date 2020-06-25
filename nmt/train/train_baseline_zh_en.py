import os
import sys

cur_dir = os.path.abspath(os.path.split(__file__)[0])
sub_sub_root_dir = os.path.split(cur_dir)[0]
sub_root_dir = os.path.split(sub_sub_root_dir)[0]
root_dir = os.path.split(sub_root_dir)[0]

sys.path.append(sub_root_dir)
sys.path.append(root_dir)

from nmt.models.transformer_baseline_zh_en import Model
from nmt.load.zh_en_news import Loader
from nmt.train.train_base import Train as TrainBase

Model.name = 'transformer_nmt_baseline_zh_en'
Model.checkpoint_params['load_model'] = [Model.name, '2020_06_25_03_18_27']


class Train(TrainBase):
    TRAIN_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
    M = Model
    Loader = Loader


o_train = Train(use_cache=True)
# o_train.train()
o_train.test(load_model=True)


# Save model to D:\Github\DLM\runtime\models\transformer_nmt_baseline_zh_en\2020_06_25_03_18_27\transformer_nmt_baseline_zh_en.019-0.1308.hdf5
# 57558/57558 - 1251s - loss: -1.1356e+00 - tf_accuracy: 0.1594 - tf_perplexity: 8.2136 - val_loss: -8.7299e-01 - val_tf_accuracy: 0.1308 - val_tf_perplexity: 9.0261
# Epoch 20/800
# 57558/57558 - 1254s - loss: -1.1575e+00 - tf_accuracy: 0.1620 - tf_perplexity: 7.7194 - val_loss: -8.7407e-01 - val_tf_accuracy: 0.1307 - val_tf_perplexity: 6.4511
# Epoch 21/800
# Save model to D:\Github\DLM\runtime\models\transformer_nmt_baseline_zh_en\2020_06_25_03_18_27\transformer_nmt_baseline_zh_en.021-0.1309.hdf5
# 57558/57558 - 1265s - loss: -1.1797e+00 - tf_accuracy: 0.1649 - tf_perplexity: 7.2347 - val_loss: -8.7435e-01 - val_tf_accuracy: 0.1309 - val_tf_perplexity: 6.9717
# Epoch 22/800
