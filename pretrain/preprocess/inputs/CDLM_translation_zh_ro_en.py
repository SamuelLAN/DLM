import re
import copy
import random
import numpy as np
from functools import reduce
from pretrain.preprocess.dictionary import map_dict_ro_en as map_dict
from pretrain.preprocess.config import Ids, LanIds, SampleRatio
from pretrain.preprocess.inputs.TLM import TLM_concat
from pretrain.preprocess.inputs.CDLM_translation_ro_en import CDLM_translation as CDLM_translation_ro_en
from pretrain.preprocess.inputs.CDLM_translation_v2 import CDLM_translation as CDLM_translation_zh_en

random_state = 42

ratio_mode_0 = SampleRatio.translation['ratio_mode_0']
ratio_mode_1 = SampleRatio.translation['ratio_mode_1']
ratio_mode_2 = SampleRatio.translation['ratio_mode_2']

ratio_mode_0_1 = ratio_mode_0 + ratio_mode_1

__reg_zh = re.compile(r'[\u4e00-\u9fff]+', re.IGNORECASE)


def is_chinese(list_string):
    for string in list_string[:2]:
        if __reg_zh.search(string):
            return True
    return False


def CDLM_MLM_sample(list_of_zh_words, list_of_en_words, _tokenizer, keep_origin_rate=0.2,
                    max_ratio=0.2, max_num=4):
    zh_data = list(map(
        lambda x: CDLM_translation_zh_en(x, _tokenizer, True, keep_origin_rate)
        if is_chinese(x) else CDLM_translation_ro_en(x, _tokenizer, True, keep_origin_rate, max_ratio, max_num),
        list_of_zh_words
    ))
    en_data = list(map(
        lambda x: CDLM_translation_zh_en(x, _tokenizer, False, keep_origin_rate, max_ratio, max_num), list_of_en_words))

    data = zh_data + en_data
    data = list(filter(lambda x: x[0] and x[1] and x[2] and x[3] and x[4], data))

    random.seed(random_state)
    random.shuffle(data)

    _inputs, _outputs, _lan_inputs, _lan_outputs, _soft_pos_outputs = list(zip(*data))
    return _inputs, _outputs, _lan_inputs, _lan_outputs, _soft_pos_outputs


def MLM_pl(keep_origin_rate=0.2, max_ratio=0.2, max_num=4):
    return [
        {
            'name': 'CDLM_translation_MLM_sample',
            'func': CDLM_MLM_sample,
            'input_keys': ['input_1', 'input_2', 'tokenizer', keep_origin_rate, max_ratio, max_num],
            'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1', 'pos_for_gt_1'],
            'show_dict': {'input_1': 'input_1', 'ground_truth_1': 'ground_truth_1',
                          'lan_idx_for_input_1': 'lan_idx_for_input_1', 'lan_idx_for_gt_1': 'lan_idx_for_gt_1',
                          'pos_for_gt_1': 'pos_for_gt_1'},
        },
    ]


if __name__ == '__main__':
    from nmt.preprocess.corpus import wmt_news
    from lib.preprocess import utils
    from nmt.preprocess.inputs import noise_pl, tfds_share_pl, zh_en
    from pretrain.preprocess.inputs import pl
    from pretrain.preprocess.inputs.decode import decode_pl
    from pretrain.load.token_translation import Loader
    from nmt.load.ro_en import Loader
    from pretrain.preprocess.inputs.sampling import sample_pl

    token_loader = Loader(0.0, 0.1, 0.01)
    origin_ro_data, origin_en_data = token_loader.data()

    # origin_ro_data, origin_en_data = wmt_news.zh_en()
    params = {
        'vocab_size': 10000,
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'max_src_ground_seq_len': 12,
        'max_tar_ground_seq_len': 12,
    }

    # tokenizer_pl = noise_pl.remove_noise + tfds_share_pl.train_tokenizer
    # tokenizer = utils.pipeline(tokenizer_pl,
    #                            token_zh_data + list(origin_zh_data[:1000]), token_en_data + list(origin_en_data[:1000]), params)

    pipeline = noise_pl.remove_noise + tfds_share_pl.train_tokenizer
    # pipeline = zh_en.seg_zh_by_jieba_pipeline + noise_pl.remove_noise
    pipeline += pl.sent_2_tokens + MLM_pl(0.2) + pl.CDLM_encode + [
        {'output_keys': [
            'input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1', 'pos_for_gt_1', 'tokenizer']}
    ]

    print('\n------------------- Encoding -------------------------')
    x, y, lan_x, lan_y, soft_pos_y, tokenizer = utils.pipeline(
        preprocess_pipeline=pipeline,
        lan_data_1=origin_ro_data, lan_data_2=origin_en_data, params={**params,
                                                                      # 'tokenizer': tokenizer
                                                                      })

    print('\n----------------------------------------------')
    print(x.shape)
    print(y.shape)
    print(lan_x.shape)
    print(lan_y.shape)
    print(soft_pos_y.shape)

    print('\n------------------- Decoding -------------------------')
    x = utils.pipeline(decode_pl(''), x[:2], None, {'tokenizer': tokenizer})
    y = utils.pipeline(decode_pl(''), y[:2], None, {'tokenizer': tokenizer})
    print(x[0])
    print(soft_pos_y[0])
    print(x[1])
    print(soft_pos_y[1])
