import random
from functools import reduce
from pretrain.preprocess.config import Ids, LanIds


def MLM_list_of_list_of_words(list_of_list_of_words, _tokenizer, lan_index,
                              min_num=2, max_num=6, max_ratio=0.2, keep_origin_rate=0.2):
    """ MLM for batch data """
    data = list(map(
        lambda x: MLM(x, _tokenizer, lan_index, min_num, max_num, max_ratio, keep_origin_rate),
        list_of_list_of_words
    ))
    list_masked_input, list_of_list_tar_token_idx, list_of_lan_idx_for_input, list_of_lan_idx_for_gt = list(zip(*data))
    return list_masked_input, list_of_list_tar_token_idx, list_of_lan_idx_for_input, list_of_lan_idx_for_gt


def MLM(list_of_words_for_a_sentence, _tokenizer, lan_index,
        min_num=1, max_num=6, max_ratio=0.2, keep_origin_rate=0.2):
    """
    Masked Language Modeling (word level)
    :params
        list_of_words_for_a_sentence (list): ['I', 'am', 'a', 'student']
        tokenizer (object): tfds tokenizer object
        lan_index (int): index for language embeddings, could be 0 or 1
        min_num (int):
        max_num (int):
        max_ratio (float):
        keep_origin_rate (float):
    :returns
        masked_input (list): list of encoded and masked token idx
        list_of_tar_token_idx (list):
        list_of_lan_idx (list):
    """

    # BPE for each word
    list_of_list_token_idx = list(map(lambda x: _tokenizer.encode(x), list_of_words_for_a_sentence))

    # get indices to mask
    len_words = len(list_of_list_token_idx)
    indices_to_mask = random.sample(
        range(len_words - 1),
        min(random.randint(min_num, max_num), max(round(len_words * max_ratio), 1))
    )
    indices_to_mask.sort()

    # get ground truth
    list_of_tar_token_idx = reduce(lambda x, y: x + y, [list_of_list_token_idx[i] for i in indices_to_mask])

    # get masked input
    mask_idx = _tokenizer.vocab_size + Ids.mask
    masked_input = []
    for i in range(len_words):
        idxs_for_word = list_of_list_token_idx[i]
        if i in indices_to_mask and random.random() > keep_origin_rate:
            masked_input += [mask_idx] * len(idxs_for_word)
        else:
            masked_input += idxs_for_word

    # get index for language embeddings; and add start, end token
    list_of_lan_idx_for_input = [lan_index] * int(len(masked_input) + 2)
    list_of_lan_idx_for_gt = [lan_index] * int(len(list_of_tar_token_idx) + 2)

    start = _tokenizer.vocab_size + Ids.start_nmt
    end = _tokenizer.vocab_size + Ids.end_nmt

    # add start end token
    masked_input = [start] + masked_input + [end]
    list_of_tar_token_idx = [start] + list_of_tar_token_idx + [end]

    return masked_input, list_of_tar_token_idx, list_of_lan_idx_for_input, list_of_lan_idx_for_gt


def get_pl(min_mask_token, max_mask_token, max_ratio_of_sent_len, keep_origin_rate):
    """ Get MLM pipeline """
    return [
        {
            'name': 'MLM_for_lan_1',
            'func': MLM_list_of_list_of_words,
            'input_keys': ['input_1', 'tokenizer', LanIds.zh,
                           min_mask_token, max_mask_token, max_ratio_of_sent_len, keep_origin_rate],
            'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1'],
            'show_dict': {'input_1': 'input_1', 'ground_truth_1': 'ground_truth_1',
                          'lan_idx_for_input_1': 'lan_idx_for_input_1', 'lan_idx_for_gt_1': 'lan_idx_for_gt_1'},
        },
        {
            'name': 'MLM_for_lan_2',
            'func': MLM_list_of_list_of_words,
            'input_keys': ['input_2', 'tokenizer', LanIds.en,
                           min_mask_token, max_mask_token, max_ratio_of_sent_len, keep_origin_rate],
            'output_keys': ['input_2', 'ground_truth_2', 'lan_idx_for_input_2', 'lan_idx_for_gt_2'],
            'show_dict': {'input_2': 'input_2', 'ground_truth_2': 'ground_truth_2',
                          'lan_idx_for_input_2': 'lan_idx_for_input_2', 'lan_idx_for_gt_2': 'lan_idx_for_gt_2'},
        },
    ]


if __name__ == '__main__':
    from nmt.preprocess.corpus import wmt_news
    from lib.preprocess import utils
    from nmt.preprocess.inputs import noise_pl, tfds_share_pl, zh_en
    from pretrain.preprocess.inputs import pl

    origin_zh_data, origin_en_data = wmt_news.zh_en()
    params = {
        'vocab_size': 45000,
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'max_src_ground_seq_len': 8,
        'max_tar_ground_seq_len': 8,
    }

    pipeline = zh_en.seg_zh_by_jieba_pipeline + noise_pl.remove_noise + tfds_share_pl.train_tokenizer
    pipeline += pl.sent_2_tokens + get_pl(1, 4, 0.2, 0.2) + pl.MLM_encode + [
        {'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1', 'tokenizer']}
    ]

    print('\n------------------- Encoding -------------------------')
    x, y, lan_x, lan_y, tokenizer = utils.pipeline(
        preprocess_pipeline=pipeline,
        lan_data_1=origin_zh_data[:1000], lan_data_2=origin_en_data[:1000], params=params)

    print('\n----------------------------------------------')
    print(x.shape)
    print(y.shape)
    print(lan_x.shape)
    print(lan_y.shape)

    print('\n------------------- Decoding zh -------------------------')
    x = utils.pipeline(tfds_share_pl.decode_pipeline, x, None, {'tokenizer': tokenizer})
    y = utils.pipeline(tfds_share_pl.decode_pipeline, y, None, {'tokenizer': tokenizer})
