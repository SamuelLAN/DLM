import random
from functools import reduce


def MLM_list_of_list_of_words(list_of_list_of_words, _tokenizer, lan_index,
                              min_num=2, max_num=6, max_ratio=0.2, keep_origin_rate=0.2):
    data = list(map(
        lambda x: MLM(x, _tokenizer, lan_index, min_num, max_num, max_ratio, keep_origin_rate),
        list_of_list_of_words
    ))
    list_masked_input, list_of_list_tar_token_idx, list_of_list_lan_idx = list(zip(*data))
    return list_masked_input, list_of_list_tar_token_idx, list_of_list_lan_idx


def MLM(list_of_words_for_a_sentence, _tokenizer, lan_index, min_num=2, max_num=6, max_ratio=0.2, keep_origin_rate=0.2):
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
        range(len_words),
        min(random.randint(min_num, max_num), round(len_words * max_ratio))
    )

    # get ground truth
    list_of_tar_token_idx = reduce(lambda x, y: x + y, [list_of_list_token_idx[i] for i in indices_to_mask])

    # get masked input
    mask_idx = _tokenizer.vocab_size + 3
    masked_input = []
    for i in range(len_words):
        idxs_for_word = list_of_list_token_idx[i]
        if i in indices_to_mask and random.random() > keep_origin_rate:
            masked_input += [mask_idx] * len(idxs_for_word)
        else:
            masked_input += idxs_for_word

    # get index for language embeddings
    list_of_lan_idx = [lan_index] * len(masked_input)

    return masked_input, list_of_tar_token_idx, list_of_lan_idx


if __name__ == '__main__':
    from preprocess.corpus import wmt_news
    from lib.preprocess import utils
    from preprocess.nmt_inputs import noise_pl, tfds_share_pl, zh_en
    from preprocess.pretrain import encode_decode

    origin_zh_data, origin_en_data = wmt_news.zh_en()
    params = {
        'vocab_size': 45000,
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'max_src_ground_seq_len': 6,
        'max_tar_ground_seq_len': 6,
    }

    pipeline = zh_en.seg_zh_by_jieba_pipeline + noise_pl.remove_noise + tfds_share_pl.train_tokenizer
    pipeline += encode_decode.split
    pipeline += [
        {
            'name': 'MLM_for_lan_1',
            'func': MLM_list_of_list_of_words,
            'input_keys': ['input_1', 'tokenizer', 3, 1, 6, 0.2, 0.2],
            'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_1'],
            'show_dict': {'input_1': 'input_1', 'ground_truth_1': 'ground_truth_1', 'lan_idx_1': 'lan_idx_1'},
        },
        {
            'name': 'MLM_for_lan_2',
            'func': MLM_list_of_list_of_words,
            'input_keys': ['input_2', 'tokenizer', 4, 1, 6, 0.2, 0.2],
            'output_keys': ['input_2', 'ground_truth_2', 'lan_idx_2'],
            'show_dict': {'input_2': 'input_2', 'ground_truth_2': 'ground_truth_2', 'lan_idx_2': 'lan_idx_2'},
        },
    ]
    pipeline += encode_decode.encode

    print('\n------------------- Encoding -------------------------')
    zh_data, en_data, zh_gt, en_gt, zh_lan_idxs, en_lan_idxs, tokenizer = utils.pipeline(
        preprocess_pipeline=pipeline,
        lan_data_1=origin_zh_data[:1000], lan_data_2=origin_en_data[:1000], params=params)

    print('\n----------------------------------------------')
    print(zh_data.shape)
    print(zh_gt.shape)
    print(zh_lan_idxs.shape)
    print('')
    print(en_data.shape)
    print(en_gt.shape)
    print(en_lan_idxs.shape)
    print(tokenizer.vocab_size)

    print('\n------------------- Decoding zh -------------------------')
    zh_data = utils.pipeline(tfds_share_pl.decode_pipeline, zh_data, None, {'tokenizer': tokenizer})
    zh_gt = utils.pipeline(tfds_share_pl.decode_pipeline, zh_gt, None, {'tokenizer': tokenizer})

    print('\n------------------- Decoding en -------------------------')
    en_data = utils.pipeline(tfds_share_pl.decode_pipeline, en_data, None, {'tokenizer': tokenizer})
    en_gt = utils.pipeline(tfds_share_pl.decode_pipeline, en_gt, None, {'tokenizer': tokenizer})
