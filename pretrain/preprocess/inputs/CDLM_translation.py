import random
import copy
import numpy as np
from functools import reduce
from pretrain.preprocess.dictionary import map_dict


def CDLM_translation_list_of_list_of_words(list_of_list_of_words, _tokenizer, zh_lan_idx, en_lan_idx, is_zh,
                                           keep_origin_rate=0.2, mask_incr=3, sep_incr=4):
    """ MLM for batch data """
    data = list(map(
        lambda x: CDLM_translation(x, _tokenizer, zh_lan_idx, en_lan_idx, is_zh, keep_origin_rate, mask_incr, sep_incr),
        list_of_list_of_words
    ))

    list_input, list_output, list_lan_input, list_lan_output, list_pos_output = list(zip(*data))
    return list_input, list_output, list_lan_input, list_lan_output, list_pos_output


def CDLM_translation(list_of_words_for_a_sentence, _tokenizer, zh_lan_idx, en_lan_idx, is_zh,
                     keep_origin_rate=0.2, mask_incr=3, sep_incr=4):
    """

    :params
        list_of_words_for_a_sentence (list): ['I', 'am', 'a', 'student']
        tokenizer (object): tfds tokenizer object
        lan_index (int): index for language embeddings, could be 0 or 1
        min_num (int):
        max_num (int):
        max_ratio (float):
        keep_origin_rate (float):
        language (str): zh or en or both
    :returns
        masked_input (list): list of encoded and masked token idx
        list_of_tar_token_idx (list):
        list_of_lan_idx (list):
    """

    # get n grams
    list_of_token = list(map(lambda x: x.strip(), list_of_words_for_a_sentence))
    list_of_2_gram = map_dict.n_grams(list_of_token, 2)
    list_of_3_gram = map_dict.n_grams(list_of_token, 3)
    list_of_4_gram = map_dict.n_grams(list_of_token, 4)

    # map dictionary
    map_word = map_dict.zh_word if is_zh else map_dict.en_word
    map_phrase = map_dict.zh_phrase if is_zh else map_dict.en_phrase
    info_key = 'translation'

    list_of_info_for_word = list(map(lambda x: map_word(x, info_key), list_of_token))
    list_of_info_for_2_gram = list(map(lambda x: map_phrase(x, info_key), list_of_2_gram))
    list_of_info_for_3_gram = list(map(lambda x: map_phrase(x, info_key), list_of_3_gram))
    list_of_info_for_4_gram = list(map(lambda x: map_phrase(x, info_key), list_of_4_gram))

    # find the position that the corresponding word can be mapped with dictionary
    map_word_pos = map_dict.map_pos(list_of_info_for_word, 1)
    map_2_gram_pos = map_dict.map_pos(list_of_info_for_2_gram, 2)
    map_3_gram_pos = map_dict.map_pos(list_of_info_for_3_gram, 3)
    map_4_gram_pos = map_dict.map_pos(list_of_info_for_4_gram, 4)

    # if no map with dictionary
    if not map_word_pos and not map_2_gram_pos and not map_3_gram_pos and not map_4_gram_pos:
        return

    # BPE for each word
    list_of_list_token_idx = list(map(lambda x: _tokenizer.encode(x), list_of_words_for_a_sentence))

    # get all words or phrases that can be mapped with dictionary
    samples_to_be_selected = map_dict.merge_conflict_samples(len(list_of_words_for_a_sentence),
                                                             map_word_pos,
                                                             map_2_gram_pos,
                                                             map_3_gram_pos,
                                                             map_4_gram_pos)

    # only sample one word or phrase that can be mapped with dictionary
    sample = random.sample(samples_to_be_selected, 1)

    # get token index
    mask_idx = _tokenizer.vocab_size + mask_incr
    sep_idx = _tokenizer.vocab_size + sep_incr
    src_lan_idx = zh_lan_idx if is_zh else en_lan_idx
    tar_lan_idx = en_lan_idx if is_zh else zh_lan_idx

    _input = []
    _lan_input = []
    _output = []
    _lan_output = []
    _pos_output = []

    n = sample[1] - sample[0]
    if n == 4:
        translations = list_of_info_for_4_gram[sample[0]]
    elif n == 3:
        translations = list_of_info_for_3_gram[sample[0]]
    elif n == 2:
        translations = list_of_info_for_2_gram[sample[0]]
    else:
        translations = list_of_info_for_word[sample[0]]
    translations_ids = list(map(lambda x: tokenizer.encode(x + ' '), translations))

    mode = random.randint(0, 2)
    # replace the masked word with <mask>, and
    #    let the ground truth be its corresponding translation
    if mode == 0:

        index = 0
        len_words = len(list_of_list_token_idx)
        pos_for_mask = []

        while index < len_words:
            if index == sample[0]:
                len_tokens = sum([len(list_of_list_token_idx[i]) for i in range(sample[0], sample[1])])
                pos_for_mask = [len(_input), len(_input) + len_words]
                if random.random() < keep_origin_rate:
                    _input += reduce(lambda a, b: a + b, list_of_list_token_idx[sample[0]: sample[1]])
                else:
                    _input += [mask_idx] * len_words
                _lan_input += [tar_lan_idx] * len_tokens
                index = sample[1]
                continue

            _input += list_of_list_token_idx[index]
            _lan_input += [src_lan_idx] * len(list_of_list_token_idx[index])
            index += 1

        translations_ids = list(map(lambda x: [sep_idx] + x, translations_ids))

        # get token idxs for output
        _output = reduce(lambda a, b: a + b, translations_ids)
        _output.pop(0)

        # get language index for output
        _lan_output = [tar_lan_idx] * len(_output)

        # get soft position for output
        _pos_output = list(map(
            lambda x: list(map(int, np.linspace(pos_for_mask[0], pos_for_mask[1], len(x)))),
            translations_ids
        ))
        _pos_output = reduce(lambda a, b: a + b, _pos_output)
        _pos_output[1] = _pos_output[0]
        _pos_output.pop(0)

    # replace the masked word with <mask>, and
    #    let the ground truth be the original word
    if mode == 1:

        index = 0
        len_words = len(list_of_list_token_idx)
        pos_for_mask = []

        while index < len_words:
            if index == sample[0]:
                len_tokens = sum([len(list_of_list_token_idx[i]) for i in range(sample[0], sample[1])])
                pos_for_mask = [len(_input), len(_input) + len_words]
                _input += [mask_idx] * len_words
                _lan_input += [src_lan_idx] * len_tokens
                index = sample[1]
                continue

            _input += list_of_list_token_idx[index]
            _lan_input += [src_lan_idx] * len(list_of_list_token_idx[index])
            index += 1

        # get token idxs for output
        _output = [list_of_list_token_idx[i] for i in range(sample[0], sample[1])]
        _output = reduce(lambda a, b: a + b, _output)

        # get language index for output
        _lan_output = [src_lan_idx] * len(_output)

        # get soft position for output
        _pos_output = list(range(*pos_for_mask))

    # replace the masked word with its translation, and let the ground truth be its original word
    elif mode == 2:

        index = 0
        len_words = len(list_of_list_token_idx)
        pos_for_mask = []

        while index < len_words:
            if index == sample[0]:
                len_tokens = sum([len(list_of_list_token_idx[i]) for i in range(sample[0], sample[1])])

                pos_for_mask = [len(_input)]

                tmp_input = random.sample(translations_ids, 1)
                _input += tmp_input

                pos_for_mask.append(len(_input))

                _lan_input += [tar_lan_idx] * len(tmp_input)
                index = sample[1]
                continue

            _input += list_of_list_token_idx[index]
            _lan_input += [src_lan_idx] * len(list_of_list_token_idx[index])
            index += 1

        # get token idxs for output
        _output = [list_of_list_token_idx[i] for i in range(sample[0], sample[1])]
        _output = reduce(lambda a, b: a + b, _output)

        # get language index for output
        _lan_output = [src_lan_idx] * len(_output)

        # get soft position for output
        _pos_output = list(map(int, np.linspace(pos_for_mask[0], pos_for_mask[1], len(_output))))

    # # replace the masked word with its translation, let the ground truth be the tag of the source sequence;
    # #   the tag value is 0, 1; 0 indicates it is not replaced word, 1 indicates it is a replaced word
    # elif mode == 3:
    #     pass

    return _input, _output, _lan_input, _lan_output, _pos_output


def get_pl(keep_origin_rate, src_lan_idx=3, tar_lan_idx=4, mask_incr=3, sep_incr=4):
    """ Get MLM pipeline """
    return [
        {
            'name': 'CDLM_translation_for_lan_1',
            'func': CDLM_translation_list_of_list_of_words,
            'input_keys': ['input_1', 'tokenizer', src_lan_idx, tar_lan_idx, True,
                           keep_origin_rate, mask_incr, sep_incr],
            'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1', 'pos_for_gt_1'],
            'show_dict': {'input_1': 'input_1', 'ground_truth_1': 'ground_truth_1',
                          'lan_idx_for_input_1': 'lan_idx_for_input_1', 'lan_idx_for_gt_1': 'lan_idx_for_gt_1',
                          'pos_for_gt_1': 'pos_for_gt_1'},
        },
        {
            'name': 'CDLM_translation_for_lan_1',
            'func': CDLM_translation_list_of_list_of_words,
            'input_keys': ['input_2', 'tokenizer', tar_lan_idx, src_lan_idx, False,
                           keep_origin_rate, mask_incr, sep_incr],
            'output_keys': ['input_2', 'ground_truth_2', 'lan_idx_for_input_2', 'lan_idx_for_gt_2', 'pos_for_gt_2'],
            'show_dict': {'input_2': 'input_2', 'ground_truth_2': 'ground_truth_2',
                          'lan_idx_for_input_2': 'lan_idx_for_input_2', 'lan_idx_for_gt_2': 'lan_idx_for_gt_2',
                          'pos_for_gt_2': 'pos_for_gt_2'},
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
    pipeline += pl.sent_2_tokens + get_pl(0.2, 3, 4, 3, 4) + pl.encode + [
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
