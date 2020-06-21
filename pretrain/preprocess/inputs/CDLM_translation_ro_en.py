import copy
import random
import numpy as np
from functools import reduce
from pretrain.preprocess.dictionary import map_dict_ro_en as map_dict
from pretrain.preprocess.config import Ids, LanIds, SampleRatio
from pretrain.preprocess.inputs.TLM import TLM_concat

random_state = 42

ratio_mode_0 = SampleRatio.translation['ratio_mode_0']
ratio_mode_1 = SampleRatio.translation['ratio_mode_1']
ratio_mode_2 = SampleRatio.translation['ratio_mode_2']

ratio_mode_0_1 = ratio_mode_0 + ratio_mode_1


def CDLM_MLM_sample(list_of_zh_words, list_of_en_words, _tokenizer, keep_origin_rate=0.2,
                    max_ratio=0.2, max_num=4):
    zh_data = list(map(
        lambda x: CDLM_translation(x, _tokenizer, True, keep_origin_rate, max_ratio, max_num), list_of_zh_words))
    en_data = list(map(
        lambda x: CDLM_translation(x, _tokenizer, False, keep_origin_rate, max_ratio, max_num), list_of_en_words))

    data = zh_data + en_data
    data = list(filter(lambda x: x[0] and x[1] and x[2] and x[3] and x[4], data))

    random.seed(random_state)
    random.shuffle(data)

    _inputs, _outputs, _lan_inputs, _lan_outputs, _soft_pos_outputs = list(zip(*data))
    return _inputs, _outputs, _lan_inputs, _lan_outputs, _soft_pos_outputs


def CDLM_TLM_sample(list_of_zh_words, list_of_en_words, _tokenizer, keep_origin_rate=0.2,
                    max_ratio=0.2, max_num=4):
    data = list(zip(list_of_zh_words, list_of_en_words))
    data = list(map(
        lambda x: CDLM_translation_for_zh_en(x[0], x[1], _tokenizer, keep_origin_rate, max_ratio, max_num), data))

    data = list(filter(lambda x: x[0][0] and x[0][1], data))
    data = list(map(lambda x: TLM_concat(*x), data))

    random.seed(random_state)
    random.shuffle(data)

    _inputs, _outputs, _lan_inputs, _lan_outputs, _soft_pos_outputs = list(zip(*data))
    return _inputs, _outputs, _lan_inputs, _lan_outputs, _soft_pos_outputs


def CDLM_combine_sample(list_of_zh_words, list_of_en_words, _tokenizer, keep_origin_rate=0.2, TLM_ratio=0.7,
                        max_ratio=0.2, max_num=4):
    data = []
    for i in range(len(list_of_zh_words)):
        list_of_zh_word = list_of_zh_words[i]
        list_of_en_word = list_of_en_words[i]

        # if TLM sample
        if random.random() < TLM_ratio:
            _inputs, _outputs, _lan_inputs, _lan_outputs, _soft_pos_outputs = CDLM_translation_for_zh_en(
                list_of_zh_word, list_of_en_word, _tokenizer, keep_origin_rate, max_ratio, max_num
            )

            if not _inputs[0] or not _inputs[1] or not _soft_pos_outputs[0] or not _soft_pos_outputs[1]:
                continue

            data.append(TLM_concat(_inputs, _outputs, _lan_inputs, _lan_outputs, _soft_pos_outputs))

        # MLM sample
        else:
            zh_input, zh_output, zh_lan_input, zh_lan_output, zh_soft_pos_output = CDLM_translation(
                list_of_zh_word, _tokenizer, True, keep_origin_rate, max_ratio, max_num)
            en_input, en_output, en_lan_input, en_lan_output, en_soft_pos_output = CDLM_translation(
                list_of_en_word, _tokenizer, False, keep_origin_rate, max_ratio, max_num)

            if zh_input and zh_output and zh_lan_input and zh_lan_output and zh_soft_pos_output:
                data.append([zh_input, zh_output, zh_lan_input, zh_lan_output, zh_soft_pos_output])
            if en_input and en_output and en_lan_input and en_lan_output and en_soft_pos_output:
                data.append([en_input, en_output, en_lan_input, en_lan_output, en_soft_pos_output])

    _inputs, _outputs, _lan_inputs, _lan_outputs, _soft_pos_outputs = list(zip(*data))
    return _inputs, _outputs, _lan_inputs, _lan_outputs, _soft_pos_outputs


def CDLM_translation_for_zh_en(list_of_zh_word, list_of_en_word, _tokenizer, keep_origin_rate=0.2,
                               max_ratio=0.2, max_num=4):
    zh_input, zh_output, zh_lan_input, zh_lan_output, zh_soft_pos_output = CDLM_translation(
        list_of_zh_word, _tokenizer, True, keep_origin_rate, max_ratio, max_num)
    en_input, en_output, en_lan_input, en_lan_output, en_soft_pos_output = CDLM_translation(
        list_of_en_word, _tokenizer, False, keep_origin_rate, max_ratio, max_num)
    return [zh_input, en_input], [zh_output, en_output], [zh_lan_input, en_lan_input], \
           [zh_lan_output, en_lan_output], [zh_soft_pos_output, en_soft_pos_output]


def get_translations(sample, list_of_info_for_4_gram, list_of_info_for_3_gram, list_of_info_for_2_gram,
                     list_of_info_for_word, list_of_words_for_a_sentence):
    n = sample[1] - sample[0]
    if n == 4:
        translations = list_of_info_for_4_gram[sample[0]]
    elif n == 3:
        translations = list_of_info_for_3_gram[sample[0]]
    elif n == 2:
        translations = list_of_info_for_2_gram[sample[0]]
    else:
        translations = list_of_info_for_word[sample[0]]
        translations = list(filter(lambda x: x != list_of_words_for_a_sentence[sample[0]], translations))

    # filter some noise of the translations
    translations = list(filter(lambda x: x, translations))
    return translations


def CDLM_translation(list_of_words_for_a_sentence, _tokenizer, is_ro, keep_origin_rate=0.2, max_ratio=0.2, max_num=4):
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
    list_of_token = list(map(lambda x: x.strip(), list_of_words_for_a_sentence[:-1]))
    list_of_2_gram = map_dict.n_grams(list_of_token, 2)
    list_of_3_gram = map_dict.n_grams(list_of_token, 3)
    list_of_4_gram = map_dict.n_grams(list_of_token, 4)

    # map dictionary
    map_word = map_dict.ro_word if is_ro else map_dict.en_word
    map_phrase = map_dict.ro_phrase if is_ro else map_dict.en_phrase
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
        return [], [], [], [], []

    # BPE for each word
    list_of_list_token_idx = list(map(lambda x: _tokenizer.encode(x), list_of_words_for_a_sentence))

    # get all words or phrases that can be mapped with dictionary
    samples_to_be_selected = map_dict.merge_conflict_samples(len(list_of_words_for_a_sentence),
                                                             map_4_gram_pos,
                                                             map_3_gram_pos,
                                                             map_2_gram_pos,
                                                             map_word_pos)

    mode = random.random()
    mode = 0 if mode <= ratio_mode_0 else (1 if mode <= ratio_mode_0_1 else 2)

    # only sample one word or phrase that can be mapped with dictionary
    sample = random.sample(samples_to_be_selected, 1)[0]
    samples = random.sample(
        samples_to_be_selected,
        random.randint(
            1,
            max(min(max_num, len(samples_to_be_selected), int(len(list_of_words_for_a_sentence) * max_ratio)), 1)
        ))
    samples.sort()

    # get token index
    mask_idx = _tokenizer.vocab_size + Ids.mask
    sep_idx = _tokenizer.vocab_size + Ids.sep
    src_lan_idx = LanIds.ro if is_ro else LanIds.en
    tar_lan_idx = LanIds.en if is_ro else LanIds.ro

    _input = []
    _lan_input = []
    _output = []
    _lan_output = []
    _soft_pos_output = []

    # for mode 0, we only need one sample
    translations = get_translations(sample, list_of_info_for_4_gram, list_of_info_for_3_gram, list_of_info_for_2_gram,
                                    list_of_info_for_word, list_of_words_for_a_sentence)

    # for mode 1 and 2, we would need multiple samples
    translations_list = [
        get_translations(_sample, list_of_info_for_4_gram, list_of_info_for_3_gram, list_of_info_for_2_gram,
                         list_of_info_for_word, list_of_words_for_a_sentence) for _sample in samples
    ]

    # remove those do not have translation info samples after filtering
    delete_samples = []
    for i, v in enumerate(translations_list):
        if not v:
            delete_samples.append(i)

    delete_samples.sort(reverse=True)
    for k in delete_samples:
        del translations_list[k]
        del samples[k]

    if not translations and not translations_list:
        return [], [], [], [], []

    if samples:
        samples_start, samples_end = list(zip(*samples))
    else:
        samples_start = samples_end = []

    # apply BPE for the translations
    translations_ids = list(map(lambda x: _tokenizer.encode(x + ' '), translations))
    translations_ids_list = list(map(
        lambda _translations: list(map(lambda x: _tokenizer.encode(x + ' '), _translations)),
        translations_list
    ))

    if not translations_list and translations:
        mode = 0
    elif not translations:
        mode = random.random()
        if mode < (ratio_mode_1 / (ratio_mode_1 + ratio_mode_2)):
            mode = 1
        else:
            mode = 2

    # mode = random.random()
    # mode = 0 if mode <= ratio_mode_0 else (1 if mode <= ratio_mode_0_1 else 2)
    # mode = random.randint(0, 2)

    start = _tokenizer.vocab_size + Ids.start_nmt
    end = _tokenizer.vocab_size + Ids.end_nmt

    # replace the masked word with <mask>, and
    #    let the ground truth be its corresponding translation
    if mode == 0:

        index = 0
        len_words = len(list_of_list_token_idx)
        pos_for_mask = []

        while index < len_words:
            if index == sample[0]:
                len_tokens = sum([len(list_of_list_token_idx[i]) for i in range(sample[0], sample[1])])
                pos_for_mask = [len(_input), len(_input) + len_tokens]
                if random.random() < keep_origin_rate:
                    _input += reduce(lambda a, b: a + b, list_of_list_token_idx[sample[0]: sample[1]])
                else:
                    _input += [mask_idx] * len_tokens
                _lan_input += [tar_lan_idx] * len_tokens
                index = sample[1]
                continue

            _input += list_of_list_token_idx[index]
            _lan_input += [src_lan_idx] * len(list_of_list_token_idx[index])
            index += 1

        translations_ids.sort()
        translations_ids = list(map(lambda x: [sep_idx] + x, translations_ids[:3]))
        new_translations_ids = copy.deepcopy(translations_ids)

        # get token idxs for output
        _output = reduce(lambda a, b: a + b, new_translations_ids)
        _output.pop(0)

        # get language index for output
        _lan_output = [tar_lan_idx] * len(_output)

        # get soft position for output
        # _soft_pos_output = [pos_for_mask[0]] * int(len(_output))
        _soft_pos_output = list(map(
            lambda x: list(map(lambda a: int(round(a)), np.linspace(pos_for_mask[0], pos_for_mask[1], len(x)))),
            translations_ids
        ))
        _soft_pos_output = reduce(lambda a, b: a + b, _soft_pos_output)
        # _soft_pos_output[1] = _soft_pos_output[0]
        _soft_pos_output.pop(0)

        start = _tokenizer.vocab_size + Ids.start_cdlm_t_0
        end = _tokenizer.vocab_size + Ids.end_cdlm_t_0

    # replace the masked word with <mask>, and
    #    let the ground truth be the original word
    elif mode == 1:

        index = 0
        len_words = len(list_of_list_token_idx)
        pos_for_mask = []

        while index < len_words:
            if index in samples_start:
                sample_idx = samples_start.index(index)
                sample_end = samples_end[sample_idx]

                len_tokens = sum([len(list_of_list_token_idx[i]) for i in range(index, sample_end)])
                pos_for_mask.append([len(_input), len(_input) + len_tokens])

                if random.random() < keep_origin_rate:
                    _input += reduce(lambda a, b: a + b, list_of_list_token_idx[index: sample_end])
                else:
                    _input += [mask_idx] * len_tokens
                _lan_input += [src_lan_idx] * len_tokens

                index = sample_end
                continue

            _input += list_of_list_token_idx[index]
            _lan_input += [src_lan_idx] * len(list_of_list_token_idx[index])
            index += 1

        # get token idxs for output
        _output = [[list_of_list_token_idx[i] for i in range(_sample[0], _sample[1])] for _sample in samples]
        _output = reduce(lambda a, b: a + b, _output)
        _output = reduce(lambda a, b: a + b, _output)

        # get language index for output
        _lan_output = [src_lan_idx] * len(_output)

        # get soft position for output
        _soft_pos_output = [list(range(_pos[0], _pos[1])) for _pos in pos_for_mask]
        _soft_pos_output = reduce(lambda a, b: a + b, _soft_pos_output)

        start = _tokenizer.vocab_size + Ids.start_mlm
        end = _tokenizer.vocab_size + Ids.end_mlm

    # replace the masked word with its translation, and let the ground truth be its original word
    elif mode == 2:

        index = 0
        len_words = len(list_of_list_token_idx)
        pos_for_mask = []

        while index < len_words:
            if index in samples_start:
                sample_idx = samples_start.index(index)
                sample_end = samples_end[sample_idx]

                _pos = [len(_input)]

                tmp_input = random.sample(translations_ids_list[sample_idx], 1)[0]
                _input += tmp_input

                _pos.append(len(_input))
                pos_for_mask.append(_pos)

                _lan_input += [tar_lan_idx] * len(tmp_input)
                index = sample_end
                continue

            _input += list_of_list_token_idx[index]
            _lan_input += [src_lan_idx] * len(list_of_list_token_idx[index])
            index += 1

        # get token idxs for output
        _output = [[list_of_list_token_idx[i] for i in range(_sample[0], _sample[1])] for _sample in samples]

        _soft_pos_output = [
            list(map(
                lambda a: int(round(a)),
                np.linspace(_pos[0], _pos[1], len(reduce(lambda a, b: a + b, _output[i])))
            )) for i, _pos in enumerate(pos_for_mask)
        ]
        _soft_pos_output = reduce(lambda a, b: a + b, _soft_pos_output)
        # _soft_pos_output = list(map(lambda a: int(round(a)), _soft_pos_output))

        _output = reduce(lambda a, b: a + b, _output)
        _output = reduce(lambda a, b: a + b, _output)

        # get language index for output
        _lan_output = [src_lan_idx] * len(_output)

        # get soft position for output
        # _soft_pos_output =
        # _soft_pos_output = list(
        #     map(lambda a: int(round(a)), np.linspace(pos_for_mask[0], pos_for_mask[1], len(_output))))
        # _soft_pos_output = [pos_for_mask[0]] * int(len(_output))

        start = _tokenizer.vocab_size + Ids.start_cdlm_t_2
        end = _tokenizer.vocab_size + Ids.end_cdlm_t_2

    # replace the masked word with its translation, let the ground truth be the tag of the source sequence;
    #   the tag value is 0, 1; 0 indicates it is not replaced word, 1 indicates it is a replaced word
    # elif mode == 3:
    #     pass

    # add <start> <end> token
    _input = [start] + _input + [end]
    _output = [start] + _output + [end]
    _lan_input = _lan_input[:1] + _lan_input + _lan_input[-1:]
    _lan_output = _lan_output[:1] + _lan_output + _lan_output[-1:]
    _soft_pos_output = _soft_pos_output[:1] + _soft_pos_output + _soft_pos_output[-1:]

    return _input, _output, _lan_input, _lan_output, _soft_pos_output


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


def TLM_pl(keep_origin_rate=0.2, max_ratio=0.2, max_num=4):
    return [
        {
            'name': 'CDLM_translation_MLM_sample',
            'func': CDLM_TLM_sample,
            'input_keys': ['input_1', 'input_2', 'tokenizer', keep_origin_rate, max_ratio, max_num],
            'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1', 'pos_for_gt_1'],
            'show_dict': {'input_1': 'input_1', 'ground_truth_1': 'ground_truth_1',
                          'lan_idx_for_input_1': 'lan_idx_for_input_1', 'lan_idx_for_gt_1': 'lan_idx_for_gt_1',
                          'pos_for_gt_1': 'pos_for_gt_1'},
        },
    ]


def combine_pl(keep_origin_rate=0.2, TLM_ratio=0.5, max_ratio=0.2, max_num=4):
    return [
        {
            'name': 'CDLM_translation_combine_sample',
            'func': CDLM_combine_sample,
            'input_keys': ['input_1', 'input_2', 'tokenizer', keep_origin_rate, TLM_ratio, max_ratio, max_num],
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
