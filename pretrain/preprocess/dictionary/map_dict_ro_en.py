import re
import numpy as np
from functools import reduce
from lib.preprocess.utils import stem_ro, stem
from lib.utils import load_json
from pretrain.preprocess.config import merged_en_ro_dict_path, merged_ro_en_dict_path, \
    merged_stem_ro_dict_path, merged_stem_en_ro_dict_path
from pretrain.preprocess.dictionary.preprocess_string import filter_duplicate
from pretrain.preprocess.config import NERIds

__reg_d = re.compile(r'^\d+(\.\d+)?$')

__en_ro_dict = load_json(merged_en_ro_dict_path)
__ro_en_dict = load_json(merged_ro_en_dict_path)
__stem_dict = load_json(merged_stem_en_ro_dict_path)
__stem_ro_dict = load_json(merged_stem_ro_dict_path)

filter_en_word = {
    'the': True,
    'as': True,
    'on': True,
    'in': True,
    'a': True,
    'of': True,
    'for': True,
    'to': True,
    'over': True,
    'towards': True,
    'with': True,
    'and': True,
}


def decode_ner_id(_ner_id, offset):
    _id = _ner_id - offset
    if _id == NERIds.B:
        return 'B'
    elif _id == NERIds.M:
        return 'M'
    elif _id == NERIds.E:
        return 'E'
    elif _id == NERIds.O:
        return 'O'
    else:
        return


def __merge_dict(dict_list):
    if not dict_list:
        return {}
    elif len(dict_list) == 1:
        return dict_list[0]
    else:
        info = {}
        for _info in dict_list:
            for k, v in _info.items():
                if k not in info:
                    info[k] = v
                else:
                    info[k] += v

        for k, v in info.items():
            info[k] = list(set(v))
        return info


def __get_info(info, info_key='*'):
    # get all information
    if info_key == '*':
        return info

    # only get specific information
    if info_key in info and info[info_key]:
        return info[info_key]

    return {} if info_key == '*' else []


def ro_word(token, info_key='*'):
    if len(token) <= 2 or __reg_d.search(token):
        return {} if info_key == '*' else []

    if token in __ro_en_dict:
        return __get_info(__ro_en_dict[token], info_key)

    stem_token = stem_ro(token) if token not in __stem_ro_dict else token
    if stem_token in __stem_ro_dict:
        words = __stem_ro_dict[stem_token]
        infos = list(map(lambda x: __ro_en_dict[x], words))

        # if get all information
        if info_key == '*':
            return __merge_dict(infos)

        # only get specific information
        infos = list(map(lambda x: x[info_key] if info_key in x and x[info_key] else [], infos))
        infos = list(filter(lambda x: x, infos))

        if infos:
            return list(set(reduce(lambda a, b: a + b, infos)))

    return {} if info_key == '*' else []


def en_word(token, info_key='*'):
    if token in filter_en_word or len(token) <= 2 or __reg_d.search(token):
        return {} if info_key == '*' else []

    if token in __en_ro_dict:
        return __get_info(__en_ro_dict[token], info_key)

    stem_token = stem(token) if token not in __stem_dict else token
    if stem_token in __stem_dict:
        words = __stem_dict[stem_token]
        infos = list(map(lambda x: __en_ro_dict[x], words))

        # if get all information
        if info_key == '*':
            return __merge_dict(infos)

        # only get specific information
        infos = list(map(lambda x: x[info_key] if info_key in x and x[info_key] else [], infos))
        infos = list(filter(lambda x: x, infos))

        if infos:
            return list(set(reduce(lambda a, b: a + b, infos)))

    return {} if info_key == '*' else []


def word(token, info_key='*'):
    ro_info = ro_word(token, info_key)
    if ro_info:
        return ro_info
    return en_word(token, info_key)


def ro_phrase(list_of_tokens, info_key='*'):
    if not list_of_tokens:
        return {} if info_key == '*' else []

    if len(list_of_tokens) == 1:
        return ro_word(list_of_tokens[0], info_key)

    _phrase = ' '.join(list_of_tokens)
    phrase_info = ro_word(_phrase, info_key)
    if phrase_info:
        return phrase_info

    stem_phrase = ' '.join(list(map(stem, list_of_tokens)))
    phrase_info = ro_word(stem_phrase, info_key)
    if phrase_info:
        return phrase_info

    return {} if info_key == '*' else []


def en_phrase(list_of_tokens, info_key='*'):
    if not list_of_tokens:
        return {} if info_key == '*' else []

    if len(list_of_tokens) == 1:
        return en_word(list_of_tokens[0], info_key)

    _phrase = ' '.join(list_of_tokens)
    phrase_info = en_word(_phrase, info_key)
    if phrase_info:
        return phrase_info

    stem_phrase = ' '.join(list(map(stem, list_of_tokens)))
    phrase_info = en_word(stem_phrase, info_key)
    if phrase_info:
        return phrase_info

    return {} if info_key == '*' else []


def phrase(list_of_tokens, info_key='*'):
    zh_info = ro_phrase(list_of_tokens, info_key)
    if zh_info:
        return zh_info
    return en_phrase(list_of_tokens, info_key)


def n_grams(list_of_words_for_a_sentence, n=2):
    return [list_of_words_for_a_sentence[i: i + n] for i in range(len(list_of_words_for_a_sentence) - n + 1)]


def map_pos(list_of_info, n):
    return [[i, i + n] for i in range(len(list_of_info)) if list_of_info[i]]


def merge_conflict_samples(length, *args):
    pos = np.zeros(length)
    samples = []

    for map_pos_list in args:
        for start, end in map_pos_list:
            # if this gram has been taken by longer or shorter gram
            if np.sum(pos[start: end]):
                continue

            samples.append([start, end])
            pos[start: end] = len(samples)

    return samples


print('\nfinish loading dictionary')

# a = 'apples'
# b = '长城'
# c = '毛泽东'
# d = '茅台'
# e = 'iphone'
# f = ['takes', 'its', 'easy']
# g = ['calms', 'down']
#
# ens = [
#     a, e, 'abandoned', 'abandon', 'drink', 'Trump', 'Obama', 'Berkshire',
# ]
#
# for en in ens:
#     print(en, en_word(en, 'translation'))
