import re
import numpy as np
from functools import reduce
from lib.preprocess.utils import stem
from lib.utils import load_json
from pretrain.preprocess.config import filtered_pos_union_en_zh_dict_path, filtered_pos_union_zh_en_dict_path, \
    merged_stem_dict_path
from pretrain.preprocess.dictionary.preprocess_string import filter_duplicate
from pretrain.preprocess.config import NERIds

__reg_d = re.compile(r'^\d+(\.\d+)?$')

__en_zh_dict = load_json(filtered_pos_union_en_zh_dict_path)
__zh_en_dict = load_json(filtered_pos_union_zh_en_dict_path)
__stem_dict = load_json(merged_stem_dict_path)

filter_zh_word = {
    '的': True,
    '就': True,
    '来': True,
    '该': True,
    '在': True,
}

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

pos_dict = {'abbr_B': 1, 'abbr_I': 2, 'abbr_O': 3, 'adj_B': 4, 'adj_I': 5, 'adj_O': 6, 'adv_B': 7, 'adv_I': 8,
            'adv_O': 9, 'art_B': 10, 'art_I': 11, 'art_O': 12, 'aux_B': 13, 'aux_I': 14, 'aux_O': 15, 'c_B': 16,
            'c_I': 17, 'c_O': 18, 'comb_B': 19, 'comb_I': 20, 'comb_O': 21, 'conj_B': 22, 'conj_I': 23, 'conj_O': 24,
            'det_B': 25, 'det_I': 26, 'det_O': 27, 'inc_B': 28, 'inc_I': 29, 'inc_O': 30, 'int_B': 31, 'int_I': 32,
            'int_O': 33, 'interj_B': 34, 'interj_I': 35, 'interj_O': 36, 'n_B': 37, 'n_I': 38, 'n_O': 39, 'num_B': 40,
            'num_I': 41, 'num_O': 42, 'o_B': 43, 'o_I': 44, 'o_O': 45, 'phr_B': 46, 'phr_I': 47, 'phr_O': 48,
            'pref_B': 49, 'pref_I': 50, 'pref_O': 51, 'prep_B': 52, 'prep_I': 53, 'prep_O': 54, 'pron_B': 55,
            'pron_I': 56, 'pron_O': 57, 's_B': 58, 's_I': 59, 's_O': 60, 'st_B': 61, 'st_I': 62, 'st_O': 63, 'u_B': 64,
            'u_I': 65, 'u_O': 66, 'v_B': 67, 'v_I': 68, 'v_O': 69}

reverse_pos_dict = {1: 'abbr_B', 2: 'abbr_I', 3: 'abbr_O', 4: 'adj_B', 5: 'adj_I', 6: 'adj_O', 7: 'adv_B', 8: 'adv_I',
                    9: 'adv_O', 10: 'art_B', 11: 'art_I', 12: 'art_O', 13: 'aux_B', 14: 'aux_I', 15: 'aux_O', 16: 'c_B',
                    17: 'c_I', 18: 'c_O', 19: 'comb_B', 20: 'comb_I', 21: 'comb_O', 22: 'conj_B', 23: 'conj_I',
                    24: 'conj_O', 25: 'det_B', 26: 'det_I', 27: 'det_O', 28: 'inc_B', 29: 'inc_I', 30: 'inc_O',
                    31: 'int_B', 32: 'int_I', 33: 'int_O', 34: 'interj_B', 35: 'interj_I', 36: 'interj_O', 37: 'n_B',
                    38: 'n_I', 39: 'n_O', 40: 'num_B', 41: 'num_I', 42: 'num_O', 43: 'o_B', 44: 'o_I', 45: 'o_O',
                    46: 'phr_B', 47: 'phr_I', 48: 'phr_O', 49: 'pref_B', 50: 'pref_I', 51: 'pref_O', 52: 'prep_B',
                    53: 'prep_I', 54: 'prep_O', 55: 'pron_B', 56: 'pron_I', 57: 'pron_O', 58: 's_B', 59: 's_I',
                    60: 's_O', 61: 'st_B', 62: 'st_I', 63: 'st_O', 64: 'u_B', 65: 'u_I', 66: 'u_O', 67: 'v_B',
                    68: 'v_I', 69: 'v_O'}


def pos_id(pos, offset):
    if not pos or pos not in pos_dict:
        return
    return pos_dict[pos] + offset


def decode_pos_id(_pos_id, offset):
    _id = _pos_id - offset
    if _id not in reverse_pos_dict:
        return
    return reverse_pos_dict[_id]


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


def zh_word(token, info_key='*'):
    if token in filter_zh_word or token not in __zh_en_dict or __reg_d.search(token):
        return {} if info_key == '*' else []
    return __get_info(__zh_en_dict[token], info_key)


def en_word(token, info_key='*'):
    if token in filter_en_word or len(token) <= 2 or __reg_d.search(token):
        return {} if info_key == '*' else []

    if token in __en_zh_dict:
        return __get_info(__en_zh_dict[token], info_key)

    stem_token = stem(token) if token not in __stem_dict else token
    if stem_token in __stem_dict:
        words = __stem_dict[stem_token]
        infos = list(map(lambda x: __en_zh_dict[x], words))

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
    zh_info = zh_word(token, info_key)
    if zh_info:
        return zh_info
    return en_word(token, info_key)


def zh_phrase(list_of_tokens, info_key='*'):
    _phrase = ''.join(list_of_tokens)
    if _phrase in __zh_en_dict:
        return __get_info(__zh_en_dict[_phrase], info_key)
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
    zh_info = zh_phrase(list_of_tokens, info_key)
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

# print('\nfinish loading dictionary')
#
# a = 'apples'
# b = '长城'
# c = '毛泽东'
# d = '茅台'
# e = 'iphone'
# f = ['takes', 'its', 'easy']
# g = ['calms', 'down']
# h = ['苹果', '手机', '你好']
#
# zhs = [
#     b, c, d, '知识', '抛弃', '炮台', '酒精', '医用口罩', '语文课本', '奥巴马', '公寓', '邓小平',
# ]
# ens = [
#     a, e, 'abandoned', 'abandon', 'drink', 'Trump', 'Obama', 'Berkshire',
# ]
#
# for zh in zhs:
#     print(zh, zh_word(zh, 'translation'))
#
# for en in ens:
#     print(en, en_word(en, 'translation'))
#
# print(f, en_phrase(f, 'translation'))
# print(g, en_phrase(g, 'translation'))
# print(h, zh_phrase(h, 'translation'))


# pos_list = []
#
# for zh, val in __zh_en_dict.items():
#     if 'pos' not in val or not val['pos']:
#         continue
#     pos_list += val['pos']
#     pos_list = list(set(pos_list))
#
# for en, val in __en_zh_dict.items():
#     if 'pos' not in val or not val['pos']:
#         continue
#     pos_list += val['pos']
#     pos_list = list(set(pos_list))
#
# print(pos_list)
# print(len(pos_list))
