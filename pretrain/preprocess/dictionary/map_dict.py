import numpy as np
from functools import reduce
from lib.preprocess.utils import stem
from lib.utils import load_json
from pretrain.preprocess.config import filtered_union_en_zh_dict_path, filtered_union_zh_en_dict_path, merged_stem_dict_path
from pretrain.preprocess.dictionary.preprocess_string import filter_duplicate

__en_zh_dict = load_json(filtered_union_en_zh_dict_path)
__zh_en_dict = load_json(filtered_union_zh_en_dict_path)
__stem_dict = load_json(merged_stem_dict_path)

filter_zh_word = {
    '的': True,
    '就': True,
    '来': True,
}

filter_en_word = {
    'the': True,
}


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
        return filter_duplicate(info)


def __get_info(info, info_key='*'):
    # get all information
    if info_key == '*':
        return info

    # only get specific information
    if info_key in info and info[info_key]:
        return info[info_key]

    return {} if info_key == '*' else []


def zh_word(token, info_key='*'):
    if token in filter_zh_word or token not in __zh_en_dict:
        return {} if info_key == '*' else []
    return __get_info(__zh_en_dict[token], info_key)


def en_word(token, info_key='*'):
    if token in filter_en_word or len(token) <= 1:
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
