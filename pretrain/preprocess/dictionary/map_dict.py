import numpy as np
from functools import reduce
from lib.preprocess.utils import stem
from lib.utils import load_json
from pretrain.preprocess.config import merged_en_zh_dict_path, merged_zh_en_dict_path, merged_stem_dict_path

__en_zh_dict = load_json(merged_en_zh_dict_path)
__zh_en_dict = load_json(merged_zh_en_dict_path)
__stem_dict = load_json(merged_stem_dict_path)

filter_zh_word = {
    '的': True,
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
    if token in filter_zh_word or token not in __zh_en_dict:
        return {} if info_key == '*' else []
    return __get_info(__zh_en_dict[token], info_key)


def en_word(token, info_key='*'):
    if token in filter_zh_word:
        return {} if info_key == '*' else []

    if token in __en_zh_dict:
        return __get_info(__en_zh_dict[token], info_key)

    stem_token = stem(token)
    if stem_token in __stem_dict:
        words = __stem_dict[stem_token]
        infos = list(map(lambda x: __en_zh_dict[x], words))

        # if get all information
        if info_key == '*':
            return __merge_dict(infos)

        # only get specific information
        infos = list(map(lambda x: x[info_key] if info_key in x and x[info_key] else [], infos))
        return reduce(lambda a, b: a + b, infos)

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
    if _phrase in __en_zh_dict:
        return __get_info(__en_zh_dict[_phrase], info_key)

    list_of_list_stems = list(map(stem, list_of_tokens))

    phrase_list = list_of_list_stems[0]

    for stems in list_of_list_stems[1:]:
        len_stems = len(stems)
        len_phrase = len(phrase_list)
        phrase_list = phrase_list * len_stems
        phrase_list = [v + ' ' + stems[int(i / len_phrase)] for i, v in enumerate(phrase_list)]

    infos = [__en_zh_dict[phrase] for phrase in phrase_list if phrase in __en_zh_dict]

    if not infos:
        return {} if info_key == '*' else []

    # if get all information
    if info_key == '*':
        return __merge_dict(infos)

    # only get specific information
    infos = list(map(lambda x: x[info_key] if info_key in x and x[info_key] else [], infos))
    return reduce(lambda a, b: a + b, infos)


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
# f = ['take', 'it', 'easy']
# g = ['calm', 'down', 'man']
# h = ['苹果', '手机', '你好']
#
# zhs = [
#     b, c, d, '知识', '抛弃', '炮台', '酒精', '医用口罩', '语文课本', '奥巴马', '公寓',
# ]
# ens = [
#     a, e, 'abandoned', 'abandon', 'drink', 'Trump', 'Obama', 'Berkshire',
# ]
#
# for zh in zhs:
#     print(zh, zh_word(zh, 'translation'))
#
#
# for en in ens:
#     print(en, en_word(en, 'translation'))
#
# print(en_phrase(f, 'translation'), f)
# print(en_phrase(g, 'translation'), g)
# print(zh_phrase(h, 'translation'), h)
#
