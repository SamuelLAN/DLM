from lib.preprocess.utils import stem
from lib.utils import load_json
from pretrain.preprocess.config import merged_en_zh_dict_path, merged_zh_en_dict_path, merged_stem_dict_path

__en_zh_dict = load_json(merged_en_zh_dict_path)
__zh_en_dict = load_json(merged_zh_en_dict_path)
__stem_dict = load_json(merged_stem_dict_path)


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


def zh_word(token):
    if token in __zh_en_dict:
        return __zh_en_dict[token]
    return {}


def en_word(token):
    if token in __en_zh_dict:
        return __en_zh_dict[token]

    stem_token = stem(token)
    if stem_token in __stem_dict:
        words = __stem_dict[stem_token]
        infos = list(map(lambda x: __en_zh_dict[x], words))
        return __merge_dict(infos)

    return {}


def zh_phrase(list_of_tokens):
    phrase = ''.join(list_of_tokens)
    if phrase in __zh_en_dict:
        return __zh_en_dict[phrase]
    return {}


def en_phrase(list_of_tokens):
    if not list_of_tokens:
        return {}

    if len(list_of_tokens) == 1:
        return en_word(list_of_tokens[0])

    phrase = ' '.join(list_of_tokens)
    if phrase in __en_zh_dict:
        return __en_zh_dict[phrase]

    list_of_list_stems = list(map(stem, list_of_tokens))

    phrase_list = list_of_list_stems[0]

    for stems in list_of_list_stems[1:]:
        len_stems = len(stems)
        len_phrase = len(phrase_list)
        phrase_list = phrase_list * len_stems
        phrase_list = [v + ' ' + stems[int(i / len_phrase)] for i, v in enumerate(phrase_list)]

    infos = [__en_zh_dict[phrase] for phrase in phrase_list if phrase in __en_zh_dict]
    return __merge_dict(infos)


# print('\nfinish loading dictionary')
#
# a = 'apples'
# b = '长城'
# c = '毛泽东'
# d = '茅台'
# e = 'iphone'
# f = ['take', 'it', 'easy']
# g = ['calm', 'down']
# h = ['苹果', '手机']
#
# zhs = [
#     b, c, d, '知识', '抛弃', '炮台', '酒精', '医用口罩', '语文课本',
# ]
# ens = [
#     a, e, 'abandoned', 'abandon', 'drink', 'Trump', 'Obama'
# ]
#
# for zh in zhs:
#     print(zh, zh_word(zh))
#
#
# for en in ens:
#     print(en, en_word(en))
#
# print(en_phrase(f), f)
# print(en_phrase(g), g)
# print(zh_phrase(h), h)
#
