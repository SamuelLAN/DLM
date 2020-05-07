import os
import re
import copy
import functools
import pandas as pd
import numpy as np
from pretrain.preprocess.config import dictionary_dir
import pretrain.preprocess.dictionary.preprocess_string as utils
from lib.utils import write_json

dict_dir = os.path.join(dictionary_dir, 'ecdict')
dict_path = os.path.join(dict_dir, 'stardict.csv')

print('\nloading data ... ')

# load data
data = pd.read_csv(dict_path)
columns = list(data.columns)
data = np.array(data)

# initialize variables
zh_en_dict = {}
en_zh_dict = {}

# TODO delete this row
# data = list(filter(lambda x: str(x[5]).lower() != 'nan' or str(x[6]).lower() != 'nan', data))

print('\nformatting data ...')

# format data
data = list(map(lambda x: {'en_word': x[0], 'en_meanings': x[2], 'zh_translation': x[3], 'pos': x[4]}, data))

# filter only one character
data = list(filter(lambda x: len(str(x['en_word']).replace('\'', '').strip()) > 1, data))

# filter data that has too long translation
data = list(filter(lambda x: len(str(x['zh_translation'])) < 60, data))

# initialize some useful variables
__reg_en_split = re.compile(r'[\n;]|\\n')
__reg_zh_split = re.compile(r'[\n;；,，。或]|\\n')
__reg_pos_for_meanings = re.compile(r'^([a-z]|adv|adj|prep|noun|verb|vi|vt)\.?\s+', re.IGNORECASE)
__reg_pos_for_translation = re.compile(r'^([a-z]+)\.[a-z]*\s+', re.IGNORECASE)
__reg_remove_translation = re.compile('(^表示(?=[\u4e00-\u9fa5])|来源于[\u4e00-\u9fa5]+|含义是)')
__reg_all_num = re.compile('^\d+$')
__reg_enter = re.compile(r'\\n   (\w+\s+)')


def unify_pos_symbol(_pos_list):
    # complement pos
    if ('un' in _pos_list or 'cn' in _pos_list) and 'n' not in _pos_list:
        _pos_list.append('n')
    if ('vi' in _pos_list or 'vt' in _pos_list) and 'v' not in _pos_list:
        _pos_list.append('v')

    # replace symbol
    while 'a' in _pos_list:
        _pos_list.remove('a')
        _pos_list.append('adj')
    while 'r' in _pos_list:
        _pos_list.remove('r')
        _pos_list.append('adv')
    while 'i' in _pos_list:
        _pos_list.remove('i')
        _pos_list.append('prep')
    while 'j' in _pos_list:
        _pos_list.remove('j')
        _pos_list.append('adj')
    return list(set(_pos_list))


def initialize_dict_element():
    return {
        'translation': [],
        'pos': [],
        'src_meanings': [],
        'tar_meanings': [],
        'src_synonyms': [],
        'tar_synonyms': [],
    }


keep_bracket_content_pl = copy.deepcopy(utils.pipeline)
keep_bracket_content_pl[-2] = utils.remove_bracket

print('\ntraversing data ...\n')

length = len(data)

for i, v in enumerate(data):
    # show progress
    if i % 1000 == 0:
        progress = float(i + 1) / length * 100.
        print('\rprogress: %.2f%% ' % progress, end='')

    en_word = utils.process(v['en_word'], utils.pipeline)
    # en_meanings = utils.process(v['en_meanings'], utils.pipeline)
    en_meanings = utils.process(v['en_meanings'], keep_bracket_content_pl)
    zh_translation = utils.process(v['zh_translation'], utils.pipeline)
    pos = utils.process(v['pos'], utils.pipeline)

    if '.' in en_word:
        continue

    # if i % 1000 == 0:
    #     print(f'\n{en_word:30s} | {en_meanings:20s} | {str(v["en_meanings"]).strip().lower():20s} | {zh_translation:40s} | {pos:20s} |')

    pos_list = []

    if en_meanings:
        en_meanings = __reg_enter.sub(r' \1', en_meanings)
        en_meanings = list(map(lambda x: x.strip(), __reg_en_split.split(en_meanings)))
        pos_from_meanings = list(map(lambda x: __reg_pos_for_meanings.findall(x), en_meanings))
        en_meanings = list(map(lambda x: __reg_pos_for_meanings.sub('', x).strip(), en_meanings))
        en_meanings = list(map(lambda x: utils.remove_not_en(x).strip(), en_meanings))
        en_meanings = list(map(lambda x: __reg_all_num.sub('', x).strip().strip('-').strip(), en_meanings))
        en_meanings = list(filter(lambda x: len(x.split(' ')) < 25, en_meanings))
        while '' in en_meanings:
            en_meanings.remove('')
        pos_list += functools.reduce(lambda a, b: a + b, pos_from_meanings)
        en_meanings = list(set(en_meanings))
    else:
        en_meanings = []

    if zh_translation:
        zh_translation = list(map(lambda x: x.strip(), __reg_zh_split.split(zh_translation)))
        pos_from_translation = list(map(lambda x: __reg_pos_for_translation.findall(x), zh_translation))
        zh_translation = list(map(lambda x: __reg_pos_for_translation.sub('', x).strip(), zh_translation))
        zh_translation = list(map(lambda x: __reg_remove_translation.sub('', x).strip(), zh_translation))
        zh_translation = list(map(lambda x: __reg_all_num.sub('', x).strip(), zh_translation))
        while '' in zh_translation:
            zh_translation.remove('')
        pos_list += functools.reduce(lambda a, b: a + b, pos_from_translation)
        zh_translation = list(set(zh_translation))
    else:
        zh_translation = []

    if pos:
        pos = pos.split('/')
        pos = list(map(lambda x: x.split(':')[0].strip(), pos))
        pos_list += pos

    pos_list = unify_pos_symbol(pos_list)

    # if i % 1000 == 0:
    #     print(en_meanings)
    #     print(zh_translation)
    #     print(pos_list)
    #     # print(en_synonyms)

    # add data to en_zh_dict
    if en_word not in en_zh_dict:
        en_zh_dict[en_word] = initialize_dict_element()

    # add data to en_zh_dict
    en_zh_dict[en_word]['translation'] += zh_translation
    en_zh_dict[en_word]['pos'] += pos_list
    en_zh_dict[en_word]['src_meanings'] += en_meanings
    # en_zh_dict[en_word]['src_synonyms'].union(en_synonyms)

    # add data to zh_en_dict
    for zh_word in zh_translation:
        if zh_word not in zh_en_dict:
            zh_en_dict[zh_word] = initialize_dict_element()

        zh_en_dict[zh_word]['translation'].append(en_word)
        zh_en_dict[zh_word]['pos'] += pos_list
        zh_en_dict[zh_word]['tar_meanings'] += en_meanings

print('\n\nfiltering duplicate elements ...')

en_zh_dict = utils.filter_duplicate(en_zh_dict)
zh_en_dict = utils.filter_duplicate(zh_en_dict)

print('\nsaving data ... ')

write_json(os.path.join(dict_dir, 'en_zh_dict_from_ecdict.json'), en_zh_dict)
write_json(os.path.join(dict_dir, 'zh_en_dict_from_ecdict.json'), zh_en_dict)

print('\nanalyzing ...')

print(f'len en_zh_dict: {len(en_zh_dict)}')
print(f'len zh_en_dict: {len(zh_en_dict)}')

print('\ndone')

# len en_zh_dict: 3322644
# len zh_en_dict: 2827802
