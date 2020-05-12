from lib.utils import load_json, write_json
from pretrain.preprocess.config import filtered_union_en_zh_dict_path, filtered_union_zh_en_dict_path, \
    filtered_pos_union_en_zh_dict_path, filtered_pos_union_zh_en_dict_path
from pretrain.preprocess.dictionary.preprocess_string import filter_duplicate

__en_zh_dict = load_json(filtered_union_en_zh_dict_path)
__zh_en_dict = load_json(filtered_union_zh_en_dict_path)

__en_zh_dict = filter_duplicate(__en_zh_dict)
__zh_en_dict = filter_duplicate(__zh_en_dict)

remove_pos = ['stuff', 'onn', 'pers', 'b', 'p', 'ing', 'mr', 'pl', 'x', 'k', 'en', 'z', 'modalv', 'pla', 'r', 'quant',
              'col', 'symb', 'suf', 'exclam', 'e', 'm', 'h', 'phn', 'y', 't', 'pf', 'tambalan', 'q', 'ipcb', 'sq',
              'linkv', 'suff', 'g', 'ind', 'l', 'ltd', 'w']

replace_pos_dict = {
    'vbl': 'v',
    'vi': 'v',
    'vt': 'v',
    'noun': 'n',
    'na': 'n',
    'un': 'n',
    'verb': 'v',
    'short': 'abbr',
    'd': 'num',
    'a': 'adj',
    'ad': 'adv',
    'ph': 'phr',
    'pr': 'pref',
    'pn': 'pron',
    'pp': 'prep',
}


def clean_pos(_poss):
    _poss = list(filter(lambda x: x not in remove_pos, _poss))
    _poss = list(map(lambda x: x if x not in replace_pos_dict else replace_pos_dict[x], _poss))
    return _poss


delete_zhs = []
delete_ens = []

for zh, val in __zh_en_dict.items():
    if 'pos' not in val or not val['pos']:
        continue
    val['pos'] = clean_pos(val['pos'])

    if not val['pos']:
        del val['pos']

    if not val:
        delete_zhs.append(zh)

for en, val in __en_zh_dict.items():
    if 'pos' not in val or not val['pos']:
        continue
    val['pos'] = clean_pos(val['pos'])

    if not val['pos']:
        del val['pos']

    if not val:
        delete_ens.append(en)

for k in delete_zhs:
    del __zh_en_dict[k]

for k in delete_ens:
    del __en_zh_dict[k]

write_json(filtered_pos_union_en_zh_dict_path, __en_zh_dict)
write_json(filtered_pos_union_zh_en_dict_path, __zh_en_dict)

print('done')
