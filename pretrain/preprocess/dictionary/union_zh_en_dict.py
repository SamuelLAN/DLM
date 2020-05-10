import os
from lib.utils import load_json, write_json
from pretrain.preprocess.config import filtered_en_zh_dict_path, filtered_zh_en_dict_path, dictionary_dir, filtered_union_en_zh_dict_path, filtered_union_zh_en_dict_path
from pretrain.preprocess.dictionary.preprocess_string import filter_duplicate

print('loading dictionary ...')

__en_zh_dict = load_json(filtered_en_zh_dict_path)
__zh_en_dict = load_json(filtered_zh_en_dict_path)

print('union dictionary ... ')

delete_ens = list(filter(lambda x: '-' in x, __en_zh_dict.keys()))
for en in delete_ens:
    del __en_zh_dict[en]

for zh, val in __zh_en_dict.items():
    if 'translation' not in val:
        continue

    translations = val['translation']
    if not translations:
        continue

    new_info = {}
    for k, v in val.items():
        if k == 'translation':
            new_info[k] = [zh]
            continue

        if 'src_' in k:
            k = k.replace('src_', 'tar_')
        elif 'tar_' in k:
            k = k.replace('tar_', 'src_')

        new_info[k] = v

    for en in translations:
        if en not in __en_zh_dict:
            __en_zh_dict[en] = new_info

        else:
            en_info = __en_zh_dict[en]
            if 'translation' not in en_info or not en_info['translation']:
                __en_zh_dict[en]['translation'] = [zh]

            elif zh not in en_info['translation'] and len(en_info['translation']) < 5:
                __en_zh_dict[en]['translation'].append(zh)

for en, val in __en_zh_dict.items():
    if 'translation' not in val:
        continue

    translations = val['translation']
    if not translations:
        continue

    new_info = {}
    for k, v in val.items():
        if k == 'translation':
            new_info[k] = [en]
            continue

        if 'src_' in k:
            k = k.replace('src_', 'tar_')
        elif 'tar_' in k:
            k = k.replace('tar_', 'src_')

        new_info[k] = v

    for zh in translations:
        if zh not in __zh_en_dict:
            __zh_en_dict[zh] = new_info

        else:
            zh_info = __zh_en_dict[zh]
            if 'translation' not in zh_info or not zh_info['translation']:
                __zh_en_dict[zh]['translation'] = [en]

            elif en not in zh_info['translation'] and len(en.split(' ')) >= 2 and ' sth ' not in en and len(zh_info['translation']) < 5:
                en = en.replace(',', '')
                __zh_en_dict[zh]['translation'].append(en)

print('filtering duplicate ...')

__zh_en_dict = filter_duplicate(__zh_en_dict)
__en_zh_dict = filter_duplicate(__en_zh_dict)

print('writing data to files ... ')

write_json(filtered_union_zh_en_dict_path, __zh_en_dict)
write_json(filtered_union_en_zh_dict_path, __en_zh_dict)

print('\ndone')
