import os
from lib.utils import load_json, write_json
from pretrain.preprocess.config import merged_en_ro_dict_path, merged_ro_en_dict_path,\
    filtered_merged_en_ro_dict_path, filtered_merged_ro_en_dict_path, \
    dictionary_dir, filtered_union_en_ro_dict_path, filtered_union_ro_en_dict_path
from pretrain.preprocess.dictionary.preprocess_string import filter_duplicate

print('loading dictionary ...')

__en_ro_dict = load_json(merged_en_ro_dict_path)
__ro_en_dict = load_json(merged_ro_en_dict_path)

print('union dictionary ... ')

delete_ens = list(filter(lambda x: '-' in x or '/' in x or ' or ' in x, __en_ro_dict.keys()))
for en in delete_ens:
    del __en_ro_dict[en]

for ro, val in __ro_en_dict.items():
    if 'translation' not in val:
        continue

    translations = val['translation']
    if not translations:
        continue

    new_info = {}
    for k, v in val.items():
        if k == 'translation':
            new_info[k] = [ro]
            continue

        if 'src_' in k:
            k = k.replace('src_', 'tar_')
        elif 'tar_' in k:
            k = k.replace('tar_', 'src_')

        new_info[k] = v

    for en in translations:
        if en not in __en_ro_dict:
            __en_ro_dict[en] = new_info

        else:
            en_info = __en_ro_dict[en]
            if 'translation' not in en_info or not en_info['translation']:
                __en_ro_dict[en]['translation'] = [ro]

            elif ro not in en_info['translation'] and len(en_info['translation']) < 5:
                __en_ro_dict[en]['translation'].append(ro)

for en, val in __en_ro_dict.items():
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

    for ro in translations:
        if ro not in __ro_en_dict:
            __ro_en_dict[ro] = new_info

        else:
            ro_info = __ro_en_dict[ro]
            if 'translation' not in ro_info or not ro_info['translation']:
                __ro_en_dict[ro]['translation'] = [en]

            elif en not in ro_info['translation'] and len(ro_info['translation']) < 5:
                en = en.replace(',', '')
                __ro_en_dict[ro]['translation'].append(en)

print('filtering duplicate ...')

__ro_en_dict = filter_duplicate(__ro_en_dict)
__en_ro_dict = filter_duplicate(__en_ro_dict)

print('writing data to files ... ')

write_json(filtered_union_ro_en_dict_path, __ro_en_dict)
write_json(filtered_union_en_ro_dict_path, __en_ro_dict)

print('\ndone')
