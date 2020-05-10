import os
from pretrain.preprocess.config import dictionary_dir
from pretrain.preprocess.dictionary.preprocess_string import process, pipeline
from lib.utils import load_json, write_json

zh_en_dir = os.path.join(dictionary_dir, 'zh_en')
en_zh_dir = os.path.join(dictionary_dir, 'en_zh')

merged_zh_en_dict_path = os.path.join(dictionary_dir, 'zh_en_merged.json')
merged_en_zh_dict_path = os.path.join(dictionary_dir, 'en_zh_merged.json')


def __check_has_val(val):
    for k, l in val.items():
        if l:
            return True
    return False


def __filter_noise(val):
    for k, l in val.items():
        l = list(map(lambda x: process(x, pipeline), l))
        l = list(filter(lambda x: x, l))

        if k == 'translation' and len(l) >= 5:
            l = list(filter(lambda x: '.' not in x, l))

        if k == 'translation' and len(l) >= 5 and len(list(filter(lambda x: len(x) >= 5, l))) >= 3:
            l = list(filter(lambda x: len(x) >= 5, l))

        if not l:
            continue

        val[k] = l
    return val


def __merge_dict(_merged_dict, key, val, mode=0):
    # filter the noise and check if it still has values after filtering
    val = __filter_noise(val)
    if not __check_has_val(val):
        return _merged_dict

    # preprocess key
    key = process(key, pipeline)

    # if the dictionary does not contain this key
    if key not in _merged_dict:
        _merged_dict[key] = val
        return _merged_dict

    # if the dictionary contains this key
    for _type, l in val.items():
        if _type not in _merged_dict[key] or not _merged_dict[key][_type]:
            _merged_dict[key][_type] = l
            continue

        if mode != 0 and _type == 'translation' and _merged_dict[key][_type]:
            continue

        _merged_dict[key][_type] += l

    return _merged_dict


zh_en_dict = {}
en_zh_dict = {}

for file_name in os.listdir(zh_en_dir):
    file_path = os.path.join(zh_en_dir, file_name)

    print(f'\nloading dictionary from {file_path} ...')

    tmp_dict = load_json(file_path)

    print(f'merging dict {file_name} ...')

    mode = 0 if '_v_all' not in file_name else 1
    length = len(tmp_dict)
    i = 0
    for key, val in tmp_dict.items():
        if i % 50 == 0:
            progress = float(i + 1) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        __merge_dict(zh_en_dict, key, val, mode)
        i += 1

print('\nwriting data to files ...')

write_json(merged_zh_en_dict_path, zh_en_dict)
write_json(merged_en_zh_dict_path, en_zh_dict)

print('\ndone')

