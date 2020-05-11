from lib.utils import load_json, write_json
from pretrain.preprocess.config import filtered_en_zh_dict_path, filtered_zh_en_dict_path
from pretrain.preprocess.dictionary.preprocess_string import filter_duplicate

__en_zh_dict = load_json(filtered_en_zh_dict_path)
__zh_en_dict = load_json(filtered_zh_en_dict_path)

__en_zh_dict = filter_duplicate(__en_zh_dict)
__zh_en_dict = filter_duplicate(__zh_en_dict)

delete_en_list = list(filter(lambda x: not x or x[0] == '-' or x[-1] == '-' or '/' in x or ' or ' in x, __en_zh_dict.keys()))
for en in delete_en_list:
    del __en_zh_dict[en]


delete_zh_list = []
for zh, val in __zh_en_dict.items():
    if 'translation' not in val or not val['translation']:
        continue

    translations = list(
        filter(lambda x: x and x[0] != '-' and x[-1] != '-' and '/' not in x and ' or ' not in x, val['translation']))

    if translations:
        __zh_en_dict[zh]['translation'] = translations

    else:
        if len(list(val.keys())) > 1:
            del __zh_en_dict[zh]['translation']
        else:
            delete_zh_list.append(zh)

for zh in delete_zh_list:
    del __zh_en_dict[zh]

write_json(filtered_en_zh_dict_path, __en_zh_dict)
write_json(filtered_zh_en_dict_path, __zh_en_dict)
