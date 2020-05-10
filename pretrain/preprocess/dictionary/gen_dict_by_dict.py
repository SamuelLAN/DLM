import os
from pretrain.preprocess.config import dictionary_dir
import pretrain.preprocess.dictionary.preprocess_string as utils
from lib.preprocess.utils import zh_traditional_2_simplified
from lib.utils import load_json, write_json

_dict_path = os.path.join(dictionary_dir, 'muse', 'zh_en_dict.json')
new_dict_path = os.path.join(dictionary_dir, 'wiki-titles', 'en_zh_dict_wiki_titles.json')

# dictionary = load_json(_dict_path)
dictionary = load_json(new_dict_path)


s = 0
for k, v in dictionary.items():
    print(k, v['translation'])
    s += 1
    if s > 1000:
        break

print('\n-----------------------------')
print(len(dictionary))

exit()

new_dict = {}


def __add_dict(_dict, k, v):
    if k not in _dict:
        _dict[k] = {'translation': []}
    _dict[k]['translation'].append(v)


for key, val in dictionary.items():
    translations = val['translation']
    for translation in translations:
        if translation not in new_dict:
            __add_dict(new_dict, translation, key)

write_json(new_dict_path, new_dict)

print('done')
