import os
from lib.preprocess import utils
from lib.utils import write_json, load_json
from pretrain.preprocess.config import dictionary_dir
from pretrain.preprocess.dictionary.preprocess_string import process, pipeline, filter_duplicate

facebook_dir = os.path.join(dictionary_dir, 'muse', 'ro_en')

ro_en_dict_path = os.path.join(facebook_dir, 'ro_en_dict.json')
en_ro_dict_path = os.path.join(facebook_dir, 'en_ro_dict.json')

ro_en_dict = {}
en_ro_dict = {}


def __add_dict(_dict, k, v):
    if k not in _dict:
        _dict[k] = {'translation': []}
    _dict[k]['translation'].append(v)


for file_name in os.listdir(facebook_dir):
    if file_name not in ['en-ro.txt', 'ro-en.txt']:
        continue

    file_path = os.path.join(facebook_dir, file_name)
    print(f'reading from {file_path} ...')

    # read data
    lines = utils.read_lines(file_path)
    lines = list(map(lambda x: x.strip().split('\t'), lines))
    lines = list(filter(lambda x: x and len(x) == 2, lines))

    if file_name[:2].lower() == 'ro':
        for ro, en in lines:
            ro = process(ro, pipeline)
            en = process(en, pipeline)

            __add_dict(ro_en_dict, ro, en)
            __add_dict(en_ro_dict, en, ro)

    else:
        for en, ro in lines:
            ro = process(ro, pipeline)
            en = process(en, pipeline)

            __add_dict(ro_en_dict, ro, en)
            __add_dict(en_ro_dict, en, ro)

print('filtering duplicate ...')

ro_en_dict = filter_duplicate(ro_en_dict)
en_ro_dict = filter_duplicate(en_ro_dict)

print('writing dictionary to files ... ')

write_json(ro_en_dict_path, ro_en_dict)
write_json(en_ro_dict_path, en_ro_dict)

print('\ndone')
