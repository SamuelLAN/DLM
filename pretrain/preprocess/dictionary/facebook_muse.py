import os
from lib.preprocess import utils
from lib.utils import write_json, load_json
from pretrain.preprocess.config import dictionary_dir
from pretrain.preprocess.dictionary.preprocess_string import process, pipeline, filter_duplicate

facebook_dir = os.path.join(dictionary_dir, 'muse')

zh_en_dict_path = os.path.join(facebook_dir, 'zh_en_dict.json')
en_zh_dict_path = os.path.join(facebook_dir, 'en_zh_dict.json')

zh_en_dict = {}
en_zh_dict = {}


def __add_dict(_dict, k, v):
    if k not in _dict:
        _dict[k] = {'translation': []}
    _dict[k]['translation'].append(v)


for file_name in os.listdir(facebook_dir):
    file_path = os.path.join(facebook_dir, file_name)
    print(f'reading from {file_path} ...')

    suffix = os.path.splitext(file_name)[1].lower()
    if suffix == '.json':
        data = load_json(file_path)

        if file_name[:2].lower() == 'zh':

            for zh, val in data.items():
                zh = process(zh, pipeline)

                ens = val['translation']
                for en in ens:
                    en = process(en, pipeline)

                    __add_dict(zh_en_dict, zh, en)
                    __add_dict(en_zh_dict, en, zh)

        else:
            for en, val in data.items():
                en = process(en, pipeline)

                zhs = val['translation']
                for zh in zhs:
                    zh = process(zh, pipeline)

                    __add_dict(zh_en_dict, zh, en)
                    __add_dict(en_zh_dict, en, zh)

    else:
        # read data
        lines = utils.read_lines(file_path)
        lines = list(map(lambda x: x.strip().split(' '), lines))
        lines = list(filter(lambda x: x and len(x) == 2, lines))

        if file_name[:2].lower() == 'zh':
            for zh, en in lines:
                zh = process(zh, pipeline)
                en = process(en, pipeline)

                __add_dict(zh_en_dict, zh, en)
                __add_dict(en_zh_dict, en, zh)

        else:
            for en, zh in lines:
                zh = process(zh, pipeline)
                en = process(en, pipeline)

                __add_dict(zh_en_dict, zh, en)
                __add_dict(en_zh_dict, en, zh)

print('filtering duplicate ...')

zh_en_dict = filter_duplicate(zh_en_dict)
en_zh_dict = filter_duplicate(en_zh_dict)

print('writing dictionary to files ... ')

write_json(zh_en_dict_path, zh_en_dict)
write_json(en_zh_dict_path, en_zh_dict)

print('\ndone')
