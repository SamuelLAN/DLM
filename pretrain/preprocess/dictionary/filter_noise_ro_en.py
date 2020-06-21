import os
from pretrain.preprocess.config import dictionary_dir, merged_en_ro_dict_path, merged_ro_en_dict_path
from nmt.load import ro_en
from pretrain.preprocess.dictionary import map_dict_ro_en as map_dict
from lib.utils import load_json, write_json, cache, read_cache
from lib.preprocess.utils import stem, zh_word_seg_by_jieba
from pretrain.preprocess.dictionary.preprocess_string import filter_duplicate

cache_dict_path = os.path.join(dictionary_dir, 'cache_dict_ro_en.pkl')
if os.path.exists(cache_dict_path):
    ro_word_dict, en_word_dict = read_cache(cache_dict_path)

else:
    ro_word_dict = {}
    en_word_dict = {}


def __add_to_dict(lan_data, is_ro):
    length = len(lan_data)
    for i, sentence in enumerate(lan_data):
        if i % 20 == 0:
            progress = float(i + 1) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        words = sentence.strip('.').strip('?').strip('!').strip(';').strip().split(' ')

        list_of_token = list(map(lambda x: x.strip(), words))
        list_of_2_gram = map_dict.n_grams(list_of_token, 2)
        list_of_3_gram = map_dict.n_grams(list_of_token, 3)
        list_of_4_gram = map_dict.n_grams(list_of_token, 4)

        for w in list_of_token:
            if is_ro:
                ro_word_dict[w] = True
            else:
                en_word_dict[w] = True

                w_stems = stem(w)
                for w_stem in w_stems:
                    en_word_dict[w_stem] = True

        for phrase in list_of_2_gram:
            phrase = ''.join(phrase) if is_ro else ' '.join(phrase)
            if is_ro:
                ro_word_dict[phrase] = True
            else:
                en_word_dict[phrase] = True

        for phrase in list_of_3_gram:
            phrase = ''.join(phrase) if is_ro else ' '.join(phrase)
            if is_ro:
                ro_word_dict[phrase] = True
            else:
                en_word_dict[phrase] = True

        for phrase in list_of_4_gram:
            phrase = ''.join(phrase) if is_ro else ' '.join(phrase)
            if is_ro:
                ro_word_dict[phrase] = True
            else:
                en_word_dict[phrase] = True


if not os.path.exists(cache_dict_path):
    print('\nloading data from wmt news ...')
    loader_wmt = ro_en.Loader(0.0, 0.9)
    ro_data, en_data = loader_wmt.data()
    print('adding ro data to dict ... ')
    __add_to_dict(ro_data, True)
    print('adding en data to dict ... ')
    __add_to_dict(en_data, False)

    cache(cache_dict_path, [ro_word_dict, en_word_dict])

filtered_ro_en_dict_path = os.path.join(dictionary_dir, 'filtered_ro_en_merged.json')
filtered_en_ro_dict_path = os.path.join(dictionary_dir, 'filtered_en_ro_merged.json')

delete_ro_keys = []
delete_en_keys = []


def __check_has_val(val):
    for k, l in val.items():
        if l:
            return True
    return False


print('\nloading zh_en_dict ...')

ro_en_dict = load_json(merged_ro_en_dict_path)
ro_en_dict = filter_duplicate(ro_en_dict)

print('filtering zh_en_dict ...')

for ro, val in ro_en_dict.items():
    if 'translation' not in val:
        continue

    translations = val['translation']
    translations = list(filter(lambda x: x in en_word_dict, translations))

    if not translations:
        del ro_en_dict[ro]['translation']

        if not __check_has_val(ro_en_dict[ro]):
            delete_ro_keys.append(ro)

    else:
        ro_en_dict[ro]['translation'] = translations

print('filtering deleted keys for zh_en_dict ...')

for k in delete_ro_keys:
    del ro_en_dict[k]

print('writing data to file for zh_en_dict ... ')

write_json(filtered_ro_en_dict_path, ro_en_dict)

del ro_en_dict

print('\nloading en_zh_dict ...')

en_ro_dict = load_json(merged_en_ro_dict_path)
en_ro_dict = filter_duplicate(en_ro_dict)

print('filtering en_zh_dict ...')

for en, val in en_ro_dict.items():
    if 'translation' not in val:
        continue

    translations = val['translation']
    translations = list(filter(lambda x: x in ro_word_dict, translations))

    if not translations:
        del en_ro_dict[en]['translation']

        if not __check_has_val(en_ro_dict[en]):
            delete_en_keys.append(en)

    else:
        en_ro_dict[en]['translation'] = translations

print('filtering deleted keys for en_zh_dict ...')

for k in delete_en_keys:
    del en_ro_dict[k]

print('writing data to file for en_zh_dict ... ')

write_json(filtered_en_ro_dict_path, en_ro_dict)

print('\ndone')
