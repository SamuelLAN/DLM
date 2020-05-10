import os
from pretrain.preprocess.config import dictionary_dir, merged_en_zh_dict_path, merged_zh_en_dict_path
from nmt.load import zh_en_wmt_news
from nmt.load import zh_en_um_corpus
from nmt.load import zh_en_news_commentary
from pretrain.preprocess.dictionary import map_dict
from lib.utils import load_json, write_json, cache, read_cache
from lib.preprocess.utils import stem, zh_word_seg_by_jieba
from pretrain.preprocess.dictionary.preprocess_string import filter_duplicate

cache_dict_path = os.path.join(dictionary_dir, 'cache_dict.pkl')
if os.path.exists(cache_dict_path):
    zh_word_dict, en_word_dict = read_cache(cache_dict_path)

else:
    zh_word_dict = {}
    en_word_dict = {}


def __add_to_dict(lan_data, is_zh):
    length = len(lan_data)
    for i, sentence in enumerate(lan_data):
        if i % 20 == 0:
            progress = float(i + 1) / length * 100.
            print('\rprogress: %.2f%% ' % progress, end='')

        if is_zh:
            words = sentence[:-1]
        else:
            words = sentence.strip('.').strip('?').strip('!').strip(';').split(' ')

        list_of_token = list(map(lambda x: x.strip(), words))
        list_of_2_gram = map_dict.n_grams(list_of_token, 2)
        list_of_3_gram = map_dict.n_grams(list_of_token, 3)
        list_of_4_gram = map_dict.n_grams(list_of_token, 4)

        for w in list_of_token:
            if is_zh:
                zh_word_dict[w] = True
            else:
                en_word_dict[w] = True

                w_stems = stem(w)
                for w_stem in w_stems:
                    en_word_dict[w_stem] = True

        for phrase in list_of_2_gram:
            phrase = ''.join(phrase) if is_zh else ' '.join(phrase)
            if is_zh:
                zh_word_dict[phrase] = True
            else:
                en_word_dict[phrase] = True

        for phrase in list_of_3_gram:
            phrase = ''.join(phrase) if is_zh else ' '.join(phrase)
            if is_zh:
                zh_word_dict[phrase] = True
            else:
                en_word_dict[phrase] = True

        for phrase in list_of_4_gram:
            phrase = ''.join(phrase) if is_zh else ' '.join(phrase)
            if is_zh:
                zh_word_dict[phrase] = True
            else:
                en_word_dict[phrase] = True


if not os.path.exists(cache_dict_path):

    print('\nloading data from wmt news ...')
    loader_wmt = zh_en_wmt_news.Loader(0.0, 0.9)
    zh_data, en_data = loader_wmt.data()
    zh_data = zh_word_seg_by_jieba(zh_data)
    print('adding zh data to dict ... ')
    __add_to_dict(zh_data, True)
    print('adding en data to dict ... ')
    __add_to_dict(en_data, False)

    del zh_data, en_data, loader_wmt

    print('\nloading data from um corpus ...')
    loader_um_corpus = zh_en_um_corpus.Loader(0.0, 1.0, is_test=False)
    zh_data, en_data = loader_um_corpus.data()
    zh_data = zh_word_seg_by_jieba(zh_data)
    print('adding zh data to dict ... ')
    __add_to_dict(zh_data, True)
    print('adding en data to dict ... ')
    __add_to_dict(en_data, False)

    del zh_data, en_data, loader_um_corpus

    print('\nloading data from news commentary ...')
    loader_news_commentary = zh_en_news_commentary.Loader(0.0, 0.98)
    zh_data, en_data = loader_news_commentary.data()
    zh_data = zh_word_seg_by_jieba(zh_data)
    print('adding zh data to dict ... ')
    __add_to_dict(zh_data, True)
    print('adding en data to dict ... ')
    __add_to_dict(en_data, False)

    del zh_data, en_data, loader_news_commentary

    # cache(cache_dict_path, [zh_word_dict, en_word_dict])

filtered_zh_en_dict_path = os.path.join(dictionary_dir, 'filtered_zh_en_merged.json')
filtered_en_zh_dict_path = os.path.join(dictionary_dir, 'filtered_en_zh_merged.json')

delete_zh_keys = []
delete_en_keys = []


def __check_has_val(val):
    for k, l in val.items():
        if l:
            return True
    return False


print('\nloading zh_en_dict ...')

zh_en_dict = load_json(merged_zh_en_dict_path)
zh_en_dict = filter_duplicate(zh_en_dict)

print('filtering zh_en_dict ...')

for zh, val in zh_en_dict.items():
    if 'translation' not in val:
        continue

    translations = val['translation']
    translations = list(filter(lambda x: x in en_word_dict, translations))

    if not translations:
        del zh_en_dict[zh]['translation']

        if not __check_has_val(zh_en_dict[zh]):
            delete_zh_keys.append(zh)

    else:
        zh_en_dict[zh]['translation'] = translations

print('filtering deleted keys for zh_en_dict ...')

for k in delete_zh_keys:
    del zh_en_dict[k]

print('writing data to file for zh_en_dict ... ')

write_json(filtered_zh_en_dict_path, zh_en_dict)

del zh_en_dict

print('\nloading en_zh_dict ...')

en_zh_dict = load_json(merged_en_zh_dict_path)
en_zh_dict = filter_duplicate(en_zh_dict)

print('filtering en_zh_dict ...')

for en, val in en_zh_dict.items():
    if 'translation' not in val:
        continue

    translations = val['translation']
    translations = list(filter(lambda x: x in zh_word_dict, translations))

    if not translations:
        del en_zh_dict[en]['translation']

        if not __check_has_val(en_zh_dict[en]):
            delete_en_keys.append(en)

    else:
        en_zh_dict[en]['translation'] = translations

print('filtering deleted keys for en_zh_dict ...')

for k in delete_en_keys:
    del en_zh_dict[k]

print('writing data to file for en_zh_dict ... ')

write_json(filtered_en_zh_dict_path, en_zh_dict)

print('\ndone')
