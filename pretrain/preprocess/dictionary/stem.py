import os
from pretrain.preprocess.config import dictionary_dir
from lib.preprocess import utils
from lib.utils import load_json, write_json

dictionary_path = os.path.join(dictionary_dir, 'ecdict', 'en_zh_dict_from_ecdict.json')
stem_dictionary_path = os.path.join(dictionary_dir, 'stem_dictionary.json')

print(f'loading dictionary from {dictionary_path} ...')

dictionary = load_json(dictionary_path)

print('getting stems ...')

stem_words = list(dictionary.keys())
stem_words = list(filter(lambda x: len(x.split(' ')) <= 1, stem_words))
stem_words = list(map(lambda x: [utils.stem(x), x], stem_words))
stem_words = list(filter(lambda x: x[0] != x[1], stem_words))

print('converting to dict ...')

stem_dictionary = {}
len_stems = len(stem_words)
for i, (stem, word) in enumerate(stem_words):
    if i % 100 == 0:
        progress = float(i + 1) / len_stems * 100.
        print('\rprogress: %.2f%% ' % progress, end='')

    if stem not in stem_dictionary:
        stem_dictionary[stem] = []
    stem_dictionary[stem].append(word)

print('removing duplicate values ...')

for stem, words in stem_dictionary.items():
    words = list(set(words))
    stem_dictionary[stem] = words

print(f'\nwriting stem dictionary to file ...')

write_json(stem_dictionary_path, stem_dictionary)

print(f'\nlen_stem_dictionary: {len(stem_dictionary)}')

"""
The format of the stem dictionary is :
    {
        'like': ['likely', 'liking', 'alike'], 
        'cry': ['cries', 'crying'],
        ...
    }
"""
