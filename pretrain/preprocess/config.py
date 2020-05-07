import os

data_dir = r'D:\Data\DLM\data'
dictionary_dir = os.path.join(data_dir, 'dictionaries')

if not os.path.exists(dictionary_dir):
    os.mkdir(dictionary_dir)

merged_en_zh_dict_path = os.path.join(dictionary_dir, 'en_zh_merged.json')
merged_zh_en_dict_path = os.path.join(dictionary_dir, 'zh_en_merged.json')
merged_stem_dict_path = os.path.join(dictionary_dir, 'stem_dictionary.json')
