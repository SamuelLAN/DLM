import os

data_dir = r'D:\Data\DLM\data'
dictionary_dir = os.path.join(data_dir, 'dictionaries')

if not os.path.exists(dictionary_dir):
    os.mkdir(dictionary_dir)

merged_en_zh_dict_path = os.path.join(dictionary_dir, 'en_zh_merged_v_all.json')
merged_zh_en_dict_path = os.path.join(dictionary_dir, 'zh_en_merged_v_all.json')
merged_stem_dict_path = os.path.join(dictionary_dir, 'stem_dictionary.json')

filtered_en_zh_dict_path = os.path.join(dictionary_dir, 'filtered_en_zh_merged.json')
filtered_zh_en_dict_path = os.path.join(dictionary_dir, 'filtered_zh_en_merged.json')

filtered_union_en_zh_dict_path = os.path.join(dictionary_dir, 'filtered_en_zh_merged_union.json')
filtered_union_zh_en_dict_path = os.path.join(dictionary_dir, 'filtered_zh_en_merged_union.json')


# vocabulary index
class Ids:
    start_nmt = 1
    end_nmt = 2
    mask = 3
    sep = 4
    start_cdlm_t_0 = 5
    end_cdlm_t_0 = 6
    start_cdlm_t_1 = 7
    end_cdlm_t_1 = 8
    start_cdlm_t_2 = 9
    end_cdlm_t_2 = 10
    start_cdlm_pos = 11
    end_cdlm_pos = 12
    start_cdlm_ner = 13
    end_cdlm_ner = 14
    start_cdlm_synonym = 15
    end_cdlm_synonym = 16
    start_cdlm_def = 17
    end_cdlm_def = 18


class LanIds:
    zh = 0
    en = 1
