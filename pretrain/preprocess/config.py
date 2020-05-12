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

filtered_pos_union_en_zh_dict_path = os.path.join(dictionary_dir, 'filtered_pos_en_zh_merged_union.json')
filtered_pos_union_zh_en_dict_path = os.path.join(dictionary_dir, 'filtered_pos_zh_en_merged_union.json')


# vocabulary index
class Ids:
    multi_task = False

    start_nmt = 1
    end_nmt = 2
    mask = 3
    sep = 4
    start_mlm = 5
    end_mlm = 6
    start_cdlm_t_0 = 7
    end_cdlm_t_0 = 8
    start_cdlm_t_2 = 9
    end_cdlm_t_2 = 10
    start_cdlm_pos_0 = 11 if multi_task else 7
    end_cdlm_pos_0 = 12 if multi_task else 8
    start_cdlm_pos_2 = 13 if multi_task else 9
    end_cdlm_pos_2 = 14 if multi_task else 10
    start_cdlm_ner = 15 if multi_task else 7
    end_cdlm_ner = 16 if multi_task else 8
    start_cdlm_synonym = 17 if multi_task else 7
    end_cdlm_synonym = 18 if multi_task else 8
    start_cdlm_def = 19 if multi_task else 7
    end_cdlm_def = 20 if multi_task else 8


class LanIds:
    zh = 0
    en = 1
    POS = 2
