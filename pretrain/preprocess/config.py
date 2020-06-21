import os

data_dir = r'C:\Users\zshua\PycharmProjects\DLM\data\dictionaries\WMT-News.en-zh.en'
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

merged_en_ro_dict_path = os.path.join(dictionary_dir, 'en_ro_merged_v_all.json')
merged_ro_en_dict_path = os.path.join(dictionary_dir, 'ro_en_merged_v_all.json')
merged_stem_ro_dict_path = os.path.join(dictionary_dir, 'stem_ro_dictionary.json')
merged_stem_en_ro_dict_path = os.path.join(dictionary_dir, 'stem_en_ro_dictionary.json')

filtered_merged_en_ro_dict_path = os.path.join(dictionary_dir, 'filtered_en_ro_merged.json')
filtered_merged_ro_en_dict_path = os.path.join(dictionary_dir, 'filtered_ro_en_merged.json')

filtered_union_en_ro_dict_path = os.path.join(dictionary_dir, 'filtered_en_ro_union.json')
filtered_union_ro_en_dict_path = os.path.join(dictionary_dir, 'filtered_ro_en_union.json')


# vocabulary index
class IdsClass:
    multi_task = False
    multi_lan = True

    cdlm_tasks = []
    # cdlm_tasks = ['translation', 'pos', 'ner', 'synonym', 'def']

    pos_ids = 69
    ner_ids = 4

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

    @property
    def start_cdlm_pos_0(self):
        return 11 if self.multi_task else 7

    @property
    def end_cdlm_pos_0(self):
        return 12 if self.multi_task else 8

    @property
    def start_cdlm_pos_2(self):
        return 13 if self.multi_task else 9

    @property
    def end_cdlm_pos_2(self):
        return 14 if self.multi_task else 10

    @property
    def start_cdlm_ner_0(self):
        return 15 if self.multi_task else 7

    @property
    def end_cdlm_ner_0(self):
        return 16 if self.multi_task else 8

    @property
    def start_cdlm_ner_2(self):
        return 17 if self.multi_task else 9

    @property
    def end_cdlm_ner_2(self):
        return 18 if self.multi_task else 10

    @property
    def start_cdlm_synonym_0(self):
        return 19 if self.multi_task else 7

    @property
    def end_cdlm_synonym_0(self):
        return 20 if self.multi_task else 8

    @property
    def start_cdlm_synonym_2(self):
        return 21 if self.multi_task else 9

    @property
    def end_cdlm_synonym_2(self):
        return 22 if self.multi_task else 10

    @property
    def start_cdlm_def(self):
        return 23 if self.multi_task else 7

    @property
    def end_cdlm_def(self):
        return 24 if self.multi_task else 8

    @property
    def offset_pos(self):
        if not self.multi_task:
            return self.end_cdlm_pos_2

        incr = self.end_mlm
        if 'def' in self.cdlm_tasks:
            incr += 2
            incr += 4 * (len(self.cdlm_tasks) - 1)
        else:
            incr += 4 * len(self.cdlm_tasks)

        return incr

    @property
    def offset_ner(self):
        if not self.multi_task:
            return self.end_cdlm_ner_2

        incr = self.end_mlm
        if 'def' in self.cdlm_tasks:
            incr += 2
            incr += 4 * (len(self.cdlm_tasks) - 1)
        else:
            incr += 4 * len(self.cdlm_tasks)

        if 'pos' in self.cdlm_tasks:
            incr += self.pos_ids
        return incr


Ids = IdsClass()


class NERIds:
    B = 1
    M = 2
    E = 3
    O = 4


class LanIds:
    zh = 0
    en = 1
    ro = 2
    POS = 3 if Ids.multi_lan else 2
    NER = POS + 1 if Ids.multi_task else (3 if Ids.multi_lan else 2)


class SampleRatio:
    translation = {
        'ratio_mode_0': 0.3,
        'ratio_mode_1': 0.15,
        'ratio_mode_2': 0.55,
    }

    pos = {
        'ratio_mode_0': 0.5,
        'ratio_mode_1': 0.15,
        'ratio_mode_2': 0.35,
    }

    ner = {
        'ratio_mode_0': 0.3,
        'ratio_mode_1': 0.15,
        'ratio_mode_2': 0.55,
    }

    synonym = {
        'ratio_mode_0': 0.3,
        'ratio_mode_1': 0.15,
        'ratio_mode_2': 0.55,
    }
