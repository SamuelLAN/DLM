import os
import sys
sys.path.append('/Users/jiayonglin/Desktop/828B/DLM/')
from pretrain.preprocess.config import dictionary_dir

from lib.utils import load_json, write_json
from collections import defaultdict

en_ro_dir = os.path.join(dictionary_dir, 'en_ro')
ro_en_dir = os.path.join(dictionary_dir, 'ro_en')


def __Invert_Dict(_d):
    inverted_d = defaultdict(lambda: defaultdict(list))
    for k, v in _d.items():
        inverted_d[v["translation"][0]]["translation"].append(k)

    return inverted_d


for file_name in os.listdir(en_ro_dir):
    file_path = os.path.join(en_ro_dir, file_name)
    en_ro_dict = load_json(file_path)
    ro_en_dict = __Invert_Dict(en_ro_dict)
    write_json(os.path.join(ro_en_dir, "ro_en" + file_name[5:]), ro_en_dict)