from lib.utils import load_json, write_json
from pretrain.preprocess.config import filtered_en_zh_dict_path, filtered_zh_en_dict_path
from pretrain.preprocess.dictionary.preprocess_string import filter_duplicate

__en_zh_dict = load_json(filtered_en_zh_dict_path)
__zh_en_dict = load_json(filtered_zh_en_dict_path)

__en_zh_dict = filter_duplicate(__en_zh_dict)
__zh_en_dict = filter_duplicate(__zh_en_dict)

write_json(filtered_en_zh_dict_path, __en_zh_dict)
write_json(filtered_zh_en_dict_path, __zh_en_dict)
