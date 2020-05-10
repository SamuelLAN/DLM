import json
import os
import sys
from pretrain.preprocess.config import dictionary_dir


def merge_dicts(dict_a, dict_b, union):
    d = {}
    dicts = [dict_a, dict_b]
    for dict in dicts:
        for key in list(dict):
            for subKey in list(dict[key].keys()):
                try:
                    if union:
                        d[key][subKey] = d[key][subKey] + list(set(dict[key][subKey]) - set(d[key][subKey]))
                    else:
                        continue
                except KeyError:
                    d[key] = (dict[key])
    return d


def merge_files(_input, output):
    data_lists = []
    for i in _input:
        with open(i, encoding="utf-8") as f:
            data = json.load(f)
            data_lists.append(data)

    out = {}
    for i, k in zip(data_lists[0::2], data_lists[1::2]):
        out = merge_dicts(out, i, k)
    with open(os.path.join(output, "eng-keys.json"), 'w') as outfile:
        json.dump(out, outfile)


def main():
    merge_files(sys.argv, dictionary_dir)


if __name__ == "__main__":
    main()
