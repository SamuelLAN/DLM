from pretrain.preprocess.config import dictionary_dir
import queue
import sys
import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor

data_dict = {}


def worker(input):
    quoted = "\"" + input + "\""
    out = subprocess.Popen(['youdao', quoted],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    check = out.communicate()[0].decode("utf-8").split("\n")
    for i in check[2:-2]:
        i = i.replace("\r", "")
        if "有道翻译" in i:
            data_dict[input] = {}
            data_dict[input]['translation'] = []
            data_dict[input]['translation'].append(i.replace("有道翻译：", ""))
            continue
        # i = i.replace("[;,；，]", "")
        if "n." in i:
            if "pos" not in data_dict[input]:
                data_dict[input]['pos'] = []
            if "noun" not in data_dict[input]['pos']:
                data_dict[input]['pos'].append("noun")
            for j in i.replace("n.", "").split("；"):
                if j not in data_dict[input]["translation"] and j is not "":
                    data_dict[input]["translation"].append(j)
            continue
        elif "v." in i:
            if "pos" not in data_dict[input]:
                data_dict[input]['pos'] = []
            if "verb" not in data_dict[input]['pos']:
                data_dict[input]['pos'].append("verb")
            for j in i.replace("v.", "").split("；"):
                if j not in data_dict[input]["translation"] and j is not "":
                    data_dict[input]["translation"].append(j)
            continue
        elif "adv." in i:
            if "pos" not in data_dict[input]:
                data_dict[input]['pos'] = []
            if "adverb" not in data_dict[input]['pos']:
                data_dict[input]['pos'].append("adverb")
            for j in i.replace("adv.", "").split("；"):
                if j not in data_dict[input]["translation"] and j is not "":
                    data_dict[input]["translation"].append(j)
            continue
        elif "adj." in i:
            if "pos" not in data_dict[input]:
                data_dict[input]['pos'] = []
            if "adjective" not in data_dict[input]['pos']:
                data_dict[input]['pos'].append("adjective")
            for j in i.replace("adj.", "").split("；"):
                if j not in data_dict[input]["translation"] and j is not "":
                    data_dict[input]["translation"].append(j)
            continue
        elif "prep." in i:
            if "pos" not in data_dict[input]:
                data_dict[input]['pos'] = []
            if "preposition" not in data_dict[input]['pos']:
                data_dict[input]['pos'].append("preposition")
            for j in i.replace("prep.", "").split("；"):
                if j not in data_dict[input]["translation"] and j is not "":
                    data_dict[input]["translation"].append(j)
            continue

        if english_indicator == "1":
            if ":" in i:
                tokens = i.split("  :  ")
                if input.lower() == tokens[0].lower():
                    list_exp = tokens[1].split(" ")
                    for j in list_exp:
                        if j not in data_dict[input]["translation"] and j is not "":
                            data_dict[input]["translation"].append(j)


q = queue.Queue()
num_wokers = 8
english_indicator = sys.argv[2]
with open(sys.argv[1]) as f:
    data = json.load(f)
    filename = sys.argv[1].replace(".json", ".txt")
    filename = os.path.basename(filename)
    print("start the threadpool for data")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in list(data.keys()):
            future = executor.submit(worker, (i))
            futures.append(future)
        for i in futures:
            while not i.done():
                continue
        filename = os.path.join(dictionary_dir, filename)
        with open(filename, 'w') as outfile:
            json.dump(data_dict, outfile)
        print("Finished writing data to file", outfile)
