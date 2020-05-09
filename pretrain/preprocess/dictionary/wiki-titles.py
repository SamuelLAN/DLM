from pretrain.preprocess.config import dictionary_dir
from lib.preprocess import utils
from lib.utils import write_json, load_json
from pretrain.preprocess.dictionary.preprocess_string import process, pipeline, filter_duplicate
import os
import shutil
import glob
import csv
import re
import gzip

url1 = "http://data.statmt.org/wikititles/v2/wikititles-v2.zh-en.tsv.gz"
# url2 = "http://data.statmt.org/wikititles/v2/wikititles-v1.de-en.tsv.gz"
name1 = "wikititles-v1.zh-en"
# name2 = "wikititles-v1.de-en"
file_name1 = "wikititles-v1.zh-en.tsv.gz"
# file_name2 = "wikititles-v1.de-en.tsv.gz"
urls = [url1, ]
names = [name1, ]
file_names = [file_name1, ]

saving_directory = os.path.join(dictionary_dir, 'wiki-titles')
if not os.path.exists(saving_directory):
    os.mkdir(saving_directory)

saving_news_dir = os.path.splitext(saving_directory)[0]
print("starting to extract info")
print(saving_news_dir)


# if not os.path.exists(saving_news_dir):
#     print("creating the saving directory")
def unzip_gz(gzipped_file_name, work_dir):
    "gunzip the given gzipped file"
    # see warning about filename
    filename = os.path.split(gzipped_file_name)[-1]
    filename = re.sub(r"\.gz$", "", filename, flags=re.IGNORECASE)
    with gzip.open(gzipped_file_name, 'rb') as f_in:  # <<========== extraction happens here
        with open(os.path.join(work_dir, filename), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def __add_dict(_dict, k, v):
    if k not in _dict:
        _dict[k] = {'translation': []}
    _dict[k]['translation'].append(v)


for i in range(0, len(file_names)):
    pack_file = os.path.join(saving_directory, file_names[i])

    if not os.path.exists(pack_file):
        utils.download(urls[i], pack_file)
        unzip_gz(pack_file, saving_directory)
        # shutil.unpack_archive(packfile, saving_directory)

    if os.path.exists(pack_file):
        os.remove(pack_file)
    print("Download successfully, start extracting the training data.")

    target_file = os.path.join(saving_news_dir, names[i] + ".*")
    file_name = glob.glob(target_file, recursive=True)

    with open(file_name[0], encoding="utf-8") as f:
        print("reading file for ", file_name[0])
        read = csv.reader(f, delimiter='\t')

        zh_en_dict = {}
        en_zh_dict = {}

        for row in read:
            if len(row) != 2:
                continue

            zh_token, en_token = row
            if zh_token == en_token:
                continue

            zh_token = process(zh_token, pipeline)
            en_token = process(en_token, pipeline)

            if '/' in zh_token or '/' in en_token:
                if len(zh_token.split('/')) == len(en_token.split('/')):
                    zh_tokens = zh_token.split('/')
                    en_tokens = en_token.split('/')

                    for j, _zh_token in enumerate(zh_tokens):
                        _en_token = en_tokens[j]
                        __add_dict(zh_en_dict, _zh_token, _en_token)
                        __add_dict(en_zh_dict, _en_token, _zh_token)
                continue

            __add_dict(zh_en_dict, zh_token, en_token)
            __add_dict(en_zh_dict, en_token, zh_token)

    print('filtering duplicate ...')

    zh_en_dict = filter_duplicate(zh_en_dict)
    en_zh_dict = filter_duplicate(en_zh_dict)

    zh_en_dict_path = os.path.join(saving_news_dir, 'zh_en_dict_wiki_titles.json')
    en_zh_dict_path = os.path.join(saving_news_dir, 'en_zh_dict_wiki_titles.json')

    print('writing dictionary to files ... ')

    write_json(zh_en_dict_path, zh_en_dict)
    write_json(en_zh_dict_path, en_zh_dict)

print('\ndone')
