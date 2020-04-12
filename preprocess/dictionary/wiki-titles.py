from preprocess.config import dictionary_dir
from lib.preprocess import utils
import os
import shutil
import glob
import csv
import re
import gzip
import json
url1 = "http://data.statmt.org/wikititles/v1/wikititles-v1.zh-en.tsv.gz"
url2 = "http://data.statmt.org/wikititles/v1/wikititles-v1.de-en.tsv.gz"
name1 = "wikititles-v1.zh-en"
name2 = "wikititles-v1.de-en"
file_name1 = "wikititles-v1.zh-en.tsv.gz"
file_name2 = "wikititles-v1.de-en.tsv.gz"
urls = [url1, url2]
names = [name1, name2]
file_names = [file_name1, file_name2]
saving_directory = dictionary_dir
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

for i in range(0,2):
    packfile = os.path.join(saving_directory, file_names[i])
    utils.download(urls[i], packfile)
    unzip_gz(packfile, saving_directory)
    # shutil.unpack_archive(packfile, saving_directory)
    if os.path.exists(packfile):
        os.remove(packfile)
    print("Download successfully, start extracting the training data.")
    targetFile = os.path.join(saving_news_dir,names[i] + ".*")
    file_name = glob.glob(targetFile, recursive=True)
    with open(file_name[0],encoding="utf-8") as ifile:
        print("reading file for ", file_name[0])
        read = csv.reader(ifile, delimiter='\t')
        data = {}
        for row in read:
            if len(row) < 2:
                continue
            if row[0] != row[1]:
                data[row[0]] = []
                data[row[0]].append({'translation': row[1]})
    saved_json =  os.path.join(saving_news_dir,names[i] + ".json")
    print("Writing json file for ", names[i])
    with open(saved_json, 'w') as outfile:
        json.dump(data, outfile)

print("done")
