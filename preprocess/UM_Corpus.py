import os
import shutil

__data_dir = '/Users/jiayonglin/Desktop/828B/DLM'


def get(file_name):
    um_path = os.path.join(__data_dir, file_name)
    um_dir = os.path.splitext(um_path)[0]
    shutil.unpack_archive(um_path, um_dir)
    um_file_dir = os.path.join(um_dir,"UM-Corpus/data/Bilingual")
    um_file_dir_files = os.listdir(um_file_dir)
    lan_1_data, lan_2_data = [],[]
    for file in um_file_dir_files:
        with open(os.path.join(um_file_dir,file),encoding="utf8", errors='ignore') as f:
            lines = f.readlines()
            lan_1_data.extend(lines[1::2])
            lan_2_data.extend(lines[::2])
    lan_1_data = map(str.strip, lan_1_data) 
    lan_2_data = map(str.strip, lan_2_data)
    return lan_1_data, lan_2_data
    
def ch_en():
    """
    Return the Chinese-English corpus from WMT-news
    :return
        zh_data (list): list of sentences
        en_data (list): list of sentences
    """
    file_name = 'umcorpus-v1.zip'
    return get(file_name)