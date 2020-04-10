import os

data_dir = r'D:\Data\DLM\data'
dictionary_dir = os.path.join(data_dir, 'dictionaries')

if not os.path.exists(dictionary_dir):
    os.mkdir(dictionary_dir)
