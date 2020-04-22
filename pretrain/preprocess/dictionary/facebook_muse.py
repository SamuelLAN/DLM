import os
from lib.preprocess import utils
from lib.utils import write_json
from pretrain.preprocess.config import dictionary_dir

facebook_dir = os.path.join(dictionary_dir, 'Facebook_Data')
read_data_path = os.path.join(facebook_dir, 'de-en.txt')
write_data_path = os.path.join(facebook_dir, 'de-en.json')

# read data
lines = utils.read_lines(read_data_path)

# parse data
dictionary = {}
for line in lines:
    words = line.split()
    dictionary[words[1]] = {"translation": [words[0]]}

# write data
write_json(write_data_path, dictionary)
