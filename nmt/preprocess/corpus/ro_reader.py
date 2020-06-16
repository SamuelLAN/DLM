# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 23:38:34 2020

@author: Research
"""
import os
from lib.preprocess import utils
import shutil
from nmt.preprocess.config import data_dir

def en_ro():
    file_name = 'europarl_v8_en-ro.tgz'
    url =  'http://data.statmt.org/wmt16/translation-task/training-parallel-ep-v8.tgz'
    return get(url, file_name, 'training-parallel-ep-v8/europarl-v8.ro-en.en', 'training-parallel-ep-v8/europarl-v8.ro-en.ro')