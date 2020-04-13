import os
os.chdir(r"C:\Users\Adam Lin\Documents\DLM")
from preprocess.config import dictionary_dir
from nltk.corpus import wordnet as wn
import json

Dict = dict()
for word in wn.words():

    word_details = dict()
    syns = wn.synsets(word)
    src_synonyms = set()
    for x in syns:
        if x.name().split(".")[0] != word:
            src_synonyms.add( x.name().split(".")[0])
    
    src_meanings = list(filter(None,list(map(lambda x: x.definition() if (word == x.name().split(".")[0]) else None, syns))))
    pos = list(set(filter(None,list(map(lambda x: x.name().split(".")[1] if (word == x.name().split(".")[0]) else None, syns)))))
    
    if pos:
        word_details["pos"] = pos
    if src_meanings:
        word_details["src_meanings"] = src_meanings
    
    if src_synonyms:
        word_details["src_synonyms"] = list(src_synonyms)
    Dict[word] = word_details

    
Dict = json.dumps(Dict)
with open('nltk_wordnet.txt', 'w') as outfile:
    json.dump(Dict, outfile)
