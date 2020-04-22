import os
from nltk.corpus import wordnet as wn
from lib.utils import write_json
from pretrain.preprocess.dictionary import preprocess_string as utils
from pretrain.preprocess.config import dictionary_dir

write_dict_path = os.path.join(dictionary_dir, 'nltk_wordnet.txt')

dictionary = dict()
for word in wn.words():

    word_details = dict()
    syns = wn.synsets(word)
    src_synonyms = set()
    for x in syns:
        src_synonym = x.name().split(".")[0]
        if src_synonym != word:
            src_synonym = utils.process(src_synonym, utils.weak_pl)
            src_synonyms.add(src_synonym)

    src_meanings = list(filter(
        None, list(map(lambda x: utils.process(x.definition(), utils.weak_pl)
        if (word == x.name().split(".")[0]) else None, syns))
    ))
    pos = list(set(
        filter(None, list(map(lambda x: x.name().split(".")[1] if (word == x.name().split(".")[0]) else None, syns)))))

    if pos:
        word_details["pos"] = pos
    if src_meanings:
        word_details["src_meanings"] = src_meanings

    if src_synonyms:
        word_details["src_synonyms"] = list(src_synonyms)

    # convert coding and full 2 half
    word = utils.process(word, utils.weak_pl)

    dictionary[word] = word_details

write_json(write_dict_path, dictionary)
