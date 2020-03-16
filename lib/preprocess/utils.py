import os
import jieba
import pkuseg
import chardet
import zipfile
import unicodedata
import numpy as np
from gensim import corpora
import tensorflow_datasets as tfds
from six.moves.urllib.request import urlretrieve
from nltk.tokenize import word_tokenize


def download(url, file_path):
    """ download data """
    dir_path = os.path.splitext(file_path)[0]
    if os.path.exists(dir_path) or os.path.exists(file_path):
        print('%s exists' % file_path)
        return

    def progress(count, block_size, total_size):
        print('\r>> Download %.2f%% ' % (float(count * block_size) / total_size * 100.), end='')

    print('Start downloading from %s ' % url)
    new_file_path, _ = urlretrieve(url, file_path, reporthook=progress)
    stat_info = os.stat(new_file_path)
    print('\nSuccessfully download from %s %d bytes' % (url, stat_info.st_size))


def unzip_and_delete(file_path):
    """ unzip files """

    new_dir_path = os.path.splitext(file_path)[0]
    if os.path.exists(new_dir_path):
        print('%s has been unzip' % file_path)
        return
    os.mkdir(new_dir_path)

    print('\nStart unzipping data ... ')

    zip_file = zipfile.ZipFile(file_path)
    for names in zip_file.namelist():
        zip_file.extract(names, new_dir_path)
    zip_file.close()

    # delete the zip file
    os.remove(file_path)

    print('Finish unzipping \n')


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def decode_2_utf8(string):
    if not isinstance(string, bytes):
        return string

    try:
        return string.decode('utf-8')
    except:
        encoding = chardet.detect(string)['encoding']
        if encoding:
            try:
                return string.decode(encoding)
            except:
                pass
        return string


def read_lines(file_path):
    """ read files and return a list of every line in the file; each line would be decoded to utf8 """
    with open(file_path, 'rb') as f:
        content = f.readlines()
    return list(map(lambda x: unicode_to_ascii(decode_2_utf8(x)).strip(), content))


def zh_word_seg_by_pku(list_of_sentences, user_dict=[]):
    """
    Tokenize Chinese words by pkuseg
    :params
        list_of_sentences (list): [ sentence_a (str), sentence_b (str), ... ]
        user_dict (list): customized dictionary, e.g., [ '你好', '朋友', ... ]
    """
    user_dict = user_dict if user_dict else 'default'
    seg = pkuseg.pkuseg(user_dict)
    return list(map(lambda x: seg.cut(x), list_of_sentences))


def zh_word_seg_by_jieba(list_of_sentences):
    """ Tokenize Chinese words by jieba """
    return list(map(lambda x: list(jieba.cut(x)), list_of_sentences))


def en_word_seg_by_nltk(list_of_sentences):
    """ tokenize English words by NLTK """
    return list(map(lambda x: word_tokenize(x), list_of_sentences))


def char_seg(list_of_sentences):
    """ tokenize sentence to character level """
    return list(map(lambda x: list(x), list_of_sentences))


def train_subword_tokenizer_by_tfds(list_of_sentences, vocab_size=2 ** 13, max_subword_len=20, reserved_tokens=None):
    return tfds.features.text.SubwordTextEncoder.build_from_corpus(
        list_of_sentences,
        target_vocab_size=vocab_size,
        max_subword_length=max_subword_len,
        reserved_tokens=reserved_tokens,
    )


def encoder_string_2_subword_idx_by_tfds(tokenizer, list_of_sentences):
    """
    encode string to subword idx
    :param
        tokenizer (tfds object): a subword tokenizer built from corpus by tfds
        list_of_sentences (list): [
          'Hello, I am a student',
          'You are my friends',
          ...
        ]
    :return
        list_of_list_token_idx (list): [
            [12, 43, 2, 346, 436, 87, 876],   # correspond to ['He', 'llo', ',', 'I', 'am', 'stu', 'dent'],
            [32, 57, 89, 98, 96, 37],         # correspond to ['You', 'are', 'my', 'fri', 'end', 's'],
            ...
        ]
    """
    return list(map(lambda x: tokenizer.encode(x), list_of_sentences))


def decode_subword_idx_2_string_by_tfds(tokenizer, list_of_list_token_idx):
    """
    decode subword_idx to string
    :param
        tokenizer (tfds object): a subword tokenizer built from corpus by tfds
        list_of_list_token_idx (list): [
            [12, 43, 2, 346, 436, 87, 876],   # correspond to ['He', 'llo', ',', 'I', 'am', 'stu', 'dent'],
            [32, 57, 89, 98, 96, 37],         # correspond to ['You', 'are', 'my', 'fri', 'end', 's'],
            ...
        ]
    :return
        list_of_sentences (list): [
          'Hello, I am a student',
          'You are my friends',
          ...
        ]
    """
    return list(map(lambda x: tokenizer.decode(x), list_of_list_token_idx))


def decode_subword_idx_2_tokens_by_tfds(tokenizer, list_of_list_token_idx):
    """
    decode subword_idx to string
    :param
        tokenizer (tfds object): a subword tokenizer built from corpus by tfds
        list_of_list_token_idx (list): [
            [12, 43, 2, 346, 436, 87, 876],   # correspond to ['He', 'llo', ',', 'I', 'am', 'stu', 'dent'],
            [32, 57, 89, 98, 96, 37],         # correspond to ['You', 'are', 'my', 'fri', 'end', 's'],
            ...
        ]
    :return
        list_of_list_token (list): [
            ['He', 'llo', ',', 'I', 'am', 'stu', 'dent'],
            ['You', 'are', 'my', 'fri', 'end', 's'],
            ...
        ]
    """
    return list(map(lambda x: list(map(lambda a: tokenizer.decode([a]), x)), list_of_list_token_idx))


def doc_2_idx(list_of_doc, dictionary=None, keep_n=5000):
    """
    convert words to token index
    :param
      list_of_doc (list): [
        ['hello', ',', 'I', 'am', 'a', 'student', '.'],
        ['you', 'are', 'my', 'friend', '.'],
        ...
      ],
      dictionary (gensim.corpora.Dictionary): default value is None. The dictionary
        will be generated automatically if it is None.
      keep_n (int): only keep the most frequent n words in the dictionary
    """
    if isinstance(dictionary, type(None)):
        dictionary = corpora.Dictionary(list_of_doc)
        if len(dictionary) > keep_n:
            dictionary.filter_extremes(no_below=0, no_above=1.1, keep_n=keep_n)
    list_of_doc = list(map(lambda x: dictionary.doc2idx(x), list_of_doc))
    return list_of_doc, dictionary


def add_start_end_token_2_list_token(list_of_list_token):
    """ add <start> and <end> tokens to the start and the end of list_token respectively """
    return list(map(lambda x: ['<start>'] + x + ['<end>'], list_of_list_token))


def add_pad_token_2_list_token(list_of_list_token, max_seq_len):
    """ add multiple <pad> tokens to the tail of the list_token
            so that the length of list_token equal to max_seq_len """
    fix_len_list_of_list_token = []

    for list_token in list_of_list_token:
        after_pad_list_token = list_token + ['<pad>'] * (max_seq_len - len(list_token))
        fix_len_list_of_list_token.append(after_pad_list_token)

    return fix_len_list_of_list_token


def filter_exceed_max_seq_len(list_of_list_token, max_seq_len):
    """ filter sentences which exceed max_seq_len """
    fix_len_list_of_list_token = []
    for list_token in list_of_list_token:
        if len(list_token) <= max_seq_len:
            fix_len_list_of_list_token.append(list_token)
    return fix_len_list_of_list_token


def filter_exceed_max_seq_len_for_cross_lingual(list_of_list_src_token, list_of_list_tar_token,
                                                max_src_seq_len, max_tar_seq_len):
    """ filter sentences which exceed max_seq_len for two languages at the same time """
    fix_len_list_of_list_src_token = []
    fix_len_list_of_list_tar_token = []
    for i, list_src_token in enumerate(list_of_list_src_token):
        list_tar_token = list_of_list_tar_token[i]
        if len(list_src_token) > max_src_seq_len or len(list_tar_token) > max_tar_seq_len:
            continue
        fix_len_list_of_list_src_token.append(list_src_token)
        fix_len_list_of_list_tar_token.append(list_tar_token)

    return fix_len_list_of_list_src_token, fix_len_list_of_list_tar_token


def add_start_end_token_2_string(list_of_sentences):
    """ add <start> <end> token to string """
    return list(map(lambda x: '<start> ' + x + '<end>', list_of_sentences))


def add_start_end_token_idx_2_list_token_idx(list_of_list_token_idx, vocab_size):
    """ add the start token idx (vocab_size) and the end token idx (vocab_size + 1) to list_token_idx """
    return list(map(lambda x: [vocab_size] + x + [vocab_size + 1], list_of_list_token_idx))


def add_pad_token_idx_2_list_token_idx(list_of_list_token_idx, vocab_size, max_seq_len, incr=2):
    """ add the pad token idx (vocab_size + incr) """
    fix_len_list_of_list_token_idx = []
    pad_idx = vocab_size + incr

    for list_token_idx in list_of_list_token_idx:
        after_pad_list_token_idx = list_token_idx + [pad_idx] * (max_seq_len - len(list_token_idx))
        fix_len_list_of_list_token_idx.append(after_pad_list_token_idx)

    return fix_len_list_of_list_token_idx


def remove_out_of_vocab_token_idx(list_of_list_token_idx, vocab_size):
    """ remove the out of vocabulary token idx (idx for <start>, <end>, <pad>) """
    return list(map(lambda x: [v for v in x if v < vocab_size], list_of_list_token_idx))


