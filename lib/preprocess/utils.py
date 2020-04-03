import os
import re
import jieba
import pkuseg
import chardet
import zipfile
import unicodedata
# import MeCab
import tinysegmenter
import numpy as np
from gensim import corpora
import tensorflow_datasets as tfds
from six.moves.urllib.request import urlretrieve
from nltk.tokenize import word_tokenize

TOKEN_START = '<start>'
TOKEN_END = '<end>'
TOKEN_CLS = '<cls>'
TOKEN_PAD = '<pad>'
TOKEN_UNK = '<unk>'  # for unknown words


def download(url, file_path):
    """ download data """
    dir_path = os.path.splitext(file_path)[0]
    if os.path.exists(dir_path) or os.path.exists(file_path):
        # print('%s exists' % file_path)
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
        # print('%s has been unzip' % file_path)
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


def jr_word_seg_by_mecab(list_of_sentences):
    """ Tokenize japanese words by mecab """
    segmenter = tinysegmenter.TinySegmenter()
    return list(map(lambda x: list(segmenter.tokenize(x)), list_of_sentences))


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


def idx_2_doc(list_of_list_token_idx, dictionary):
    """ decode list_of_list_token_idx to list_of_list_token """
    return list(map(
        lambda x: list(map(lambda a: dictionary.get(a) if dictionary.get(a) else TOKEN_UNK, x)),
        list_of_list_token_idx
    ))


def join_list_token_2_string(list_of_list_token, delimiter=''):
    """ join the list tokens to string """
    return list(map(lambda x: delimiter.join(x), list_of_list_token))


def remove_space(list_of_sentence):
    return list(map(lambda x: x.replace(' ', ''), list_of_sentence))


def add_start_end_token_2_list_token(list_of_list_token):
    """ add <start> and <end> tokens to the start and the end of list_token respectively """
    return list(map(lambda x: [TOKEN_START] + x + [TOKEN_END], list_of_list_token))


def add_pad_token_2_list_token(list_of_list_token, max_seq_len):
    """ add multiple <pad> tokens to the tail of the list_token
            so that the length of list_token equal to max_seq_len """
    fix_len_list_of_list_token = []

    for list_token in list_of_list_token:
        after_pad_list_token = list_token + [TOKEN_PAD] * (max_seq_len - len(list_token))
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
    return list(map(lambda x: TOKEN_START + ' ' + x + ' ' + TOKEN_END, list_of_sentences))


def add_start_end_token_idx_2_list_token_idx(list_of_list_token_idx, vocab_size, incr=0):
    """ add the start token idx (vocab_size + 1) and the end token idx (vocab_size + 2) to list_token_idx """
    return list(map(lambda x: [vocab_size + 1 + incr] + x + [vocab_size + incr + 2], list_of_list_token_idx))


def add_pad_token_idx_2_list_token_idx(list_of_list_token_idx, max_seq_len):
    """ add the pad token idx (0) """
    fix_len_list_of_list_token_idx = []
    pad_idx = 0

    for list_token_idx in list_of_list_token_idx:
        after_pad_list_token_idx = list_token_idx + [pad_idx] * (max_seq_len - len(list_token_idx))
        fix_len_list_of_list_token_idx.append(after_pad_list_token_idx)

    return fix_len_list_of_list_token_idx


def remove_out_of_vocab_token_idx(list_of_list_token_idx, vocab_size):
    """ remove the out of vocabulary token idx (idx for <start>, <end>, <pad>) """
    return list(map(lambda x: [v for v in x if v < vocab_size], list_of_list_token_idx))


def remove_some_token_idx(list_of_list_token_idx, remove_idx_list):
    """ remove the out of vocabulary token idx (idx for <start>, <end>, <pad>) """
    return list(map(lambda x: [v for v in x if v not in remove_idx_list], list_of_list_token_idx))


def convert_minus_1_to_unknown_token_idx(list_of_list_token_idx, vocab_size, incr=0):
    """ convert the -1 to vocab_size + incr (because for the unknown words, the dictionary may convert it to -1) """
    return list(map(lambda x: list(map(lambda a: a if a != -1 else (vocab_size + incr), x)), list_of_list_token_idx))


def convert_list_of_list_token_idx_2_string(list_of_list_token_idx):
    return list(map(lambda x: list(map(str, x)), list_of_list_token_idx))


def pipeline(preprocess_pipeline, lan_data_1, lan_data_2=None, params={}, verbose=True):
    """
    preprocess the data according to the preprocess_pipeline
    :params
        preprocess_pipeline (list): a list of preprocessing functions
            the format must be: [
                {
                    'name': 'add_pad_token_idx_to_en', # whatever you name it, just for display
                    'func': utils.add_pad_token_idx_2_list_token_idx, # the func that will be executed
                    'input_keys': ['input_2', 'en_vocab_size', 'max_tar_seq_len', 0],
                        # generate the args for func according to the input_keys;
                    'output_keys': ['input_2'],
                        # record the output of the func to the result_dict according to the output_keys
                },
                ...
            ]
        lan_data_1 (list): list of sentences of language 1
            e.g., [ 'I am a boy', 'you are a girl.' ]
        lan_data_2 (list):  list of sentences of language 2
            e.g., [ 'I am a boy', 'you are a girl.' ]
        params (dict): pass parameters that functions in the pipeline may need
        verbose (bool): whether or not to print information
    """
    # share variables when applying different preprocess functions
    result_dict = {**params, 'input_1': lan_data_1, 'input_2': lan_data_2}

    # traverse the pipeline
    for func_dict in preprocess_pipeline:
        # for the last func of the pipeline; the last func would only contains the 'output_keys' for the return values
        if 'func' not in func_dict:
            continue

        if 'params' in func_dict:
            for k, v in func_dict['params'].items():
                result_dict[k] = v

        # get variables
        name = func_dict['name']
        func = func_dict['func']
        args = [result_dict[key] if isinstance(key, str) and key in result_dict else key
                for key in func_dict['input_keys']]
        output_keys = func_dict['output_keys']
        show_dict = {} if 'show_dict' not in func_dict else func_dict['show_dict']

        # apply preprocess function
        if verbose:
            print('preprocessing %s ...' % name)
        outputs = func(*args)

        # record output to result_dict
        if isinstance(output_keys, str):
            result_dict[output_keys] = outputs
        elif len(output_keys) == 1:
            result_dict[output_keys[0]] = outputs
        else:
            for i, key in enumerate(output_keys):
                result_dict[key] = outputs[i]

        # for display
        if verbose:
            for k, v in show_dict.items():
                v = result_dict[v]
                tmp_v = v[:2] if isinstance(v, list) or isinstance(v, np.ndarray) or isinstance(v, tuple) else v
                print('{}: {}'.format(k, tmp_v))

    # return output according to the last element's output_keys
    last_output_keys = preprocess_pipeline[-1]['output_keys']
    if isinstance(last_output_keys, str):
        return result_dict[last_output_keys]
    elif len(last_output_keys) == 1:
        return result_dict[last_output_keys]
    return [result_dict[key] for key in last_output_keys]


def analyze(lan_data, lan_name, bin_size=50):
    import numpy as np
    import matplotlib.pyplot as plt

    len_list = list(map(len, lan_data))
    print('\nmean length of {}: {}\nmax length of {}: {}\n'
          'min length of {}: {}\nstd length of {}: {}\n'.format(
        lan_name, np.mean(len_list), lan_name, np.max(len_list),
        lan_name, np.min(len_list), lan_name, np.std(len_list)))

    plt.hist(len_list, bins=bin_size, edgecolor='#E6E6E6')
    plt.title('histogram of length of {}'.format(lan_name))
    plt.xlabel('length (num of tokens in a sentence)')
    plt.ylabel('size')
    plt.grid(linestyle='dashed')
    plt.show()


def combine_multi_space(list_of_sentences):
    reg = re.compile(r'\s+')
    return list(map(lambda x: reg.sub(' ', x), list_of_sentences))


__reg_delimiter = re.compile(r'([?!,;:])')
__reg_spot = re.compile(r'(\.)(?!\d)')
__reg_split = re.compile('["“”‘’ ><\[\]{}《》【】（）()]+')
__reg_space = re.compile(r"[^\da-zA-Z?.!,_\-;:'\u4e00-\u9fa5\u30a0-\u30ff\u3040-\u309f\u3000-\u303f\ufb00-\ufffd]+")
__reg_num_space = re.compile(r'(\d+)\s+(\d+)')
__reg_num_spot = re.compile(r'(\d+)\s+\.\s+(\d+)')


def remove_special_chars(string):
    # convert chinese punctuations to english punctuations
    string = string.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?'). \
        replace('：', ':').replace('；', ';').replace(';', '.')

    # insert space to the front of the delimiter
    string = __reg_delimiter.sub(r' \1 ', string)
    string = __reg_spot.sub(r' \1 ', string)

    # concat noise numbers
    tmp_string = __reg_num_space.sub(r'\1\2', string)
    while tmp_string != string:
        string = tmp_string
        tmp_string = __reg_num_space.sub(r'\1\2', string)
    string = __reg_num_spot.sub(r'\1.\2', string)

    # replace some special chars to space
    string = __reg_split.sub(' ', string)

    # replace everything except normal chars to space
    string = __reg_space.sub(' ', string)

    string = string.strip()

    # if end punctuation is , or ;
    if string and string[-1] in [',', ';']:
        string = string[:-1] + '.'

    # if no end punctuations, add one
    if string and string[-1] not in ['.', ',', '?', '!', ';']:
        string += '.'
    return string


def remove_noise_for_sentences(list_of_sentences):
    return list(map(remove_special_chars, list_of_sentences))


def stat_en_words(en_sentences):
    return sum(list(map(lambda x: len(x.split(' ')), en_sentences)))


__reg_sent_delimiter = re.compile(r'[.!?。！？;；](?!\d)')
__reg_num = re.compile('^\d+$')
__reg_num_end = re.compile('\d+$')


def split_sentences(src_sentences, tar_sentences):
    sentences = list(zip(src_sentences, tar_sentences))
    new_sentences = []
    for index, (src_sent, tar_sent) in enumerate(sentences):
        src_l = __reg_sent_delimiter.split(src_sent)
        tar_l = __reg_sent_delimiter.split(tar_sent)

        if not src_l or not tar_l:
            continue

        if not src_l[-1].strip():
            src_l = src_l[:-1]

        if not tar_l[-1].strip():
            tar_l = tar_l[:-1]

        src_delimiters = __reg_sent_delimiter.findall(src_sent)
        tar_delimiters = __reg_sent_delimiter.findall(tar_sent)

        if len(src_l) != len(tar_l):
            if len(src_delimiters) == 1:
                tar_sent = __reg_sent_delimiter.sub(',', tar_sent, count=len(tar_delimiters) - 1)
                sentences[index] = (src_sent, tar_sent)
            if len(tar_delimiters) == 1:
                src_sent = __reg_sent_delimiter.sub(',', src_sent, count=len(src_delimiters) - 1)
                sentences[index] = (src_sent, tar_sent)
            continue

        if len(src_l) <= 1:
            continue

        cur = len(src_l) - 1
        while cur > 0:
            if __reg_num.search(src_l[cur]) and __reg_num.search(tar_l[cur]) and \
                    __reg_num_end.search(src_l[cur - 1]) and __reg_num_end.search(tar_l[cur - 1]) and \
                    src_delimiters[cur - 1] == '.':
                src_l[cur - 1] += '.' + src_l[cur]
                tar_l[cur - 1] += '.' + tar_l[cur]
                del src_delimiters[cur - 1]
                cur -= 1
            cur -= 1

        if len(src_l) <= 1 or abs(len(src_l[0].split(' ')) - len(tar_l[0].split(' '))) > 5:
            continue

        new_sentences += [(src_l[i] + src_delimiters[i], tar_l[i] + src_delimiters[i])
                          for i in range(len(src_l)) if i > 0]
        sentences[index] = (src_l[0] + src_delimiters[0], tar_l[0] + src_delimiters[0])

    sentences += new_sentences
    src_sentences, tar_sentences = list(zip(*sentences))

    return list(src_sentences), list(tar_sentences)


def lower_sentences(list_of_sentences):
    return list(map(lambda x: x.lower(), list_of_sentences))
