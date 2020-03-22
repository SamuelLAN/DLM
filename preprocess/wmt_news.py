import os
from lib.preprocess import utils
import shutil

__data_dir = 'D:\Data\DLM\data'


def get(url, file_name, lan_1_name, lan_2_name):
    wmt_news_path = os.path.join(__data_dir, file_name)
    wmt_news_dir = os.path.splitext(wmt_news_path)[0]
    lan_1_file_path = os.path.join(wmt_news_dir, lan_1_name)
    lan_2_file_path = os.path.join(wmt_news_dir, lan_2_name)

    # download and unzip data
    utils.download(url, wmt_news_path)
    utils.unzip_and_delete(wmt_news_path)

    # read data
    lan_1_data = utils.read_lines(lan_1_file_path)
    lan_2_data = utils.read_lines(lan_2_file_path)
    return lan_1_data, lan_2_data

def get_europarl(url, file_name, lan_1_name, lan_2_name):
    europarl_news_path = os.path.join(__data_dir, file_name)
    europarl_news_dir = os.path.splitext(europarl_news_path)[0]
    lan_1_file_path = os.path.join(europarl_news_dir, lan_1_name)
    lan_2_file_path = os.path.join(europarl_news_dir, lan_2_name)

    utils.download(url, europarl_news_path)
    shutil.unpack_archive(europarl_news_path, europarl_news_dir)
    os.remove(europarl_news_path)
   
    # read data
    lan_1_data = utils.read_lines(lan_1_file_path)
    lan_2_data = utils.read_lines(lan_2_file_path)
    return lan_1_data, lan_2_data


def fr_en_europarl():
    file_name = 'europarl_v7_fr-en.tgz'
    url = 'http://www.statmt.org/europarl/v7/fr-en.tgz'
    return get_europarl(url, file_name, 'europarl-v7.fr-en.fr', 'europarl-v7.fr-en.en')

def de_en_europarl():
    file_name = 'europarl_v7_de-en.tgz'
    url = 'http://www.statmt.org/europarl/v7/de-en.tgz'
    return get_europarl(url, file_name, 'europarl-v7.de-en.fr', 'europarl-v7.de-en.en')




def zh_en():
    """
    Return the Chinese-English corpus from WMT-news
    :return
        zh_data (list): list of sentences
        en_data (list): list of sentences
    """
    file_name = 'wmt_news_2019_zh_en.zip'
    url = 'http://opus.nlpl.eu/download.php?f=WMT-News/v2019/moses/en-zh.txt.zip'
    return get(url, file_name, 'WMT-News.en-zh.zh', 'WMT-News.en-zh.en')


def de_en():
    """
    Return the Chinese-English corpus from WMT-news
    :return
        zh_data (list): list of sentences
        en_data (list): list of sentences
    """
    file_name = 'wmt_news_2019_de_en.zip'
    url = 'http://opus.nlpl.eu/download.php?f=WMT-News/v2019/moses/de-en.txt.zip'
    return get(url, file_name, 'WMT-News.de-en.de', 'WMT-News.de-en.en')


def fr_en():
    """
    Return the Chinese-English corpus from WMT-news
    :return
        zh_data (list): list of sentences
        en_data (list): list of sentences
    """
    file_name = 'wmt_news_2019_fr_en.zip'
    url = 'http://opus.nlpl.eu/download.php?f=WMT-News/v2019/moses/en-fr.txt.zip'
    return get(url, file_name, 'WMT-News.en-fr.fr', 'WMT-News.en-fr.en')


def get_europarl(url, file_name, lan_1_name, lan_2_name):



# lan_1_data, lan_2_data = zh_en()
#
# print('\nlen of en data: %d' % len(lan_2_data))
# print('len of de data: %d' % len(lan_1_data))
#
# print('\nlearning to tokenize ...')
#
# seg_en_data = utils.en_word_seg_by_nltk(en_data)
# seg_zh_data = utils.zh_word_seg_by_jieba(zh_data)
# # seg_zh_data = utils.zh_word_seg_by_pku(zh_data)
#
# tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     en_data,
#     target_vocab_size=2 ** 13
# )
# # tokenizer_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus(
# #     zh_data,
# #     target_vocab_size=2 ** 13
# # )
#
# print('finish learning \n')
#
# # show data examples
# print('Examples:')
#
# for i in range(len(en_data[:10])):
#     en = en_data[i]
#     zh = zh_data[i]
#
#     tokenized_string_en = tokenizer_en.encode(en)
#     original_string_en = tokenizer_en.decode(tokenized_string_en)
#     tokens_en = [tokenizer_en.decode([v]) for v in tokenized_string_en]
#
#     # tokenized_string_zh = tokenizer_zh.encode(zh)
#     # original_string_zh = tokenizer_zh.decode(tokenized_string_zh)
#     # tokens_zh = [tokenizer_zh.decode([v]) for v in tokenized_string_zh]
#
#     print('\n------------------ %d ---------------------' % i)
#     print(en)
#     print(seg_en_data[i])
#     print(tokenized_string_en)
#     print(original_string_en)
#     print(tokens_en)
#     print('')
#     print(zh)
#     print(seg_zh_data[i])
#     # print(tokenized_string_zh)
#     # print(original_string_zh)
#     # print(tokens_zh)
#     print('----------------------------------\n')
#
# print('\ndone')
