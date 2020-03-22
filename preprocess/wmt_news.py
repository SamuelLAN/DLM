import os
from lib.preprocess import utils

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


if __name__ == '__main__':
    def show(name_1, name_2, lan_1_data, lan_2_data):
        print('\nlen of {} data: {}'.format(name_1, len(lan_1_data)))
        print('len of {} data: {}'.format(name_2, len(lan_2_data)))


    show('zh', 'en', *zh_en())
    show('de', 'en', *de_en())
    show('fr', 'en', *fr_en())

    # len of zh data: 19965
    # len of en data: 19965
    #
    # len of de data: 45913
    # len of en data: 45913
    #
    # len of fr data: 25576
    # len of en data: 25576
