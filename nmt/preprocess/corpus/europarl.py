import os
from lib.preprocess import utils
import shutil
from nmt.preprocess.config import data_dir


def get(url, file_name, lan_1_name, lan_2_name):
    europarl_news_path = os.path.join(data_dir, file_name)
    europarl_news_dir = os.path.splitext(europarl_news_path)[0]
    lan_1_file_path = os.path.join(europarl_news_dir, lan_1_name)
    lan_2_file_path = os.path.join(europarl_news_dir, lan_2_name)

    utils.download(url, europarl_news_path)
    if os.path.isfile(europarl_news_path):
        shutil.unpack_archive(europarl_news_path, europarl_news_dir)
        os.remove(europarl_news_path)

    # read data
    lan_1_data = utils.read_lines(lan_1_file_path)
    lan_2_data = utils.read_lines(lan_2_file_path)
    return lan_1_data, lan_2_data


def fr_en():
    file_name = 'europarl_v7_fr-en.tgz'
    url = 'http://www.statmt.org/europarl/v7/fr-en.tgz'
    return get(url, file_name, 'europarl-v7.fr-en.fr', 'europarl-v7.fr-en.en')


def de_en():
    file_name = 'europarl_v7_de-en.tgz'
    url = 'http://www.statmt.org/europarl/v7/de-en.tgz'
    return get(url, file_name, 'europarl-v7.de-en.de', 'europarl-v7.de-en.en')


def ro_en():
    file_name = 'europarl_v8_en-ro.tgz'
    url = 'http://data.statmt.org/wmt16/translation-task/training-parallel-ep-v8.tgz'
    return get(url, file_name, 'training-parallel-ep-v8/europarl-v8.ro-en.ro',
               'training-parallel-ep-v8/europarl-v8.ro-en.en')


if __name__ == '__main__':
    def show(name_1, name_2, lan_1_data, lan_2_data):
        print('\nlen of {} data: {}'.format(name_1, len(lan_1_data)))
        print('len of {} data: {}'.format(name_2, len(lan_2_data)))


    show('ro', 'en', *ro_en())
    # show('fr', 'en', *fr_en())
    # show('fr', 'en', *fr_en())

    # len of ro data: 399375
    # len of en data: 399375

    # len of de data: 45913
    # len of en data: 45913
    #
    # len of fr data: 25576
    # len of en data: 25576
