import os
import gzip
from lib.preprocess import utils
from lib.utils import cache, read_cache
from nmt.preprocess.config import data_dir


def get(url, file_name):
    data_path = os.path.join(data_dir, file_name)
    cache_name = os.path.splitext(data_path)[0] + '.pkl'
    if os.path.exists(cache_name):
        return read_cache(cache_name)

    # download and unzip data
    utils.download(url, data_path)
    with gzip.open(data_path, 'rb') as f:
        _data = f.read().decode('utf-8')

    _data = utils.full_2_half(utils.unicode_to_ascii(_data))
    _data = _data.replace('\r', '').strip().split('\n\t\n')
    _data = list(map(
        lambda x: list(map(
            lambda line: [line.split('\t')[1].strip(), line.split('\t')[0].strip()],
            x.split('\n')
        )),
        _data)
    )

    cache(cache_name, _data)
    return _data


def zh_en():
    """
    Return the Chinese-English corpus from News Commentary
    :return
        zh_data (list): list of sentences
        en_data (list): list of sentences
    """
    file_name = 'news-commentary-v15.en-zh.tsv.gz'
    url = 'http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-zh.tsv.gz'
    return get(url, file_name)


if __name__ == '__main__':
    data = zh_en()

    len_doc = len(data)
    len_sent = sum(list(map(lambda x: len(x), data)))

    for _doc in data[:3]:
        print('\n------------------------------------')
        for line in _doc:
            print(line)
        print('\n')

    print(f'len_doc: {len_doc}')
    print(f'len_sent: {len_sent}')

    # len_doc: 7947
    # len_sent: 312766
