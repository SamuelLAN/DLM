import os
import shutil
from lib.preprocess import utils
from preprocess.config import data_dir


def get(file_name, domain='*', get_test=False):
    domain = domain.lower()
    um_path = os.path.join(data_dir, file_name)
    um_dir = os.path.splitext(um_path)[0]

    # unzip data
    if not os.path.exists(um_dir):
        shutil.unpack_archive(um_path, um_dir)
        os.remove(um_path)

    dir_name = 'Bilingual' if not get_test else 'Testing'
    um_file_dir = os.path.join(um_dir, 'UM-Corpus', 'data', dir_name)
    um_file_dir_files = os.listdir(um_file_dir)

    # get domain list
    if not get_test:
        domain_list = [str(v.split('.')[0]).lower().split('-')[1] for v in um_file_dir_files if
                       v.split('.')[1].lower() == 'txt']
        assert domain == '*' or domain in domain_list

    # traverse directory to get data
    lan_1_data, lan_2_data = [], []
    for file_name in um_file_dir_files:
        # continue if the file is not data
        if os.path.splitext(file_name)[1].lower() != '.txt':
            continue

        # check domain
        if not get_test:
            tmp_domain = str(file_name.split('.')[0]).lower().split('-')[1]
            if domain != '*' and domain != tmp_domain:
                continue

        # read and add data
        lines = utils.read_lines(os.path.join(um_file_dir, file_name))
        lan_1_data.extend(lines[1::2])
        lan_2_data.extend(lines[::2])

    return lan_1_data, lan_2_data


def zh_en(domain='*', get_test=False):
    """
    Return the Chinese-English corpus from WMT-news
    :return
        zh_data (list): list of sentences
        en_data (list): list of sentences
    """
    file_name = 'umcorpus-v1.zip'
    return get(file_name, domain, get_test)


if __name__ == '__main__':
    def show(name_1, name_2, lan_1_data, lan_2_data):
        print('\nlen of {} data: {}'.format(name_1, len(lan_1_data)))
        print('len of {} data: {}'.format(name_2, len(lan_2_data)))

        for i, v in enumerate(lan_1_data[:5]):
            print('\n------------- {} ---------------'.format(i))
            print(v)
            print(lan_2_data[i])

        for i in range(-1, -6, -1):
            print('\n------------- {} ---------------'.format(i))
            print(lan_1_data[i])
            print(lan_2_data[i])


    zh_data, en_data = zh_en('news')
    len_data = len(en_data)
    sample_rate = 0.05
    end_index = int(len_data * sample_rate)
    zh_data = zh_data[:end_index]
    en_data = en_data[:end_index]
    show('zh', 'en', zh_data, en_data)

    print('English words num: {}'.format(utils.stat_en_words(en_data)))
