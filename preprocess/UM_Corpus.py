import os
import shutil

__data_dir = r'D:\Data\DLM\data'


def get(file_name):
    um_path = os.path.join(__data_dir, file_name)
    um_dir = os.path.splitext(um_path)[0]
    shutil.unpack_archive(um_path, um_dir)
    um_file_dir = os.path.join(um_dir, 'UM-Corpus', 'data', 'Bilingual')
    um_file_dir_files = os.listdir(um_file_dir)
    lan_1_data, lan_2_data = [], []
    for file in um_file_dir_files:
        with open(os.path.join(um_file_dir, file), encoding="utf8", errors='ignore') as f:
            lines = f.readlines()
            lan_1_data.extend(lines[1::2])
            lan_2_data.extend(lines[::2])
    lan_1_data = list(map(str.strip, lan_1_data))
    lan_2_data = list(map(str.strip, lan_2_data))[1:]
    return lan_1_data, lan_2_data


def zh_en():
    """
    Return the Chinese-English corpus from WMT-news
    :return
        zh_data (list): list of sentences
        en_data (list): list of sentences
    """
    file_name = 'umcorpus-v1.zip'
    return get(file_name)


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


    def stat_en_words(_en_data):
        return sum(list(map(lambda x: len(x.split(' ')), _en_data)))

    zh_data, en_data = zh_en()
    len_data = len(en_data)
    sample_rate = 0.05
    end_index = int(len_data * sample_rate)
    zh_data = zh_data[:end_index]
    en_data = en_data[:end_index]
    show('zh', 'en', zh_data, en_data)
    print('English words num: {}'.format(stat_en_words(en_data)))
