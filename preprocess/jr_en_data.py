from lib.preprocess import utils
import os
import shutil
import glob

__data_dir = r'D:\Data\DLM\data'


def get(url, file_name):
    kyoto_news_path = os.path.join(__data_dir, file_name)
    kyoto_news_dir = os.path.splitext(kyoto_news_path)[0]

    if not os.path.exists(kyoto_news_dir):
        utils.download(url, kyoto_news_path)
        shutil.unpack_archive(kyoto_news_path, kyoto_news_dir)
        os.remove(kyoto_news_path)
        print("Download successfully, start extracting the training data.")

    # Iterate over all the entries
    train_files = glob.glob(kyoto_news_dir + "/**/*-train.ja", recursive=True)
    test_files = glob.glob(kyoto_news_dir + "/**/*-train.en", recursive=True)

    lan_1_data = utils.read_lines(train_files[0])
    lan_2_data = utils.read_lines(test_files[0])

    print("Done! going to the tokenizer next.")

    return lan_1_data, lan_2_data


def jr_en():
    """
    Return the Chinese-English corpus from WMT-news
    :return
        zh_data (list): list of sentences
        en_data (list): list of sentences
    """
    file_name = 'kftt-data-1.0.tar.gz'
    url = 'http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz'
    return get(url, file_name)


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


    jr_data, en_data = jr_en()
    len_data = len(en_data)
    sample_rate = 0.22
    end_index = int(len_data * sample_rate)
    jr_data = jr_data[:end_index]
    en_data = en_data[:end_index]
    show('jr', 'en', jr_data, en_data)
    print('English words num: {}'.format(stat_en_words(en_data)))

    # len of jr data: 96863
    # len of en data: 96863
    # English words num: 2053752
