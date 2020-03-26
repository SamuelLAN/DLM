from lib.preprocess import utils
import os
import shutil
import glob

__data_dir = "./"


def get(url, file_name):
    kyoto_news_path = os.path.join(__data_dir, file_name)
    kyoto_news_dir = os.path.splitext(kyoto_news_path)[0]

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
