from nmt.preprocess.corpus.wmt_news import get


def ro_en():
    """
    Return the Romanian-English corpus from SETIMES
    :return
        ro_data (list): list of sentences
        en_data (list): list of sentences
    """
    file_name = 'en-ro.txt.zip'
    url = 'http://opus.nlpl.eu/download.php?f=SETIMES/v2/moses/en-ro.txt.zip'
    return get(url, file_name, 'SETIMES.en-ro.ro', 'SETIMES.en-ro.en')


if __name__ == '__main__':
    def show(name_1, name_2, lan_1_data, lan_2_data):
        print('\nlen of {} data: {}'.format(name_1, len(lan_1_data)))
        print('len of {} data: {}'.format(name_2, len(lan_2_data)))

    show('ro', 'en', *ro_en())


# len of ro data: 213047
# len of en data: 213047
