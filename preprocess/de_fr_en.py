from lib.preprocess import utils

if __name__ == '__main__':
    from preprocess import wmt_news
    from preprocess import tfds_pl

    de_data, en_data = wmt_news.de_en()
    # de_data, en_data = wmt_news.fr_en()

    params = {
        'src_vocab_size': 2 ** 13,
        'tar_vocab_size': 2 ** 13,
        'max_src_seq_len': 50,
        'max_tar_seq_len': 60,
    }

    print('\n------------------- Encoding -------------------------')
    de_data, en_data, de_tokenizer, en_tokenizer = utils.pipeline(
        preprocess_pipeline=tfds_pl.train_tokenizer_pipeline + tfds_pl.encode_pipeline,
        lan_data_1=de_data, lan_data_2=en_data, params=params)

    print('\n----------------------------------------------')
    print(de_data.shape)
    print(en_data.shape)
    print(de_tokenizer.vocab_size)
    print(en_tokenizer.vocab_size)

    print('\n------------------- Decoding -------------------------')
    de_data = utils.pipeline(tfds_pl.decode_pipeline,
                             de_data, None, {'tokenizer': de_tokenizer})

    print('\n------------------- Decoding -------------------------')
    en_data = utils.pipeline(tfds_pl.decode_pipeline, en_data, None, {'tokenizer': en_tokenizer})
