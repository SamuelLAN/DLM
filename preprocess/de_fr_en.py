from lib.preprocess import utils

if __name__ == '__main__':
    from preprocess.corpus import wmt_news
    from preprocess import tfds_share_pl, noise_pl

    # origin_de_data, origin_en_data = wmt_news.de_en()
    origin_de_data, origin_en_data = wmt_news.fr_en()

    params = {
        'vocab_size': 40000,
        'src_vocab_size': 2 ** 13,
        'tar_vocab_size': 2 ** 13,
        'max_src_seq_len': 50,
        'max_tar_seq_len': 60,
    }

    print('\n------------------- Encoding -------------------------')
    de_data, en_data, de_tokenizer, en_tokenizer = utils.pipeline(
        preprocess_pipeline=noise_pl.remove_noise + tfds_share_pl.train_tokenizer + tfds_share_pl.encode_pipeline,
        lan_data_1=origin_de_data, lan_data_2=origin_en_data, params=params)

    print('\n----------------------------------------------')
    print(de_data.shape)
    print(en_data.shape)
    print(de_tokenizer.vocab_size)
    print(en_tokenizer.vocab_size)

    print('\n------------------- Decoding -------------------------')
    de_data = utils.pipeline(tfds_share_pl.decode_pipeline,
                             de_data, None, {'tokenizer': de_tokenizer})

    print('\n------------------- Decoding -------------------------')
    en_data = utils.pipeline(tfds_share_pl.decode_pipeline, en_data, None, {'tokenizer': en_tokenizer})

    print('\n------------------- Analyzing -------------------------')
    analyze_pipeline = noise_pl.remove_noise + tfds_share_pl.encode_pipeline[1:3] + [{'output_keys': ['input_1', 'input_2']}]
    de_data, en_data = utils.pipeline(analyze_pipeline, origin_de_data, origin_en_data, {
        'tokenizer': de_tokenizer}, verbose=0)

    utils.analyze(de_data, 'de')
    utils.analyze(en_data, 'en')
