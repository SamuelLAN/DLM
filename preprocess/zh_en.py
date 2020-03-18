from lib.preprocess import utils

seg_zh_by_jieba_pipeline = [
    {
        'name': 'tokenize_src_lan',
        'func': utils.zh_word_seg_by_jieba,
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'join_list_token_2_string_with_space_for_src_lan',
        'func': utils.join_list_token_2_string,
        'input_keys': ['input_1', ' '],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
]

remove_zh_space_pipeline = [
    {
        'name': 'remove_space',
        'func': utils.remove_space,
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
        'show_dict': {'lan': 'input_1'},
    },
]

if __name__ == '__main__':
    from preprocess import wmt_news
    from preprocess import tfds_pl

    zh_data, en_data = wmt_news.zh_en()
    params = {
        'src_vocab_size': 2 ** 13,
        'tar_vocab_size': 2 ** 13,
        'max_src_seq_len': 50,
        'max_tar_seq_len': 60,
    }

    print('\n------------------- Encoding -------------------------')
    zh_data, en_data, zh_tokenizer, en_tokenizer = utils.pipeline(
        preprocess_pipeline=seg_zh_by_jieba_pipeline + tfds_pl.train_tokenizer_pipeline + tfds_pl.encode_pipeline,
        lan_data_1=zh_data, lan_data_2=en_data, params=params)

    print('\n----------------------------------------------')
    print(zh_data.shape)
    print(en_data.shape)
    print(zh_tokenizer.vocab_size)
    print(en_tokenizer.vocab_size)

    print('\n------------------- Decoding -------------------------')
    zh_data = utils.pipeline(tfds_pl.decode_pipeline + remove_zh_space_pipeline,
                             zh_data, None, {'tokenizer': zh_tokenizer})

    print('\n------------------- Decoding -------------------------')
    en_data = utils.pipeline(tfds_pl.decode_pipeline, en_data, None, {'tokenizer': en_tokenizer})
