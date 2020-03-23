from lib.preprocess import utils
from preprocess.zh_en import remove_space_pipeline, seg_char_pipeline

seg_jr_by_mecab_pipeline = [
    {
        'name': 'tokenize_src_lan',
        'func': utils.jr_word_seg_by_mecab,
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

if __name__ == '__main__':
    from preprocess import wmt_news
    from preprocess import tfds_pl

    # TODO change this one to jr-en data source
    jr_data, en_data = wmt_news.zh_en()

    params = {
        'src_vocab_size': 2 ** 13,
        'tar_vocab_size': 2 ** 13,
        'max_src_seq_len': 50,
        'max_tar_seq_len': 60,
    }

    print('\n------------------- Encoding -------------------------')
    jr_data, en_data, jr_tokenizer, en_tokenizer = utils.pipeline(
        preprocess_pipeline=seg_jr_by_mecab_pipeline + tfds_pl.train_tokenizer_pipeline + tfds_pl.encode_pipeline,
        lan_data_1=jr_data, lan_data_2=en_data, params=params)

    print('\n----------------------------------------------')
    print(jr_data.shape)
    print(en_data.shape)
    print(jr_tokenizer.vocab_size)
    print(en_tokenizer.vocab_size)

    print('\n------------------- Decoding -------------------------')
    jr_data = utils.pipeline(tfds_pl.decode_pipeline + remove_space_pipeline,
                             jr_data, None, {'tokenizer': jr_tokenizer})

    print('\n------------------- Decoding -------------------------')
    en_data = utils.pipeline(tfds_pl.decode_pipeline, en_data, None, {'tokenizer': en_tokenizer})
