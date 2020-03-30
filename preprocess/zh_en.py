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

remove_space_pipeline = [
    {
        'name': 'remove_space',
        'func': utils.remove_space,
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
        'show_dict': {'lan': 'input_1'},
    },
]

seg_char_pipeline = [
    {
        'name': 'char_seg',
        'func': utils.char_seg,
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
    from preprocess import wmt_news, UM_Corpus
    from preprocess import noise_pl, tfds_share_pl
    import numpy as np
    import matplotlib.pyplot as plt

    # origin_zh_data, origin_en_data = wmt_news.zh_en()
    origin_zh_data, origin_en_data = UM_Corpus.zh_en()
    params = {
        'vocab_size': 45000,
        'max_src_seq_len': 79,
        'max_tar_seq_len': 98,
    }

    seg_pipeline = seg_zh_by_jieba_pipeline

    print('\n------------------- Encoding -------------------------')
    zh_data, en_data, zh_tokenizer, en_tokenizer = utils.pipeline(
        preprocess_pipeline=seg_pipeline + noise_pl.remove_noise + tfds_share_pl.train_tokenizer + tfds_share_pl.encode_pipeline,
        lan_data_1=origin_zh_data, lan_data_2=origin_en_data, params=params)

    print('\n----------------------------------------------')
    print(zh_data.shape)
    print(en_data.shape)
    print(zh_tokenizer.vocab_size)
    print(en_tokenizer.vocab_size)

    print('\n------------------- Decoding -------------------------')
    zh_data = utils.pipeline(tfds_share_pl.decode_pipeline + remove_space_pipeline,
                             zh_data, None, {'tokenizer': zh_tokenizer})

    print('\n------------------- Decoding -------------------------')
    en_data = utils.pipeline(tfds_share_pl.decode_pipeline, en_data, None, {'tokenizer': en_tokenizer})

    print('\n------------------- Analyzing -------------------------')
    analyze_pipeline = seg_pipeline + tfds_share_pl.encode_pipeline[1:3] + [{'output_keys': ['input_1', 'input_2']}]
    zh_data, en_data = utils.pipeline(analyze_pipeline, origin_zh_data, origin_en_data, {
        'src_tokenizer': zh_tokenizer, 'tar_tokenizer': en_tokenizer}, verbose=0)

    utils.analyze(zh_data, 'zh')
    utils.analyze(en_data, 'en')

    # mean length of zh: 26.75571467268623
    # max length of zh: 633
    # min length of zh: 1
    # std length of zh: 20.216869776258417
    #
    # mean length of en: 32.43115711060948
    # max length of en: 861
    # min length of en: 5
    # std length of en: 23.08921787547763
