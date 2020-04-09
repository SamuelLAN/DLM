from lib.preprocess import utils
from preprocess.nmt_inputs.zh_en import remove_space_pipeline

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
    from preprocess.corpus import KFTT
    from preprocess.nmt_inputs import tfds_share_pl

    # TODO change this one to jr-en data source
    origin_jr_data, origin_en_data = KFTT.jr_en()

    params = {
        'vocab_size': 2 ** 13,
        'max_src_seq_len': 50,
        'max_tar_seq_len': 60,
    }

    seg_pipeline = seg_jr_by_mecab_pipeline

    print('\n------------------- Encoding -------------------------')
    jr_data, en_data, jr_tokenizer, en_tokenizer = utils.pipeline(
        preprocess_pipeline=seg_pipeline + tfds_share_pl.train_tokenizer + tfds_share_pl.encode_pipeline,
        lan_data_1=origin_jr_data, lan_data_2=origin_en_data, params=params)

    print('\n----------------------------------------------')
    print(jr_data.shape)
    print(en_data.shape)
    print(jr_tokenizer.vocab_size)
    print(en_tokenizer.vocab_size)

    print('\n------------------- Decoding -------------------------')
    jr_data = utils.pipeline(tfds_share_pl.decode_pipeline + remove_space_pipeline,
                             jr_data, None, {'tokenizer': jr_tokenizer})

    print('\n------------------- Decoding -------------------------')
    en_data = utils.pipeline(tfds_share_pl.decode_pipeline, en_data, None, {'tokenizer': en_tokenizer})

    print('\n------------------- Analyzing -------------------------')
    analyze_pipeline = seg_pipeline + tfds_share_pl.encode_pipeline[1:3] + [{'output_keys': ['input_1', 'input_2']}]
    jr_data, en_data = utils.pipeline(analyze_pipeline, origin_jr_data, origin_en_data, {
        'src_tokenizer': jr_tokenizer, 'tar_tokenizer': en_tokenizer}, verbose=0)

    utils.analyze(jr_data, 'jr')
    utils.analyze(en_data, 'en')
