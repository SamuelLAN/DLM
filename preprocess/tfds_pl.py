import numpy as np
from lib.preprocess import utils

train_tokenizer_pipeline = [
    {
        'name': 'train_subword_tokenizer_by_tfds_for_src_lan',
        'func': utils.train_subword_tokenizer_by_tfds,
        'input_keys': ['input_1', 'src_vocab_size', 'max_src_seq_len'],
        'output_keys': 'src_tokenizer',
    },
    {
        'name': 'train_subword_tokenizer_by_tfds_for_tar_lan',
        'func': utils.train_subword_tokenizer_by_tfds,
        'input_keys': ['input_2', 'tar_vocab_size', 'max_tar_seq_len'],
        'output_keys': 'tar_tokenizer',
    },
]

encode_pipeline = [
    {
        'name': 'update vocab_size',
        'func': lambda a, b: [a.vocab_size, b.vocab_size],
        'input_keys': ['src_tokenizer', 'tar_tokenizer'],
        'output_keys': ['src_vocab_size', 'tar_vocab_size'],
        'show_dict': {'src_vocab_size': 'src_vocab_size', 'tar_vocab_size': 'tar_vocab_size',
                      'src_lan': 'input_1', 'tar_lan': 'input_2'},
    },
    {
        'name': 'encoder_string_2_subword_idx_for_src_lan',
        'func': utils.encoder_string_2_subword_idx_by_tfds,
        'input_keys': ['src_tokenizer', 'input_1'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'encoder_string_2_subword_idx_for_tar_lan',
        'func': utils.encoder_string_2_subword_idx_by_tfds,
        'input_keys': ['tar_tokenizer', 'input_2'],
        'output_keys': 'input_2',
        'show_dict': {'tar_lan': 'input_2'},
    },
    {
        'name': 'max_seq_len minus 2',
        'func': lambda a, b: [a - 2, b - 2],
        'input_keys': ['max_src_seq_len', 'max_tar_seq_len'],
        'output_keys': ['max_src_seq_len_2', 'max_tar_seq_len_2'],
    },
    {
        'name': 'filter_exceed_max_seq_len',
        'func': utils.filter_exceed_max_seq_len_for_cross_lingual,
        'input_keys': ['input_1', 'input_2', 'max_src_seq_len_2', 'max_tar_seq_len_2'],
        'output_keys': ['input_1', 'input_2'],
    },
    {
        'name': 'add_start_end_token_to_src_lan',
        'func': utils.add_start_end_token_idx_2_list_token_idx,
        'input_keys': ['input_1', 'src_vocab_size'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'add_start_end_token_to_tar_lan',
        'func': utils.add_start_end_token_idx_2_list_token_idx,
        'input_keys': ['input_2', 'tar_vocab_size'],
        'output_keys': 'input_2',
        'show_dict': {'tar_lan': 'input_2'},
    },
    {
        'name': 'add_pad_token_to_src_lan',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['input_1', 'max_src_seq_len'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'add_pad_token_to_tar_lan',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['input_2', 'max_tar_seq_len'],
        'output_keys': 'input_2',
        'show_dict': {'tar_lan': 'input_2'},
    },
    {
        'name': 'convert_input_to_array',
        'func': lambda a, b: [np.array(a), np.array(b)],
        'input_keys': ['input_1', 'input_2'],
        'output_keys': ['input_1', 'input_2'],
    },
    {'output_keys': ['input_1', 'input_2', 'src_tokenizer', 'tar_tokenizer']},
]

encode_pipeline_for_src = [
    {
        'name': 'update vocab_size',
        'func': lambda a: a.vocab_size,
        'input_keys': ['src_tokenizer'],
        'output_keys': 'src_vocab_size',
        'show_dict': {'src_vocab_size': 'src_vocab_size'},
    },
    {
        'name': 'encoder_string_2_subword_idx_for_src_lan',
        'func': utils.encoder_string_2_subword_idx_by_tfds,
        'input_keys': ['src_tokenizer', 'input_1'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'add_start_end_token_to_src_lan',
        'func': utils.add_start_end_token_idx_2_list_token_idx,
        'input_keys': ['input_1', 'src_vocab_size'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'filter_exceed_max_seq_len',
        'func': utils.filter_exceed_max_seq_len,
        'input_keys': ['input_1', 'max_src_seq_len'],
        'output_keys': 'input_1',
    },
    {
        'name': 'add_pad_token_to_src_lan',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['input_1', 'max_src_seq_len'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'convert_input_to_array',
        'func': lambda a: np.array(a),
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
    },
]

decode_pipeline = [
    {
        'name': 'get_vocab_size',
        'func': lambda x: x.vocab_size,
        'input_keys': ['tokenizer'],
        'output_keys': 'vocab_size',
    },
    {
        'name': 'remove_out_of_vocab_token_idx',
        'func': utils.remove_out_of_vocab_token_idx,
        'input_keys': ['input_1', 'vocab_size'],
        'output_keys': 'input_1',
        'show_dict': {'lan': 'input_1'},
    },
    {
        'name': 'remove_out_of_vocab_token_idx',
        'func': utils.remove_some_token_idx,
        'input_keys': ['input_1', [0]],
        'output_keys': 'input_1',
        'show_dict': {'lan': 'input_1'},
    },
    {
        'name': 'decode_to_sentences',
        'func': lambda tok, x: list(map(lambda a: tok.decode(a), x)),
        'input_keys': ['tokenizer', 'input_1'],
        'output_keys': 'input_2',
        'show_dict': {'lan': 'input_2'},
    },
    {
        'name': 'decode_subword_idx_2_tokens_by_tfds',
        'func': utils.decode_subword_idx_2_tokens_by_tfds,
        'input_keys': ['tokenizer', 'input_1'],
        'output_keys': 'input_1',
        'show_dict': {'lan': 'input_1'},
    },
    {
        'name': 'join_list_token_2_string',
        'func': utils.join_list_token_2_string,
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
        'show_dict': {'lan': 'input_1'},
    },
]

if __name__ == '__main__':
    from preprocess.corpus import wmt_news
    from preprocess.zh_en import seg_zh_by_jieba_pipeline, remove_space_pipeline

    zh_data, en_data = wmt_news.zh_en()
    params = {
        'src_vocab_size': 2 ** 13,
        'tar_vocab_size': 2 ** 13,
        'max_src_seq_len': 50,
        'max_tar_seq_len': 60,
    }

    print('\n------------------- Encoding -------------------------')
    zh_data, en_data, zh_tokenizer, en_tokenizer = utils.pipeline(
        preprocess_pipeline=seg_zh_by_jieba_pipeline + train_tokenizer_pipeline + encode_pipeline,
        lan_data_1=zh_data, lan_data_2=en_data, params=params)

    print('\n----------------------------------------------')
    print(zh_data.shape)
    print(en_data.shape)
    print(zh_tokenizer.vocab_size)
    print(en_tokenizer.vocab_size)

    print('\n------------------- Decoding -------------------------')
    zh_data = utils.pipeline(decode_pipeline + remove_space_pipeline,
                             zh_data, None, {'tokenizer': zh_tokenizer})

    print('\n------------------- Decoding -------------------------')
    en_data = utils.pipeline(decode_pipeline, en_data, None, {'tokenizer': en_tokenizer})
