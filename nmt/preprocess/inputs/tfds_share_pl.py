import numpy as np
from lib.preprocess import utils

train_tokenizer = [
    {
        'name': 'combine src and tar input',
        'func': lambda a, b: list(a) + list(b),
        'input_keys': ['input_1', 'input_2'],
        'output_keys': 'input_all',
    },
    {
        'name': 'find max seq len',
        'func': lambda a, b: max(a, b),
        'input_keys': ['max_src_seq_len', 'max_tar_seq_len'],
        'output_keys': 'max_seq_len',
    },
    {
        'name': 'train_subword_tokenizer_by_tfds',
        'func': utils.train_subword_tokenizer_by_tfds,
        'input_keys': ['input_all', 'vocab_size', 'max_seq_len'],
        'output_keys': 'tokenizer',
    },
    {
        'name': 'get_vocab_size',
        'func': lambda x: x.vocab_size,
        'input_keys': ['tokenizer'],
        'output_keys': 'vocab_size',
        'show_dict': {'vocab_size': 'vocab_size'}
    },
    {'output_keys': 'tokenizer'}
]

encode_pipeline = [
    {
        'name': 'update vocab_size',
        'func': lambda a: a.vocab_size,
        'input_keys': ['tokenizer'],
        'output_keys': 'vocab_size',
        'show_dict': {'vocab_size': 'vocab_size', 'src_lan': 'input_1', 'tar_lan': 'input_2'},
    },
    {
        'name': 'encoder_string_2_subword_idx_for_src_lan',
        'func': utils.encoder_string_2_subword_idx_by_tfds,
        'input_keys': ['tokenizer', 'input_1'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'encoder_string_2_subword_idx_for_tar_lan',
        'func': utils.encoder_string_2_subword_idx_by_tfds,
        'input_keys': ['tokenizer', 'input_2'],
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
        'input_keys': ['input_1', 'vocab_size'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'add_start_end_token_to_tar_lan',
        'func': utils.add_start_end_token_idx_2_list_token_idx,
        'input_keys': ['input_2', 'vocab_size'],
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
    {'output_keys': ['input_1', 'input_2', 'tokenizer', 'tokenizer']},
]

encode_pipeline_for_src = [
    {
        'name': 'update vocab_size',
        'func': lambda a: a.vocab_size,
        'input_keys': ['tokenizer'],
        'output_keys': 'vocab_size',
        'show_dict': {'vocab_size': 'vocab_size'},
    },
    {
        'name': 'encoder_string_2_subword_idxn',
        'func': utils.encoder_string_2_subword_idx_by_tfds,
        'input_keys': ['tokenizer', 'input_1'],
        'output_keys': 'input_1',
        'show_dict': {'lan': 'input_1'},
    },
    {
        'name': 'add_start_end_token',
        'func': utils.add_start_end_token_idx_2_list_token_idx,
        'input_keys': ['input_1', 'vocab_size'],
        'output_keys': 'input_1',
        'show_dict': {'lan': 'input_1'},
    },
    {
        'name': 'filter_exceed_max_seq_len',
        'func': utils.filter_exceed_max_seq_len,
        'input_keys': ['input_1', 'max_seq_len'],
        'output_keys': 'input_1',
    },
    {
        'name': 'add_pad_token_to_src_lan',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['input_1', 'max_seq_len'],
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

from pretrain.preprocess.dictionary.map_dict import decode_pos_id, decode_ner_id
from pretrain.preprocess.config import Ids


def decode_subword_idx_2_tokens_by_tfds(_tokenizer, list_of_list_token_idx):
    """
    decode subword_idx to string
    :param
        tokenizer (tfds object): a subword tokenizer built from corpus by tfds
        list_of_list_token_idx (list): [
            [12, 43, 2, 346, 436, 87, 876],   # correspond to ['He', 'llo', ',', 'I', 'am', 'stu', 'dent'],
            [32, 57, 89, 98, 96, 37],         # correspond to ['You', 'are', 'my', 'fri', 'end', 's'],
            ...
        ]
    :return
        list_of_list_token (list): [
            ['He', 'llo', ',', 'I', 'am', 'stu', 'dent'],
            ['You', 'are', 'my', 'fri', 'end', 's'],
            ...
        ]
    """
    # return list(map(lambda x: list(map(lambda a: tokenizer.decode([a]), x)), list_of_list_token_idx))
    return list(map(lambda x: list(map(
        lambda a: _tokenizer.decode([a]) if a <= _tokenizer.vocab_size else (
            decode_pos_id(a, _tokenizer.vocab_size + Ids.end_cdlm_pos_2) + ' ' if decode_pos_id(
                # decode_ner_id(a, _tokenizer.vocab_size + Ids.end_cdlm_ner_2) + ' ' if decode_ner_id(
                #     a, _tokenizer.vocab_size + Ids.end_cdlm_ner_2) else '<spe> '),
                a, _tokenizer.vocab_size + Ids.end_cdlm_pos_2) else '<spe> '),
        x
    )), list_of_list_token_idx))


decode_pipeline = [
    {
        'name': 'get_vocab_size',
        'func': lambda x: x.vocab_size,
        'input_keys': ['tokenizer'],
        'output_keys': 'vocab_size',
    },
    # {
    #     'name': 'remove_out_of_vocab_token_idx',
    #     'func': utils.remove_out_of_vocab_token_idx,
    #     'input_keys': ['input_1', 'vocab_size'],
    #     'output_keys': 'input_1',
    #     'show_dict': {'lan': 'input_1'},
    # },
    {
        'name': 'get_start_end_ids',
        'func': lambda x: [0, x + 1, x + 2] + list(range(x + 5, x + 11)),
        'input_keys': ['vocab_size'],
        'output_keys': 'start_end_ids',
    },
    {
        'name': 'remove_start_end_ids',
        'func': utils.remove_some_token_idx,
        'input_keys': ['input_1', 'start_end_ids'],
        'output_keys': 'input_1',
        'show_dict': {'lan': 'input_1'},
    },
    # {
    #     'name': 'remove_pad_token_idx',
    #     'func': utils.remove_some_token_idx,
    #     'input_keys': ['input_1', [0]],
    #     'output_keys': 'input_1',
    #     'show_dict': {'lan': 'input_1'},
    # },
    # {
    #     'name': 'decode_to_sentences',
    #     'func': lambda tok, x: list(map(lambda a: tok.decode(a), x)),
    #     'input_keys': ['tokenizer', 'input_1'],
    #     'output_keys': 'input_1',
    #     'show_dict': {'lan': 'input_1'},
    # },
    # {
    #     'name': 'decode_to_sentences',
    #     'func': lambda tok, x: list(map(lambda a: tok.decode(a), x)),
    #     'input_keys': ['tokenizer', 'input_1'],
    #     'output_keys': 'input_2',
    #     'show_dict': {'lan': 'input_2'},
    # },
    {
        'name': 'decode_subword_idx_2_tokens_by_tfds',
        'func': decode_subword_idx_2_tokens_by_tfds,
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
    from nmt.preprocess.corpus import wmt_news
    from nmt.preprocess import remove_noise_pipeline
    from nmt.preprocess.inputs.zh_en import seg_zh_by_jieba_pipeline, remove_space_pipeline

    zh_data, en_data = wmt_news.zh_en()
    params = {
        'vocab_size': 2 ** 15,
        'max_src_seq_len': 50,
        'max_tar_seq_len': 60,
    }

    print('\n------------------- Encoding -------------------------')
    zh_data, en_data, tokenizer = utils.pipeline(
        preprocess_pipeline=seg_zh_by_jieba_pipeline + remove_noise_pipeline + train_tokenizer + encode_pipeline,
        lan_data_1=zh_data, lan_data_2=en_data, params=params)

    print('\n----------------------------------------------')
    print(zh_data.shape)
    print(en_data.shape)
    print(tokenizer.vocab_size)

    print('\n------------------- Decoding -------------------------')
    zh_data = utils.pipeline(decode_pipeline + remove_space_pipeline,
                             zh_data, None, {'tokenizer': tokenizer})

    print('\n------------------- Decoding -------------------------')
    en_data = utils.pipeline(decode_pipeline, en_data, None, {'tokenizer': tokenizer})
