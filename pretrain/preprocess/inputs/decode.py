import numpy as np
from lib.preprocess import utils
from pretrain.preprocess.dictionary.map_dict import decode_pos_id, decode_ner_id
from pretrain.preprocess.config import Ids


def decode_one_idx(x, _tokenizer, mode=''):
    v_size = _tokenizer.vocab_size
    if x <= v_size:
        return _tokenizer.decode([x])
    else:
        tok = '<sep>'
        if mode == 'pos':
            tok = decode_pos_id(x, v_size + Ids.offset_pos)

        elif mode == 'ner':
            tok = decode_ner_id(x, v_size + Ids.offset_ner)

        elif mode == 'multi':
            if x <= v_size + Ids.offset_ner:
                tok = decode_pos_id(x, v_size + Ids.offset_pos)
            else:
                tok = decode_ner_id(x, v_size + Ids.offset_ner)

        if tok:
            return tok + ' '
        return '<sep> '


def decode_subword_idx_2_tokens_by_tfds(_tokenizer, list_of_list_token_idx, mode):
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
    return list(map(lambda x: list(map(lambda a: decode_one_idx(a, _tokenizer, mode), x)), list_of_list_token_idx))


def decode_pl(mode='pos', multi_task=False, delimiter=''):
    end_idx = Ids.end_cdlm_def + 1 if multi_task else 11
    return [
        {
            'name': 'get_vocab_size',
            'func': lambda x: x.vocab_size,
            'input_keys': ['tokenizer'],
            'output_keys': 'vocab_size',
        },
        {
            'name': 'get_start_end_ids',
            'func': lambda x: [0, x + 1, x + 2] + list(range(x + 5, x + end_idx)),
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
        {
            'name': 'decode_subword_idx_2_tokens_by_tfds',
            'func': decode_subword_idx_2_tokens_by_tfds,
            'input_keys': ['tokenizer', 'input_1', mode],
            'output_keys': 'input_1',
            'show_dict': {'lan': 'input_1'},
        },
        {
            'name': 'join_list_token_2_string',
            'func': utils.join_list_token_2_string,
            'input_keys': ['input_1', delimiter],
            'output_keys': 'input_1',
            'show_dict': {'lan': 'input_1'},
        },
    ]
