import numpy as np
from lib.preprocess import utils


def split_sentence(list_of_sentences):
    list_of_list_token = list(map(lambda x: x.split(' '), list_of_sentences))
    list_of_list_token = list(map(lambda x: list(map(lambda a: a + ' ', x)), list_of_list_token))
    list_of_list_token = list(map(lambda x: x[:-1] + [x[-1].strip()], list_of_list_token))
    return list_of_list_token


split = [
    {
        'name': 'split_lan_1',
        'func': split_sentence,
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
        'show_dict': {'input_1': 'input_1'},
    },
    {
        'name': 'split_lan_2',
        'func': split_sentence,
        'input_keys': ['input_2'],
        'output_keys': 'input_2',
        'show_dict': {'input_2': 'input_2'},
    }
]

encode = [
    {
        'name': 'update vocab_size',
        'func': lambda a: a.vocab_size,
        'input_keys': ['tokenizer'],
        'output_keys': 'vocab_size',
        'show_dict': {'vocab_size': 'vocab_size', 'src_lan': 'input_1', 'tar_lan': 'input_2'},
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
        'name': 'add_start_end_token_to_input_lan_1',
        'func': utils.add_start_end_token_idx_2_list_token_idx,
        'input_keys': ['input_1', 'vocab_size'],
        'output_keys': 'input_1',
        'show_dict': {'input_1': 'input_1'},
    },
    {
        'name': 'add_start_end_token_to_input_lan_2',
        'func': utils.add_start_end_token_idx_2_list_token_idx,
        'input_keys': ['input_2', 'vocab_size'],
        'output_keys': 'input_2',
        'show_dict': {'input_2': 'input_2'},
    },
    {
        'name': 'add_start_end_token_to_ground_truth_1',
        'func': utils.add_start_end_token_idx_2_list_token_idx,
        'input_keys': ['ground_truth_1', 'vocab_size'],
        'output_keys': 'ground_truth_1',
        'show_dict': {'ground_truth_1': 'ground_truth_1'},
    },
    {
        'name': 'add_start_end_token_to_ground_truth_2',
        'func': utils.add_start_end_token_idx_2_list_token_idx,
        'input_keys': ['ground_truth_2', 'vocab_size'],
        'output_keys': 'ground_truth_2',
        'show_dict': {'ground_truth_2': 'ground_truth_2'},
    },
    {
        'name': 'add_start_end_token_to_lan_idx_1',
        'func': lambda list_of_list_lan_idx: list(map(lambda x: [1] + x + [2], list_of_list_lan_idx)),
        'input_keys': ['lan_idx_1'],
        'output_keys': 'lan_idx_1',
        'show_dict': {'lan_idx_1': 'lan_idx_1'},
    },
    {
        'name': 'add_start_end_token_to_lan_idx_2',
        'func': lambda list_of_list_lan_idx: list(map(lambda x: [1] + x + [2], list_of_list_lan_idx)),
        'input_keys': ['lan_idx_2'],
        'output_keys': 'lan_idx_2',
        'show_dict': {'lan_idx_2': 'lan_idx_2'},
    },
    {
        'name': 'add_pad_token_to_input_lan_1',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['input_1', 'max_src_seq_len'],
        'output_keys': 'input_1',
        'show_dict': {'input_1': 'input_1'},
    },
    {
        'name': 'add_pad_token_to_input_lan_2',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['input_2', 'max_tar_seq_len'],
        'output_keys': 'input_2',
        'show_dict': {'input_2': 'input_2'},
    },
    {
        'name': 'add_pad_token_to_ground_truth_1',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['ground_truth_1', 'max_src_ground_seq_len'],
        'output_keys': 'ground_truth_1',
        'show_dict': {'ground_truth_1': 'ground_truth_1'},
    },
    {
        'name': 'add_pad_token_to_ground_truth_2',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['ground_truth_2', 'max_tar_ground_seq_len'],
        'output_keys': 'ground_truth_2',
        'show_dict': {'ground_truth_2': 'ground_truth_2'},
    },
    {
        'name': 'add_pad_token_to_lan_idx_1',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['lan_idx_1', 'max_src_seq_len'],
        'output_keys': 'lan_idx_1',
        'show_dict': {'lan_idx_1': 'lan_idx_1'},
    },
    {
        'name': 'add_pad_token_to_lan_idx_2',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['lan_idx_2', 'max_tar_seq_len'],
        'output_keys': 'lan_idx_2',
        'show_dict': {'lan_idx_2': 'lan_idx_2'},
    },
    {
        'name': 'convert_input_to_array',
        'func': lambda a, b, c, d, e, f: [np.array(a), np.array(b), np.array(c), np.array(d), np.array(e), np.array(f)],
        'input_keys': ['input_1', 'input_2', 'ground_truth_1', 'ground_truth_2', 'lan_idx_1', 'lan_idx_2'],
        'output_keys': ['input_1', 'input_2', 'ground_truth_1', 'ground_truth_2', 'lan_idx_1', 'lan_idx_2'],
    },
    {'output_keys': ['input_1', 'input_2', 'ground_truth_1', 'ground_truth_2', 'lan_idx_1', 'lan_idx_2', 'tokenizer']},
]
