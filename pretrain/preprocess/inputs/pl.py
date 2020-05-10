import numpy as np
from lib.preprocess import utils


def convert_sent_2_tokens(list_of_sentences):
    list_of_list_token = list(map(lambda x: x.split(' '), list_of_sentences))
    list_of_list_token = list(map(lambda x: list(map(lambda a: a + ' ', x)), list_of_list_token))
    list_of_list_token = list(map(lambda x: x[:-1] + [x[-1].strip()], list_of_list_token))
    return list_of_list_token


sent_2_tokens = [
    {
        'name': 'split_lan_1',
        'func': convert_sent_2_tokens,
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
        'show_dict': {'input_1': 'input_1'},
    },
    {
        'name': 'split_lan_2',
        'func': convert_sent_2_tokens,
        'input_keys': ['input_2'],
        'output_keys': 'input_2',
        'show_dict': {'input_2': 'input_2'},
    }
]

filter_exceed_len_inp = [
    {
        'name': 'update vocab_size',
        'func': lambda a: a.vocab_size,
        'input_keys': ['tokenizer'],
        'output_keys': 'vocab_size',
        'show_dict': {'vocab_size': 'vocab_size', 'src_lan': 'input_1', 'tar_lan': 'input_2',
                      'lan_idx_for_gt_1': 'lan_idx_for_gt_1', 'lan_idx_for_gt_2': 'lan_idx_for_gt_2'},
    },
    {
        'name': 'filter_exceed_max_seq_len_with_max_src_seq_len',
        'func': utils.filter_exceed_max_seq_len_together,
        'input_keys': ['max_src_seq_len', 0, 'input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1'],
        'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1'],
        'show_dict': {'lan_idx_for_gt_1': 'lan_idx_for_gt_1'},
    },
    {
        'name': 'filter_exceed_max_seq_len_with_max_tar_seq_len',
        'func': utils.filter_exceed_max_seq_len_together,
        'input_keys': ['max_tar_seq_len', 0, 'input_2', 'ground_truth_2', 'lan_idx_for_input_2', 'lan_idx_for_gt_2'],
        'output_keys': ['input_2', 'ground_truth_2', 'lan_idx_for_input_2', 'lan_idx_for_gt_2'],
        'show_dict': {'lan_idx_for_gt_2': 'lan_idx_for_gt_2'},
    },
]

MLM_filter_exceed_lan_gt = [
    {
        'name': 'filter_exceed_max_seq_len_with_max_src_ground_seq_len',
        'func': utils.filter_exceed_max_seq_len_together,
        'input_keys': ['max_src_ground_seq_len', 1, 'input_1', 'ground_truth_1', 'lan_idx_for_input_1',
                       'lan_idx_for_gt_1'],
        'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1'],
        'show_dict': {'lan_idx_for_gt_1': 'lan_idx_for_gt_1'},
    },
    {
        'name': 'filter_exceed_max_seq_len_with_max_tar_ground_seq_len',
        'func': utils.filter_exceed_max_seq_len_together,
        'input_keys': ['max_tar_ground_seq_len', 1, 'input_2', 'ground_truth_2', 'lan_idx_for_input_2',
                       'lan_idx_for_gt_2'],
        'output_keys': ['input_2', 'ground_truth_2', 'lan_idx_for_input_2', 'lan_idx_for_gt_2'],
    },
]

CDLM_filter_exceed_len_gt = [
    {
        'name': 'filter_exceed_max_seq_len_with_max_src_ground_seq_len',
        'func': utils.filter_exceed_max_seq_len_together,
        'input_keys': ['max_src_ground_seq_len', 1, 'input_1', 'ground_truth_1', 'lan_idx_for_input_1',
                       'lan_idx_for_gt_1', 'pos_for_gt_1'],
        'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1', 'pos_for_gt_1'],
        'show_dict': {'lan_idx_for_gt_1': 'lan_idx_for_gt_1'},
    },
    {
        'name': 'filter_exceed_max_seq_len_with_max_tar_ground_seq_len',
        'func': utils.filter_exceed_max_seq_len_together,
        'input_keys': ['max_tar_ground_seq_len', 1, 'input_2', 'ground_truth_2', 'lan_idx_for_input_2',
                       'lan_idx_for_gt_2', 'pos_for_gt_2'],
        'output_keys': ['input_2', 'ground_truth_2', 'lan_idx_for_input_2', 'lan_idx_for_gt_2', 'pos_for_gt_2'],
        'show_dict': {'lan_idx_for_gt_2': 'lan_idx_for_gt_2'},
    },
]

MLM_add_pad = [
    {
        'name': 'add_pad_token_to_lan_for_src',
        'func': lambda l, list_of_list_lan_idx: list(
            map(lambda x: x + [x[-1]] * int(l - len(x)), list_of_list_lan_idx)),
        'input_keys': ['max_src_seq_len', 'lan_idx_for_input_1'],
        'output_keys': ['lan_idx_for_input_1'],
        'show_dict': {'lan_idx_for_input_1': 'lan_idx_for_input_1'},
    },
    {
        'name': 'add_pad_token_to_lan_for_tar',
        'func': lambda l, list_of_list_lan_idx: list(
            map(lambda x: x + [x[-1]] * int(l - len(x)), list_of_list_lan_idx)),
        'input_keys': ['max_tar_seq_len', 'lan_idx_for_input_2'],
        'output_keys': ['lan_idx_for_input_2'],
        'show_dict': {'lan_idx_for_input_2': 'lan_idx_for_input_2'},
    },
    {
        'name': 'add_pad_token_to_lan_gt_for_src',
        'func': lambda l, list_of_list_lan_idx: list(
            map(lambda x: x + [x[-1]] * int(l - len(x)), list_of_list_lan_idx)),
        'input_keys': ['max_src_ground_seq_len', 'lan_idx_for_gt_1'],
        'output_keys': ['lan_idx_for_gt_1'],
        'show_dict': {'lan_idx_for_gt_1': 'lan_idx_for_gt_1'},
    },
    {
        'name': 'add_pad_token_to_lan_gt_for_tar',
        'func': lambda l, list_of_list_lan_idx: list(
            map(lambda x: x + [x[-1]] * int(l - len(x)), list_of_list_lan_idx)),
        'input_keys': ['max_tar_ground_seq_len', 'lan_idx_for_gt_2'],
        'output_keys': ['lan_idx_for_gt_2'],
        'show_dict': {'lan_idx_for_gt_2': 'lan_idx_for_gt_2'},
    },
    {
        'name': 'add_pad_token_to_input_for_src',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['input_1', 'max_src_seq_len'],
        'output_keys': ['input_1'],
        'show_dict': {'input_1': 'input_1'},
    },
    {
        'name': 'add_pad_token_to_input_for_tar',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['input_2', 'max_tar_seq_len'],
        'output_keys': ['input_2'],
        'show_dict': {'input_2': 'input_2'},
    },
    {
        'name': 'add_pad_token_to_ground_truth_for_src',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['ground_truth_1', 'max_src_ground_seq_len'],
        'output_keys': ['ground_truth_1'],
        'show_dict': {'ground_truth_1': 'ground_truth_1'},
    },
    {
        'name': 'add_pad_token_to_ground_truth_for_tar',
        'func': utils.add_pad_token_idx_2_list_token_idx,
        'input_keys': ['ground_truth_2', 'max_tar_ground_seq_len'],
        'output_keys': ['ground_truth_2'],
        'show_dict': {'ground_truth_2': 'ground_truth_2'},
    },
]

CDLM_add_pad = [
    {
        'name': 'add_pad_token_to_pos_gt_for_src',
        'func': lambda l, list_of_pos_ids: list(map(
            lambda x: x + list(range(int(x[-1] + 1), int(x[-1] + 1) + int(l - len(x)))), list_of_pos_ids)),
        'input_keys': ['max_src_ground_seq_len', 'pos_for_gt_1'],
        'output_keys': ['pos_for_gt_1'],
        'show_dict': {'pos_for_gt_1': 'pos_for_gt_1'},
    },
    {
        'name': 'add_pad_token_to_pos_gt_for_tar',
        'func': lambda l, list_of_pos_ids: list(map(
            lambda x: x + list(range(int(x[-1] + 1), int(x[-1] + 1) + int(l - len(x)))), list_of_pos_ids)),
        'input_keys': ['max_tar_ground_seq_len', 'pos_for_gt_2'],
        'output_keys': ['pos_for_gt_2'],
        'show_dict': {'pos_for_gt_2': 'pos_for_gt_2'},
    },
]

MLM_encode = [
    *filter_exceed_len_inp,
    *MLM_filter_exceed_lan_gt,
    *MLM_add_pad,
    {
        'name': 'merge_src_tar_lan',
        'func': lambda a1, a2, b1, b2, c1, c2, d1, d2: [np.array(a1 + a2), np.array(b1 + b2),
                                                        np.array(c1 + c2), np.array(d1 + d2)],
        'input_keys': ['input_1', 'input_2', 'ground_truth_1', 'ground_truth_2',
                       'lan_idx_for_input_1', 'lan_idx_for_input_2', 'lan_idx_for_gt_1', 'lan_idx_for_gt_2'],
        'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1'],
    },
]

CDLM_encode = [
    *filter_exceed_len_inp,
    *CDLM_filter_exceed_len_gt,
    *MLM_add_pad,
    *CDLM_add_pad,
    {
        'name': 'merge_src_tar_lan',
        'func': lambda a1, a2, b1, b2, c1, c2, d1, d2, e1, e2: [np.array(a1 + a2), np.array(b1 + b2),
                                                                np.array(c1 + c2), np.array(d1 + d2),
                                                                np.array(e1 + e2)],
        'input_keys': ['input_1', 'input_2', 'ground_truth_1', 'ground_truth_2',
                       'lan_idx_for_input_1', 'lan_idx_for_input_2', 'lan_idx_for_gt_1', 'lan_idx_for_gt_2',
                       'pos_for_gt_1', 'pos_for_gt_2'],
        'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1', 'pos_for_gt_1'],
    },
]
