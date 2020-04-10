from lib.preprocess import utils

remove_noise = [
    {
        'name': 'remove_noise_for_src',
        'func': utils.remove_noise_for_sentences,
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'remove_noise_for_tar',
        'func': utils.remove_noise_for_sentences,
        'input_keys': ['input_2'],
        'output_keys': 'input_2',
        'show_dict': {'tar_lan': 'input_2'},
    },
    {
        'name': 'split_sentences',
        'func': utils.split_sentences,
        'input_keys': ['input_1', 'input_2'],
        'output_keys': ['input_1', 'input_2'],
        'show_dict': {'src_lan': 'input_1', 'tar_lan': 'input_2'},
    },
    {
        'name': 'lower_sentences_for_src',
        'func': utils.lower_sentences,
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'lower_sentences_for_tar',
        'func': utils.lower_sentences,
        'input_keys': ['input_2'],
        'output_keys': 'input_2',
        'show_dict': {'tar_lan': 'input_2'},
    },
]

remove_noise_for_src = [
    {
        'name': 'remove_noise_for_src',
        'func': utils.remove_noise_for_sentences,
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'split_sentences',
        'func': utils.split_sentences,
        'input_keys': ['input_1', 'input_1'],
        'output_keys': ['input_1', 'input_2'],
        'show_dict': {'src_lan': 'input_1'},
    },
    {
        'name': 'lower_sentences',
        'func': utils.lower_sentences,
        'input_keys': ['input_1'],
        'output_keys': 'input_1',
        'show_dict': {'src_lan': 'input_1'},
    },
]
