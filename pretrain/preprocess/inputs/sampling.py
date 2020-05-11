def sample(_list, ratio):
    """ sample data from list_of_sentences """
    if ratio <= 0:
        return []
    elif ratio <= 1:
        return _list[: int(len(_list) * ratio)]
    else:
        return _list * int(ratio) + \
               _list[: int(len(_list) * (ratio - int(ratio)))]


def sample_together(src_data, tar_data, ratio=1.0):
    if ratio == 1.0:
        return src_data, tar_data
    src_list = sample(src_data, ratio)
    tar_list = sample(tar_data, ratio)
    return src_list, tar_list


def sample_pl(ratio=1.0):
    return [
        {
            'name': 'sample data',
            'func': sample_together,
            'input_keys': ['input_1', 'input_2', ratio],
            'output_keys': ['input_1', 'input_2'],
        },
    ]
