def __sample(_list, ratio):
    """ sample data from list_of_sentences """
    if ratio <= 0:
        return []
    elif ratio <= 1:
        return _list[: int(len(_list) * ratio)]
    else:
        return _list * int(ratio) + \
               _list[: int(len(_list) * (ratio - int(ratio)))]


def sampling(src_data, tar_data, src_ratio=1.0, tar_ratio=1.0, dual_ratio=1.0):
    src_list = __sample(src_data, src_ratio)
    tar_list = __sample(tar_data, tar_ratio)
    dual_list = __sample(list(zip(src_data, tar_data)), dual_ratio)
    return src_list, tar_list, dual_list
