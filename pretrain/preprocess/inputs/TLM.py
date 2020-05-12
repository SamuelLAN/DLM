from pretrain.preprocess.inputs.MLM import MLM
from pretrain.preprocess.config import LanIds


def TLM_list_of_tokens(list_of_zh_words, list_of_en_words, _tokenizer,
                       min_num=1, max_num=4, max_ratio=0.2, keep_origin_rate=0.2):
    data = list(zip(list_of_zh_words, list_of_en_words))
    data = list(map(lambda x: TLM(x[0], x[1], _tokenizer, min_num, max_num, max_ratio, keep_origin_rate), data))
    data = list(filter(lambda x: x, data))

    _inputs, _outputs, _lan_inputs, _lan_outputs = list(zip(*data))
    return _inputs, _outputs, _lan_inputs, _lan_outputs


def TLM_concat(*args):
    return list(map(lambda x: x[0] + x[1], args))


def TLM(list_of_zh_word, list_of_en_word, _tokenizer, min_num=1, max_num=4, max_ratio=0.2, keep_origin_rate=0.2):
    zh_input, zh_output, zh_lan_input, zh_lan_output = MLM(list_of_zh_word, _tokenizer, LanIds.zh, min_num, max_num,
                                                           max_ratio, keep_origin_rate)

    en_input, en_output, en_lan_input, en_lan_output = MLM(list_of_en_word, _tokenizer, LanIds.en, min_num, max_num,
                                                           max_ratio, keep_origin_rate)

    if not zh_input or not en_input:
        return

    _input, _output, _lan_input, _lan_output = TLM_concat(
        [zh_input, en_input], [zh_output, en_output], [zh_lan_input, en_lan_input], [zh_lan_output, en_lan_output])

    return _input, _output, _lan_input, _lan_output


def get_pl(min_num=1, max_num=4, max_ratio=0.2, keep_origin_rate=0.2):
    return [{
        'name': 'TLM',
        'func': TLM_list_of_tokens,
        'input_keys': ['input_1', 'input_2', 'tokenizer', min_num, max_num, max_ratio, keep_origin_rate],
        'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1'],
        'show_dict': {'input_1': 'input_1', 'ground_truth_1': 'ground_truth_1',
                      'lan_idx_for_input_1': 'lan_idx_for_input_1', 'lan_idx_for_gt_1': 'lan_idx_for_gt_1'},
    }]


if __name__ == '__main__':
    from nmt.preprocess.corpus import wmt_news
    from lib.preprocess import utils
    from nmt.preprocess.inputs import noise_pl, tfds_share_pl, zh_en
    from pretrain.preprocess.inputs import pl
    from pretrain.preprocess.inputs.sampling import sample_pl

    origin_zh_data, origin_en_data = wmt_news.zh_en()
    params = {
        'vocab_size': 45000,
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'max_src_ground_seq_len': 8,
        'max_tar_ground_seq_len': 8,
    }

    pipeline = zh_en.seg_zh_by_jieba_pipeline + noise_pl.remove_noise + tfds_share_pl.train_tokenizer
    pipeline += pl.sent_2_tokens + sample_pl(3.0) + get_pl(1, 4, 0.2, 0.2) + pl.TLM_encode + [
        {'output_keys': ['input_1', 'ground_truth_1', 'lan_idx_for_input_1', 'lan_idx_for_gt_1', 'tokenizer']}
    ]

    print('\n------------------- Encoding -------------------------')
    x, y, lan_x, lan_y, tokenizer = utils.pipeline(
        preprocess_pipeline=pipeline,
        lan_data_1=origin_zh_data[:1000], lan_data_2=origin_en_data[:1000], params=params)

    print('\n----------------------------------------------')
    print(x.shape)
    print(y.shape)
    print(lan_x.shape)
    print(lan_y.shape)

    print('\n------------------- Decoding zh -------------------------')
    x = utils.pipeline(tfds_share_pl.decode_pipeline, x, None, {'tokenizer': tokenizer})
    y = utils.pipeline(tfds_share_pl.decode_pipeline, y, None, {'tokenizer': tokenizer})
