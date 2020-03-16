import numpy as np
from lib.preprocess import utils
import lib.utils as l_utils
from preprocess import wmt_news


def char_zh_word_en_monolingual(max_zh_seq_len=80, max_en_seq_len=60,
                                zh_vocab_size=3000, en_vocab_size=2 ** 13, use_cache=True):
    # read from cache
    cache_name = f'char_zh_word_en_monolingual_{max_zh_seq_len}_{max_en_seq_len}.pkl'
    if use_cache:
        data = l_utils.read_cache(cache_name)
        if not isinstance(data, type(None)):
            return data

    # load data
    zh_data, en_data = wmt_news.zh_en()

    # seg chinese to character level
    list_of_zh_token = utils.char_seg(zh_data)

    # add <start> <end> token
    en_data = utils.add_start_end_token_2_string(en_data)
    # tokenize English sentences to token
    en_tokenizer = utils.train_subword_tokenizer_by_tfds(en_data, en_vocab_size, max_en_seq_len, ['<start>', '<end>'])
    # encode string to list of token idx
    list_of_en_token_idx = utils.encoder_string_2_subword_idx_by_tfds(en_tokenizer, en_data)

    # filter sentences which exceed max_seq_len
    list_of_zh_token, list_of_en_token_idx = utils.filter_exceed_max_seq_len_for_cross_lingual(
        list_of_zh_token, list_of_en_token_idx, max_zh_seq_len - 2, max_en_seq_len
    )

    # add <start> <end> tokens
    list_of_zh_token = utils.add_start_end_token_2_list_token(list_of_zh_token)
    # add <pad> tokens and filter sentences which exceed max_seq_len
    list_of_zh_token = utils.add_pad_token_2_list_token(list_of_zh_token, max_zh_seq_len)
    # convert tokens to token_idx
    list_of_zh_token_idx, zh_char_dictionary = utils.doc_2_idx(list_of_zh_token, keep_n=zh_vocab_size)

    vocab_size = en_tokenizer.vocab_size
    # add pad token idx
    list_of_en_token_idx = utils.add_pad_token_idx_2_list_token_idx(list_of_en_token_idx, vocab_size, max_en_seq_len, 0)

    # # show some examples
    # # remove the out of vocabulary token idx (idx for <start>, <end>, <pad>)
    # tmp_list_of_en_token_idx = utils.remove_out_of_vocab_token_idx(list_of_en_token_idx, vocab_size)
    # # decode token idx to tokens
    # list_of_en_token = utils.decode_subword_idx_2_tokens_by_tfds(en_tokenizer, tmp_list_of_en_token_idx)
    # for i, v in enumerate(list_of_en_token[:10]):
    #     print(f'\n---------------- {i} -------------------')
    #     print(en_data[i])
    #     print(v)
    #     print(zh_data[i])
    #     print(list_of_zh_token[i])
    #     print('')
    #
    # print('\n###################################33')
    # print(en_tokenizer.vocab_size)
    # print(len(zh_char_dictionary))
    # print(np.array(list_of_zh_token_idx).shape)
    # print(np.array(list_of_en_token_idx).shape)

    # convert list to array
    arr_of_zh_token_idx = np.array(list_of_zh_token_idx, dtype=np.int32)
    arr_of_en_token_idx = np.array(list_of_en_token_idx, dtype=np.int32)

    l_utils.cache(cache_name, [arr_of_zh_token_idx, arr_of_en_token_idx, zh_char_dictionary, en_tokenizer])
    return arr_of_zh_token_idx, arr_of_en_token_idx, zh_char_dictionary, en_tokenizer


def word_zh_word_en_monolingual(max_zh_seq_len=50, max_en_seq_len=60,
                                zh_vocab_size=5000, en_vocab_size=2 ** 13, use_cache=True):
    # read from cache
    cache_name = f'word_zh_word_en_monolingual_{max_zh_seq_len}_{max_en_seq_len}.pkl'
    if use_cache:
        data = l_utils.read_cache(cache_name)
        if not isinstance(data, type(None)):
            return data

    # load data
    zh_data, en_data = wmt_news.zh_en()

    # seg chinese to character level
    list_of_zh_token = utils.zh_word_seg_by_jieba(zh_data)

    # add <start> <end> token
    en_data = utils.add_start_end_token_2_string(en_data)
    # tokenize English sentences to token
    en_tokenizer = utils.train_subword_tokenizer_by_tfds(en_data, en_vocab_size, max_en_seq_len, ['<start>', '<end>'])
    # encode string to list of token idx
    list_of_en_token_idx = utils.encoder_string_2_subword_idx_by_tfds(en_tokenizer, en_data)

    # filter sentences which exceed max_seq_len
    list_of_zh_token, list_of_en_token_idx = utils.filter_exceed_max_seq_len_for_cross_lingual(
        list_of_zh_token, list_of_en_token_idx, max_zh_seq_len - 2, max_en_seq_len
    )

    # add <start> <end> tokens
    list_of_zh_token = utils.add_start_end_token_2_list_token(list_of_zh_token)
    # add <pad> tokens and filter sentences which exceed max_seq_len
    list_of_zh_token = utils.add_pad_token_2_list_token(list_of_zh_token, max_zh_seq_len)
    # convert tokens to token_idx
    list_of_zh_token_idx, zh_char_dictionary = utils.doc_2_idx(list_of_zh_token, keep_n=zh_vocab_size)

    vocab_size = en_tokenizer.vocab_size
    # add pad token idx
    list_of_en_token_idx = utils.add_pad_token_idx_2_list_token_idx(list_of_en_token_idx, vocab_size, max_en_seq_len, 0)

    # # show some examples
    # # remove the out of vocabulary token idx (idx for <start>, <end>, <pad>)
    # tmp_list_of_en_token_idx = utils.remove_out_of_vocab_token_idx(list_of_en_token_idx, vocab_size)
    # # decode token idx to tokens
    # list_of_en_token = utils.decode_subword_idx_2_tokens_by_tfds(en_tokenizer, tmp_list_of_en_token_idx)
    # for i, v in enumerate(list_of_en_token[:10]):
    #     print(f'\n---------------- {i} -------------------')
    #     print(en_data[i])
    #     print(v)
    #     print(zh_data[i])
    #     print(list_of_zh_token[i])
    #     print('')
    #
    # print('\n###################################33')
    # print(en_tokenizer.vocab_size)
    # print(len(zh_char_dictionary))
    # print(np.array(list_of_zh_token_idx).shape)
    # print(np.array(list_of_en_token_idx).shape)

    # convert list to array
    arr_of_zh_token_idx = np.array(list_of_zh_token_idx, dtype=np.int32)
    arr_of_en_token_idx = np.array(list_of_en_token_idx, dtype=np.int32)

    l_utils.cache(cache_name, [arr_of_zh_token_idx, arr_of_en_token_idx, zh_char_dictionary, en_tokenizer])
    return arr_of_zh_token_idx, arr_of_en_token_idx, zh_char_dictionary, en_tokenizer


# arr_of_zh_token_idx, arr_of_en_token_idx, zh_char_dictionary, en_tokenizer = word_zh_word_en_monolingual(50, 60)
# print(arr_of_zh_token_idx.shape)
# print(arr_of_en_token_idx.shape)
# print(len(zh_char_dictionary))
# print(en_tokenizer.vocab_size)
