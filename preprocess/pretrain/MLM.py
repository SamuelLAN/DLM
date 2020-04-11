import random
from functools import reduce


def MLM(list_of_words_for_a_sentence, tokenizer, lan_index, min_num=2, max_num=6, max_ratio=0.2):
    """
    Masked Language Modeling (word level)
    :params
        list_of_words_for_a_sentence (list): ['I', 'am', 'a', 'student']
        tokenizer (object): tfds tokenizer object
        lan_index (int): index for language embeddings, could be 0 or 1
        min_num (int):
        max_num (int):
        max_ratio (float):
    :returns
        masked_input (list): list of encoded and masked token idx
        list_of_tar_token_idx (list):
        list_of_lan_idx (list):
    """

    # BPE for each word
    list_of_list_token_idx = list(map(lambda x: tokenizer.encode(x), list_of_words_for_a_sentence))

    # get indices to mask
    len_words = len(list_of_list_token_idx)
    indices_to_mask = random.sample(
        range(len_words),
        min(random.randint(min_num, max_num), round(len_words * max_ratio))
    )

    # get ground truth
    list_of_tar_token_idx = reduce(lambda x, y: x + y, [list_of_list_token_idx[i] for i in indices_to_mask])

    # get masked input
    mask_idx = tokenizer.vocab_size + 3
    masked_input = []
    for i in range(len_words):
        idxs_for_word = list_of_list_token_idx[i]
        masked_input += [mask_idx] * len(idxs_for_word) if i in indices_to_mask else idxs_for_word

    # get index for language embeddings
    list_of_lan_idx = [lan_index] * len(masked_input)

    return masked_input, list_of_tar_token_idx, list_of_lan_idx
