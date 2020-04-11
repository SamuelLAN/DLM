import random
from functools import reduce

def MLM(list_of_words, min_num = 2, max_num = 6, max_ratio = 0.2, tokenizer):
    
    list_of_words = list(map(lambda x: tokenizer.encode(x), list_of_words))
    index_to_mask = random.sample(range(0,len(list_of_words)), min(random.randint(min_num, max_num), round(len(list_of_words) * max_ratio))  )
    tar = reduce(lambda x,y : x+y, [list_of_words[i] for i in index_to_mask])
    masked = []
    for i in range(len(list_of_words)):
        if i in index_to_mask:
            masked.append(["<masked>"]*len(list_of_words[i]))
        else:
            masked.append(list_of_words[i])
    masked = reduce(lambda x,y : x+y, masked)
    return(masked, tar, [0]*len(masked))
    