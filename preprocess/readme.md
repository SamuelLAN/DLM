# preprocess corpus data

> what are done

- Segment sentences

- Tokenize (segment words)

- Subword (BPE)

    could use BPE (Byte-Pair-Encoding), WordPiece or SentencePiece

- Filter data which exceed maximum length

- add <start> and <end> tokens

- add <pad> tokens so that the length of sentence is equal to max sequence length

> Directory Structure

- wmt_news.py

    download and format data from WMT-news-2019.

- europarl.py

    download and format data from europarl-2017.

- KFTT.py

    download and format data from KFTT.

- UM_corpus.py

    format data from UM_corpus,

- tfds_pl.py

    use tensorflow_datasets.features.text.SubwordTextEncoder to do the above-mentioned preprocess process

- tfds_share_pl.py

    same as tfds_pl.py, the only difference is that the source language and the target language would share the same subword tokenizer, which means they have share BPE encodings and use the same vocabulary.

- noise_pl.py

    remove noise for the source and the target sentences.

- zh_en.py

    provide some preprocess pipeline for zh-en.

- jr_en.py

    provide some preprocess pipeline for jr-en.

- de_fr_en.py

    provide some preprocess pipeline for de-en or fr-en.
