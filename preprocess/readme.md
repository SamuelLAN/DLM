# preprocess data

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

- tfds_pl.py

    use tensorflow_datasets.features.text.SubwordTextEncoder to do the above-mentioned preprocess process

- zh_en.py

- jr_en.py

- de_fr_en.py
