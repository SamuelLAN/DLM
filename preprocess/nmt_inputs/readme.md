# preprocess the inputs for Neural Machine Translatioin

- remove noises

- Segment sentences

- Tokenize (segment words)

- Subword (BPE)

    could use BPE (Byte-Pair-Encoding), WordPiece or SentencePiece

- Filter data which exceed maximum length

- add <start> and <end> tokens

- add <pad> tokens so that the length of sentence is equal to max sequence length

