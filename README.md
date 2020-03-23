# DLM
Dictionary Language Modeling for cross-lingual pretraining on Neural Machine Translation

### environment

- Python 3.6+
- tensorflow==2.0.0+
- tensorflow_datasets=2.1.0
- nltk==3.4.5
- mecab==0.996.2	
- jieba==0.42.1
- pkuseg==0.0.22
- gensim==3.8.1
- chardet==3.0.4
- six==1.12.0
- matplotlib==3.1.1

### dataset

zh-en:
- [wikititles-v1.zh-en.tsv.gz](http://data.statmt.org/wikititles/v1/wikititles-v1.zh-en.tsv.gz)
- [news-commentary-v14.en-zh.tsv.gz](http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-zh.tsv.gz)
- [WMT-News](http://opus.nlpl.eu/download.php?f=WMT-News/v2019/moses/en-zh.txt.zip)

jr-en
- [ASPEC](http://orchid.kuee.kyoto-u.ac.jp/ASPEC/)

de-en:
- [wikititles-v1.de-en.tsv.gz](http://data.statmt.org/wikititles/v1/wikititles-v1.de-en.tsv.gz)
- [news-commentary-v14.de-en.tsv.gz](http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.de-en.tsv.gz)
- [europarl-v9.de-en.tsv.gz](http://www.statmt.org/europarl/v9/training/europarl-v9.de-en.tsv.gz)
- [Wikipedia](http://opus.nlpl.eu/download.php?f=Wikipedia/v1.0/moses/de-en.txt.zip)
- [WMT-News](http://opus.nlpl.eu/download.php?f=WMT-News/v2019/moses/de-en.txt.zip)

fr-en:
- [europarl-v9.fi-en.tsv.gz](http://www.statmt.org/europarl/v9/training/europarl-v9.fi-en.tsv.gz)
- [Wikipedia](http://opus.nlpl.eu/download.php?f=Wikipedia/v1.0/tmx/en-fr.tmx.gz)
- [WMT-News](http://opus.nlpl.eu/download.php?f=WMT-News/v2019/moses/en-fr.txt.zip)

### note

- When using the project, <br>
  please remember to modify the "__data_dir" variable <br> 
  in the top of "/preprocess/wmt-news.py" and "/preprocess/europarl.py"
