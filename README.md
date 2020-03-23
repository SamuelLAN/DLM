# DLM
Dictionary Language Modeling for cross-lingual pretraining on Neural Machine Translation

### environment

- Python 3.7
- tensorflow==2.0.0+
- tensorflow_datasets=2.1.0
- nltk==3.4.5
- mecab-python3==0.996.2	
- jieba==0.42.1
- pkuseg==0.0.22
- gensim==3.8.1
- chardet==3.0.4
- six==1.12.0
- matplotlib==3.1.1
- torch=1.4.0
- torchvision=0.5.0

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
  in the top of "[preprocess/wmt-news.py](preprocess/wmt-news.py)" and "[preprocess/europarl.py](preprocess/europarl.py)"

### for training

Tutorial for training the models

> load data

    at the top of train.py
    
    There is a line of code like "from load.xxx import Loader"
    
    change the xxx to zh_en; or any other languages

> if use transformer in pytorch instead of tensorflow (optional)

    at the top of train.py
    
    There is a line of code like "from models.transformer_for_nmt import Model"
    
    change it to "from models.transformer_for_nmt_torch import Model"
    
> train.py

    at the bottom of train.py
    
    make sure it is 
        o_train = Train(use_cache=True)
        o_train.train()
        o_train.test()

    The "use_cache" param indicate whether to load the preprocessed data from cache if there is cache

> for adjusting the parameters, loss, optimizer and so on

    it is in models/transformer_for_nmt.py
    
    You can change
        
    + data_params
    
    + model_params
    
    + train_params
    
    + compile_params
    
    + monitor_params

> if you want to choose dataset

    it is in top of the "__init__" function of "Loader" in load/zh_en.py

> if you want to change the preprocess pipeline

    it is in models/transformer_for_nmt.py

    you could change the pipelines at the top of the "Model"

> if you want to load the trained model

    it is in the "checkpoint_params" of models/transformer_for_nmt.py

    you can specify the "name, time" of the model_dir, then it would load the best model automatically.

> tensorboard

    the tensorboard files will be save in the "runtime/tensorboard" directory.
    
    This directory will be generated automatically after running the train.py

> model files

    all model files will be saved in "runtime/models"
    
    This directory will be generated automatically after running the train.py

> log

    All the results of running train.py will be logged into the "runtime/log".
    
    Including the data params, model params, train params and the results.
    
    But if the train.py exits before it finishes, then there would be no logs.

### for testing

Tutorial for testing the models

> train.py

    at the bottom of train.py
    
    make sure it is 
        o_train = Train(use_cache=True)
        # o_train.train()
        o_train.test(True)

    The "use_cache" param indicate whether to load the preprocessed data from cache if there is cache

> choose which model to load

    it is in the "checkpoint_params" of models/transformer_for_nmt.py

    you can specify the "name, time" of the model_dir, then it would load the best model automatically.
