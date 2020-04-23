from nltk.translate.bleu_score import corpus_bleu
from nmt.models.base_model import BaseModel
from nmt.preprocess.inputs import jr_en, tfds_pl
from lib.preprocess import utils
from lib.utils import create_dir_in_root
import tensorflow as tf
import torch
import torch.nn as nn
import math
import numpy as np


# from pytorchtools import
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            # device = src.device
            # mask = self._generate_square_subsequent_mask(len(src)).to(device)
            mask = self._generate_square_subsequent_mask(len(src))

            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

class Model(TransformerModel):
    name = 'transformer_for_nmt_torch'

    preprocess_pipeline = jr_en.seg_jr_by_mecab_pipeline + tfds_pl.train_tokenizer_pipeline + \
                          tfds_pl.encode_pipeline
    # for test
    encode_pipeline = jr_en.seg_jr_by_mecab_pipeline + tfds_pl.encode_pipeline
    encode_pipeline_for_src = jr_en.seg_jr_by_mecab_pipeline + tfds_pl.encode_pipeline_for_src
    encode_pipeline_for_tar = tfds_pl.encode_pipeline_for_src
    decode_pipeline_for_src = tfds_pl.decode_pipeline + jr_en.remove_space_pipeline
    decode_pipeline_for_tar = tfds_pl.decode_pipeline

    data_params = {
        **BaseModel.data_params,
        'src_vocab_size': 15000,  # approximate
        'tar_vocab_size': 2 ** 13,  # approximate
        'max_src_seq_len': 60,
        'max_tar_seq_len': 60,
        'sample_rate': 1.0,  # sample "sample_rate" percentage of data into dataset; range from 0 ~ 1
    }

    model_params = {
        'emb_dim': 128,
        'dim_model': 1024,
        'ff_units': 1024,
        'num_layers': 6,
        'num_heads': 8,
        'max_pe_input': data_params['max_src_seq_len'],
        'max_pe_target': data_params['max_tar_seq_len'],
        'drop_rate': 0.1,
        'use_beam_search': False,
        'top_k': 5,
        'get_random': False,
    }

    train_params = {
        **BaseModel.train_params,
        'learning_rate': 2e-5,
        # 'learning_rate': CustomSchedule(model_params['dim_model']),
        'batch_size': 64,
        'epoch': 100,
        'early_stop': 20,
    }

    checkpoint_params = {
        'load_model': [],  # [name, time]
    }

    def __init__(self, input_vocab_size, target_vocab_size, name='', finish_train=False):
        self.name = name if name else self.name
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.__finish_train = finish_train

        # create directories for tensorboard files and model files
        self.__create_dir()

        # build models
        self.build()

    def build(self):
        """
        Build the model to self.model
            no returns
        """
        # TODO build the pytorch model here
        #   the "model" should provide the following functions:
        #       __call__(inputs, training=None)
        #           :params: inputs is ( encoder_input, decoder_input )
        #           :return: decoder_output
        #           (It would be used in the "self.calculate_loss_for_encoded" functions during the test process)
        #       evaluate_list_of_list_token_idx
        #           (for generating the translation, which means iterate the predictions to do translations.
        #               Details are in the /lib/tf_models/transformer.)
        #       beam_search_list_of_list_token_idx (optional)
        #
        #   for the params of the model, please use the above data_params, model_params and train_params
        #
        # initialize the model instance
        # self.model = pytorch Model
        # ...
        self.model = TransformerModel(self.data_params["ff_units"], self.data_params["emb_dim"],
                                      self.data_params['num_heads'], self.data_params['num_heads'],
                                      self.data_params['num_layers'], self.data_params['drop_rate']).to(device)
        print("okay")

    def load_model(self, model_dir='', x=None, y=None):
        """
        load the model to self.model
            no returns
        """
        # TODO
        #   given model_dir, you could load the best model to self.model in the model dir
        #   if model_dir is empty, then you could use self.model_dir to load the model
        # ...
        self.model = torch.load(model_dir)
        self.model.eval()

    def loss(self, y_true, y_pred, from_logits=False, label_smoothing=0):
        """
        Calculate the loss
          (The test process will use this function)
        TODO
            you should provide this function no matter you use it in training or not;
            because the test process would call this function
        :return: loss (float)
        """
        y_true = torch.view(-1, y_true)

        # calculate the padding mask
        mask = torch.cast(torch.math.not_equal(y_true, 0), y_pred.dtype)

        # calculate the loss
        loss_ = self.compile_params['loss'](y_true, y_pred)

        # remove the padding part's loss by timing the mask
        loss_ *= mask

        # calculate the mean loss
        return tf.reduce_mean(loss_)

    def train(self, train_x, train_y, val_x=None, val_y=None):
        """
        TODO
            you need to override this train functions

        :params: train_x (tuple): a tuple with two elements, each one is a np.array
                                    (encoder_input, decoder_input)
                                    e.g., (train_src_encode, train_tar_encode[:, :-1])
        :params: train_y (np.array): decoder_output (ground_truth)
                                    train_tar_encode[:, 1:]
        :params: val_x (tuple): Use for validation, could be None.
                                If None, then no validation; if it is not None, it would be the same as train_X
        :params: val_y (np.array): Use for validation, could be None.
                                If None, then no validation; if it is not None, it would be the same as train_y

        no returns
        """

        # if we want to load a trained model
        if self.checkpoint_params['load_model']:
            model_dir = create_dir_in_root(*(['runtime', 'models'] + self.checkpoint_params['load_model']))
            self.load_model(model_dir)

        if not self.__finish_train:
            # TODO
            #   you should write down the train process here
            #   This process should also provide the following functions:
            #       calculate the loss
            #       update model weights
            #       early stop
            #       validation
            #       something like tensorboard (for saving files for tensorboard, you should save it to self.tb_dir)
            #       print out the results to console for each epoch
            #       save the best model to self.model_dir
            # ...
            #
            # for step in range(steps)
            # ...

            # load the best model so that it could be tested
            # from torchtext.data import Field, BucketIterator, TabularDataset
            # import load.jr_en as loader
            # JP_TEXT = Field(tokenize=jr_en.jr_tokenizer)
            # EN_TEXT = Field(tokenize=jr_en.en_tokenizer)
            # print("Preparing to load data to batchify data")
            # dataLoader = loader.Loader()
            # jr_data, en_data = dataLoader.data()
            # print("Successfully splited the data")
            # JP_TEXT.build_vocab(jr_data, val)
            # EN_TEXT.build_vocab(en_data, val)

            from torchtext.data import Field
            JP_TEXT = Field(tokenize=jr_en.jr_tokenizer)
            EN_TEXT = Field(tokenize=jr_en.en_tokenizer)
            ##batchify the data
            def batchify(data, batchsize, TEXT):
                data = TEXT.numericalize([data.examples[0].text])
                # Divide the dataset into bsz parts.
                nbatch = data.size(0) // batchsize
                # Trim off any extra elements that wouldn't cleanly fit (remainders).
                data = data.narrow(0, 0, nbatch * batchsize)
                # Evenly divide the data across the bsz batches.
                data = data.view(batchsize, -1).t().contiguous()
                return data.to(device)
            jr_data = batchify(train_x, self.train_params['batch_size'], JP_TEXT)
            en_data = batchify(train_x, self.train_params['batch_size'], EN_TEXT)
            emb_dim = self.model_params["emb_dim"]
            def get_batch(source, i):
                seq_len = emb_dim
                data = source[i:i+seq_len]
                target = source[i+1:i+1+seq_len].view(-1)
                return data, target
            import time
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.train_params['learning_rate'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
            for epoch in range(1, self.checkpoint_params["epoch"] + 1):
            #     optimizer.zero_grad()
            #     output = self.model(train_x)
            #     loss = criterion(output.view(-1, self.checkpoint_params["epoch"]), train_y)
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            #     optimizer.step()
            #
            # # load the best model so that it could be tested
            # self.load_model()
            #
            # self.__finish_train = True
            # model.train() # Turn on the train mode
                data, targets = get_batch(jr_data, epoch)
                total_loss = 0.
                start_time = time.time()
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output.view(-1, data), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                log_interval = 200
                # if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, 1, len(jr_data) // emb_dim, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))


    def translate_sentences(self, list_of_src_sentences, src_tokenizer, tar_tokenizer):
        """ translate list of sentences and decode the results """
        encoded_data = utils.pipeline(self.encode_pipeline_for_src, list_of_src_sentences, None, {
            'src_tokenizer': src_tokenizer,
            'max_src_seq_len': self.data_params['max_src_seq_len'],
        })

        pred_encoded = self.evaluate(encoded_data)
        return self.decode_tar_data(pred_encoded, tar_tokenizer)

    def translate_list_token_idx(self, list_of_list_of_src_token_idx, tar_tokenizer):
        """ translate the src list token idx to target language sentences """
        pred_encoded = self.evaluate(list_of_list_of_src_token_idx)
        return self.decode_tar_data(pred_encoded, tar_tokenizer)

    def decode_src_data(self, encoded_data, tokenizer, to_sentence=True):
        """ decode the list of list token idx to sentences """
        end_index = None if to_sentence else -2
        return utils.pipeline(self.decode_pipeline_for_src[:end_index], encoded_data, None, {'tokenizer': tokenizer},
                              False)

    def decode_tar_data(self, encoded_data, tokenizer, to_sentence=True):
        """ decode the list of list token idx to sentences """
        end_index = None if to_sentence else -1
        return utils.pipeline(self.decode_pipeline_for_tar[:end_index], encoded_data, None, {'tokenizer': tokenizer},
                              False)

    def calculate_bleu_for_encoded(self, src_encode_data, tar_encode_data, dataset=''):
        """ evaluate the BLEU according to the encoded src language data (list_of_list_token_idx)
                                and the target reference (list of sentences) """
        print('\nstart translating {} ...'.format(dataset))
        pred_encoded_data = self.evaluate(src_encode_data)

        pred_encoded_data = utils.convert_list_of_list_token_idx_2_string(pred_encoded_data)
        tar_encode_data = utils.convert_list_of_list_token_idx_2_string(tar_encode_data)

        print('calculating bleu ...')
        bleu = corpus_bleu(tar_encode_data, pred_encoded_data)

        print('{} bleu: {}'.format(dataset, bleu))
        return bleu

    def evaluate(self, list_of_list_src_token_idx):
        if self.model_params['use_beam_search']:
            return self.evaluate_encoded_beam_search(list_of_list_src_token_idx)
        return self.evaluate_encoded(list_of_list_src_token_idx)
