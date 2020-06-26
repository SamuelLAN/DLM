import tensorflow as tf
from lib.tf_models.transformer import positional_encoding, EncoderLayer, DecoderLayer
from lib.tf_models.transformer import Transformer as BaseTransformer

keras = tf.keras
layers = keras.layers


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size,
                 maximum_position_encoding, drop_rate=0.1, lan_vocab_size=2):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # token embedding and position embedding
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        # language embedding; vocab_size = 4 because there are two for <start> and <end>
        self.lan_embedding = layers.Embedding(lan_vocab_size, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff, drop_rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(drop_rate)

    def call(self, x, lan_x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        # add lan embeddings
        lan_x = self.lan_embedding(lan_x)
        x += lan_x

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size,
                 maximum_position_encoding, drop_rate=0.1, emb_layer=None, lan_emb_layer=None, lan_vocab_size=2):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # token embedding and position embedding
        self.embedding = layers.Embedding(target_vocab_size, d_model) \
            if isinstance(emb_layer, type(None)) else emb_layer
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        # language embedding; vocab_size = 4 because there are two for <start> and <end>
        self.lan_embedding = layers.Embedding(lan_vocab_size, d_model) \
            if isinstance(lan_emb_layer, type(None)) else lan_emb_layer

        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff, drop_rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(drop_rate)

    def call(self, x, enc_output, lan_y, pos_y, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x += pos_y

        # add lan embedding
        lan_y = self.lan_embedding(lan_y)
        x += lan_y

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(BaseTransformer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size,
                 target_vocab_size, max_pe_input, max_pe_target, drop_rate=0.1, share_emb=False, share_final=False,
                 lan_vocab_size=2, encoder=None):
        super(Transformer, self).__init__(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size,
                                          max_pe_input, max_pe_target, drop_rate, share_emb, share_final)

        self.__share_final = share_final

        if not isinstance(encoder, type(None)):
            self.encoder = encoder
        else:
            self.encoder = Encoder(num_layers, d_model, num_heads, d_ff,
                                   input_vocab_size, max_pe_input, drop_rate, lan_vocab_size)

        emb_layer = self.encoder.embedding if share_emb else None
        lan_emb_layer = self.encoder.lan_embedding if share_emb else None
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff,
                               target_vocab_size, max_pe_target, drop_rate, emb_layer, lan_emb_layer, lan_vocab_size)

        self.final_layer = layers.Dense(target_vocab_size, activation='softmax')

    def call(self, inputs, training=None, mask=None, show_attention_weight=False):
        enc_inp, enc_inp_lan, dec_inp, dec_inp_lan, dec_inp_pos = inputs

        if isinstance(mask, type(None)):
            enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(enc_inp, dec_inp)
        else:
            enc_padding_mask, look_ahead_mask, dec_padding_mask = mask

        enc_output = self.encoder(enc_inp, enc_inp_lan, training,
                                  enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            dec_inp, enc_output, dec_inp_lan, dec_inp_pos, training, look_ahead_mask, dec_padding_mask)

        if not self.__share_final:
            final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        else:
            W = tf.squeeze(tf.transpose(self.encoder.embedding.weights), axis=-1)
            final_output = tf.matmul(dec_output, W)
            final_output = tf.nn.softmax(final_output, axis=-1)

        if show_attention_weight:
            return final_output, attention_weights
        return final_output

    def evaluate_list_of_list_token_idx(self, list_of_list_input_token_idx, tar_start_token_idx, tar_end_token_idx,
                                        max_tar_seq_len=60, verbose=0, show_attention_weight=False):
        """
        Evaluate a sentence
        :params
            list_input_token_idx (list): [12, 43, 2, 346, 436, 87, 876],
                        # correspond to ['He', 'llo', ',', 'I', 'am', 'stu', 'dent'],
            tar_start_token_idx (int): idx of target <start> token
            tar_end_token_idx (int): idx of target <end> token
            max_tar_seq_len: max token num of target sentences
        :return
            list_target_token_idx (list): [12, 43, 2, 346, 436, 87, 876],
                        # correspond to ['He', 'llo', ',', 'I', 'am', 'stu', 'dent'],
        """
        # shape: (batch_size, len of list_input_token_idx )
        batch_size = len(list_of_list_input_token_idx)
        outputs = [{'output': [tar_start_token_idx], 'index': i} for i in range(batch_size)]

        seq_len = 1
        ret = []

        while len(outputs) and seq_len < max_tar_seq_len:
            if verbose and seq_len % 2 == 0:
                progress = float(seq_len) / max_tar_seq_len * 100.
                print('\rprogress: %.2f%% ' % progress, end='')

            # get input
            encoder_input = np.array([list_of_list_input_token_idx[val['index']] for val in outputs])
            decoder_input = np.array(list(map(lambda x: x['output'], outputs)))

            # predictions.shape == (top_k, seq_len, vocab_size)
            if show_attention_weight:
                predictions, attention_weights = self.call([encoder_input, decoder_input],
                                                           training=False, show_attention_weight=show_attention_weight)
            else:
                predictions = self.call([encoder_input, decoder_input],
                                        training=False, show_attention_weight=show_attention_weight)
                attention_weights = {}

            predictions = predictions[:, -1]

            last_token_idx = np.argmax(predictions, axis=-1)

            tmp_outputs = []
            for i, val in enumerate(outputs):
                val['output'] += [last_token_idx[i]]
                if last_token_idx[i] == tar_end_token_idx or seq_len >= max_tar_seq_len:

                    attentions = {}
                    for k, v in attention_weights.items():
                        attentions[k] = v[i]

                    val['attentions'] = attentions
                    ret.append(val)
                else:
                    tmp_outputs.append(val)

            outputs = tmp_outputs
            seq_len += 1

        if len(outputs):
            ret += outputs

        ret.sort(key=lambda x: x['index'])

        if show_attention_weight:
            return list(map(lambda x: x['output'], ret)), list(map(lambda x: x['attentions'], ret))
        return list(map(lambda x: x['output'], ret)), []
