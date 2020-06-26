import tensorflow as tf
import numpy as np

keras = tf.keras
layers = keras.layers


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    with tf.name_scope('positional_encoding'):
        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    with tf.name_scope('padding_mask'):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    with tf.name_scope('look_ahead_mask'):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    with tf.name_scope('scaled_dot_product_attention'):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to th
        # e scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, d_ff):
    return keras.Sequential([
        layers.Dense(d_ff, activation='relu'),  # (batch_size, seq_len, d_ff)
        layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, drop_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, d_ff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)
        self.dropout3 = layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


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

    def call(self, x, enc_output, lan_y, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

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


class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size,
                 target_vocab_size, max_pe_input, max_pe_target, drop_rate=0.1, share_emb=False, share_final=False,
                 lan_vocab_size=2):
        super(Transformer, self).__init__()

        self.__share_final = share_final

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff,
                               input_vocab_size, max_pe_input, drop_rate, lan_vocab_size)

        emb_layer = self.encoder.embedding if share_emb else None
        lan_emb_layer = self.encoder.lan_embedding if share_emb else None
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff,
                               target_vocab_size, max_pe_target, drop_rate, emb_layer, lan_emb_layer, lan_vocab_size)

        self.final_layer = layers.Dense(target_vocab_size, activation='softmax')

    @staticmethod
    def create_masks(inp, tar):
        with tf.name_scope('mask'):
            # Encoder padding mask
            enc_padding_mask = create_padding_mask(inp)

            # Used in the 2nd attention block in the decoder.
            # This padding mask is used to mask the encoder outputs.
            dec_padding_mask = create_padding_mask(inp)

            # Used in the 1st attention block in the decoder.
            # It is used to pad and mask future tokens in the input received by
            # the decoder.
            look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
            dec_target_padding_mask = create_padding_mask(tar)
            combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def call(self, inputs, training=None, mask=None):
        enc_inp, dec_inp = inputs

        if isinstance(mask, type(None)):
            enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(enc_inp, dec_inp)
        else:
            enc_padding_mask, look_ahead_mask, dec_padding_mask = mask

        enc_inp_lan = tf.zeros_like(enc_inp)
        enc_output = self.encoder(enc_inp, enc_inp_lan, training,
                                  enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_inp_lan = tf.ones_like(dec_inp)
        dec_output, attention_weights = self.decoder(
            dec_inp, enc_output, dec_inp_lan, training, look_ahead_mask, dec_padding_mask)

        if not self.__share_final:
            final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        else:
            W = tf.squeeze(tf.transpose(self.encoder.embedding.weights), axis=-1)
            final_output = tf.matmul(dec_output, W)
            final_output = tf.nn.softmax(final_output, axis=-1)

        # return final_output, attention_weights
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

    def beam_search_list_token_idx(self,
                                   list_input_token_idx,
                                   tar_start_token_idx,
                                   tar_end_token_idx,
                                   max_tar_seq_len=60,
                                   top_k=1,
                                   get_random=False):
        """
        Use beam search to decode
        :param
            list_input_token_idx (list): [[12, 43, 2, 346, 436, 87, 876], ..., ]
                        # correspond to [['He', 'llo', ',', 'I', 'am', 'stu', 'dent'], ..., ]
            tar_start_token_idx (int): idx of target <start> token
            tar_end_token_idx (int): idx of target <end> token
            max_tar_seq_len (int): max token num of target sentences
            top_k (int): Choose the last token from top K.
            get_random (bool): whether to choose the result from the final top_k results randomly
        :return:
            list_target_token_idx (list): [12, 43, 2, 346, 436, 87, 876],
                        # correspond to ['He', 'llo', ',', 'I', 'am', 'stu', 'dent'],
        """

        # shape: (1, len of list_input_token_idx )
        encoder_input = tf.reshape(list_input_token_idx, (1, -1))

        # the first word to the transformer should be the target start token.
        decoder_input = [tar_start_token_idx]
        output = tf.expand_dims(decoder_input, 0)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        # predictions, attention_weights = self.call([encoder_input, output], training=False)
        predictions = self.call([encoder_input, output], training=False)

        # log the probability of the last word
        predictions = np.log(predictions[0, -1] + 0.0001)

        # get the top_k result of the last word
        log_probs = [(log_prob, j) for j, log_prob in enumerate(predictions)]
        log_probs.sort(reverse=True)
        log_probs = log_probs[:top_k]

        # list for saving the top_k results
        top_k_outputs = list(map(lambda x: {'output': [tar_start_token_idx, x[1]], 'log_prob': x[0]}, log_probs))
        seq_len = 2
        ret = []

        # check if the last word is <end>
        for i in range(top_k - 1, -1, -1):
            val = top_k_outputs[i]
            if val['output'][-1] == tar_end_token_idx:
                ret.append(val)
                del top_k_outputs[i]

        while len(top_k_outputs) and seq_len < max_tar_seq_len:
            # get decoder input
            encoder_input = np.array([list_input_token_idx for i in range(len(top_k_outputs))])
            decoder_input = np.array(list(map(lambda x: x['output'], top_k_outputs)))

            # predictions.shape == (top_k, seq_len, vocab_size)
            predictions = self.call([encoder_input, decoder_input], training=False)
            predictions = np.log(predictions[:, -1] + 0.0001)

            # concat the last word of the new predictions to the top_k_outputs
            #    and calculate the total log_prob
            tmp_outputs = []
            for i, val in enumerate(top_k_outputs):
                tmp_outputs += [{'output': val['output'] + [j], 'log_prob': val['log_prob'] + log_prob}
                                for j, log_prob in enumerate(predictions[i])]

            # compare normalized log_prob and get top_k
            tmp_outputs += ret
            tmp_outputs.sort(key=lambda x: -x['log_prob'] / float(len(x['output'])))
            tmp_outputs = tmp_outputs[:top_k]

            # whether to continue prediction
            top_k_outputs = []
            for val in tmp_outputs:

                # if end, them
                last_word_id = val['output'][-1]
                if last_word_id == tar_end_token_idx:
                    ret.append(val)

                # if not end
                else:
                    top_k_outputs.append(val)

            seq_len += 1

        if len(ret) < top_k:
            ret += top_k_outputs[: top_k - len(ret)]
            ret.sort(key=lambda x: -x['log_prob'] / float(len(x['output'])))

        index = 0 if not get_random else np.random.randint(0, top_k)
        return ret[index]['output']

    def beam_search_list_of_list_token_idx(self, list_of_list_input_token_idx,
                                           tar_start_token_idx, tar_end_token_idx, max_tar_seq_len=60,
                                           top_k=1, get_random=False):
        """ Beam search list of encoded sentences """
        predictions = []

        length = len(list_of_list_input_token_idx)
        for i, x in enumerate(list_of_list_input_token_idx):
            if i % 5 == 0 and length > 20:
                progress = float(i + 1) / length * 100.
                print('\rprogress: %.2f%% ' % progress, end='')

            predictions.append(self.beam_search_list_token_idx(
                x, tar_start_token_idx, tar_end_token_idx, max_tar_seq_len, top_k, get_random))

        return predictions
