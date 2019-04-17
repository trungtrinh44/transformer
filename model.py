from collections import namedtuple

import numpy as np
import tensorflow as tf


def scaled_dot_attention(Q, K, V, mask=None):
    """
        Q: [batch_size, query_lengh, query_key_dim]
        K: [batch_size, key_value_length, query_key_dim]
        V: [batch_size, key_value_length, value_dim]
        mask: broadcastable to [batch_size, query_length, key_value_length] use in the decoder to mask subsequent tokens.
    """
    assert Q.shape[-1] == K.shape[-1], "Q and K must have a last dimension of same size"
    d_k = Q.shape[-1]
    att = tf.matmul(Q, tf.transpose(K, (0, 2, 1))) / d_k  # [batch_size, query_length, key_value_length]
    if mask is not None:
        att = att * mask + (1 - mask) * 1e-20
    att = tf.nn.softmax(att, axis=-1)  # [batch_size, query_length, key_value_length]
    att = tf.matmul(att, V)  # [batch_size, query_lengh, value_dim]
    return att


def sinusoid_positional_encoding(npos, ndims):
    """
        npos: number of positions
        ndims: number of dimensions 
    """
    cols = np.arange(0, ndims, 1)  # [ndims]
    cols = cols // 2  # [ndims]
    cols = 2 * cols / ndims  # [ndims]
    cols = np.power(10000, cols)  # [ndims]
    pe = np.arange(0, npos, 1)[:, np.newaxis]  # [npos, 1]
    pe = pe / cols
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return tf.convert_to_tensor(pe, dtype=tf.float32)


class MultiheadAttention(object):
    def __init__(self, nheads, qk_dims, v_dims, is_training=True, reuse=tf.AUTO_REUSE, name='MultiheadAttention'):
        """
            nheads: number of attention heads
            qk_dims: query and key dimension size
            v_dims: value dimension size
            is_training: is training or not
            reuse: reuse variable or not
            name: model name
        """
        self.nheads = nheads
        self.qk_dims = qk_dims
        self.v_dims = v_dims
        self.is_training = is_training
        self.reuse = reuse
        self.name = name

    def build(self, q_shape, k_shape, v_shape):
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.head_params = []
            for i in range(self.nheads):
                with tf.variable_scope('head_{}'.format(i), reuse=self.reuse):
                    weights = {}
                    weights['w_q'] = tf.get_variable(name='w_q', shape=(q_shape[-1], self.qk_dims))
                    weights['w_k'] = tf.get_variable(name='w_k', shape=(k_shape[-1], self.qk_dims))
                    weights['w_v'] = tf.get_variable(name='w_v', shape=(v_shape[-1], self.v_dims))
                    self.head_params.append(weights)

    def call(self, Q, K, V, mask=None):
        with tf.variable_scope(self.name, reuse=self.reuse):
            att_heads = []
            for i in range(self.nheads):
                with tf.variable_scope('head_{}'.format(i), reuse=self.reuse):
                    weights = self.head_params[i]
                    Q_i = tf.matmul(Q, weights['w_q'], name='Q')
                    K_i = tf.matmul(K, weights['w_k'], name='K')
                    V_i = tf.matmul(V, weights['w_v'], name='V')
                    head = scaled_dot_attention(Q_i, K_i, V_i, mask)
                    att_heads.append(head)
            att_heads = tf.concat(att_heads, axis=-1, name='att_heads')
        return att_heads


class PositionwiseFF(object):
    def __init__(self, layers, is_training=True, reuse=tf.AUTO_REUSE, name='PositionwiseFF'):
        """
            layers: list of {'size','activation'}
        """
        self.layers = layers
        self.is_training = is_training
        self.reuse = reuse
        self.name = name

    def build(self, input_shape):
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.weights = []
            prev_shape = input_shape[-1]
            for idx, layer in enumerate(self.layers):
                with tf.variable_scope('layer_{}'.format(idx), reuse=self.reuse):
                    weight = {}
                    weight['W'] = tf.get_variable(name='W', shape=(1, prev_shape, layer['size']))
                    weight['b'] = tf.get_variable(name='b', shape=(1, layer['size']))
                    self.weights.append(weight)

    def call(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            outputs = inputs
            for idx, layer in enumerate(self.layers):
                with tf.variable_scope('layer_{}'.format(idx), reuse=self.reuse):
                    weight = self.weights[idx]
                    act = layer.get('activation')
                    outputs = tf.nn.conv1d(outputs, weight['W'], 1, 'VALID', True, 'NWC') + weight['b']
                    if act is not None:
                        outputs = act(outputs)
        return outputs


class LayerNorm(object):
    def __init__(self, center=True, scale=True, reuse=tf.AUTO_REUSE, trainable=True,
                 begin_norm_axis=1, begin_params_axis=-1, name="LayerNorm"):
        self.center = center
        self.scale = scale
        self.reuse = reuse
        self.trainable = trainable
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.name = name
        self.trainable = trainable

    def build(self, inputs_shape):
        with tf.variable_scope(self.name, reuse=self.reuse):
            params_shape = inputs_shape[self.begin_params_axis:]
            beta, gamma = None, None
            if self.center:
                beta = tf.get_variable(name='beta', shape=params_shape, dtype=tf.float32,
                                       initializer=tf.zeros_initializer(),
                                       trainable=self.trainable)
            if self.scale:
                gamma = tf.get_variable(name='gamma', shape=params_shape, dtype=tf.float32,
                                        initializer=tf.ones_initializer(),
                                        trainable=self.trainable)
        self.beta, self.gamma = beta, gamma

    def call(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            inputs_shape = inputs.shape
            inputs_rank = inputs_shape.ndims
            if begin_norm_axis < 0:
                begin_norm_axis = inputs_rank + begin_norm_axis
            norm_axes = list(range(begin_norm_axis, inputs_rank))
            mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)
            variance_epsilon = 1e-12
            outputs = tf.nn.batch_normalization(inputs, mean, variance, offset=self.beta, scale=self.gamma, variance_epsilon=variance_epsilon)
        return outputs


class EncoderLayer(object):
    def __init__(self, ndims=512, nheads=8, ff_ndims=2048, dropout=0.5, is_training=True, reuse=tf.AUTO_REUSE, name='EncoderLayer'):
        """
            ndims: number of dimensions
            nheads: number of attention heads
            ff_dims: feed forward inner layer dimension
            dropout: dropout rate
            is_training: is training or not
            reuse: reuse variable or not
            name: model name
        """
        assert ndims % nheads == 0, "ndims ({}) must be divisible by nheads ({})".format(ndims, nheads)
        self.ndims = ndims
        self.nheads = nheads
        self.ff_ndims = ff_ndims
        self.is_training = is_training
        self.dropout = dropout
        self.name = name
        self.att_dims = ndims // nheads
        self.reuse = reuse

    def build(self, input_shape):
        with tf.variable_scope(self.name, reuse=self.reuse):
            ts = tf.TensorShape((None, None, self.ndims))

            self.mult_att = MultiheadAttention(self.nheads, self.att_dims, self.att_dims, self.is_training, self.reuse)
            self.mult_att.build(input_shape, input_shape, input_shape)

            self.layer_norm_1 = LayerNorm(begin_norm_axis=-1, trainable=self.is_training, reuse=self.reuse)
            self.layer_norm_1.build(ts)

            self.feed_forward = PositionwiseFF([
                {'size': self.ff_ndims, 'activation': tf.nn.relu},
                {'size': self.ndims}
            ], is_training=self.is_training, reuse=self.reuse)
            self.feed_forward.build(ts)

            self.layer_norm_2 = LayerNorm(begin_norm_axis=-1, trainable=self.is_training, reuse=self.reuse)
            self.layer_norm_2.build(ts)

    def call(self, inputs, mask):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # Multi-Head Attention
            outputs = self.mult_att.call(inputs, inputs, inputs, mask)
            if self.is_training and self.dropout > 0.0:
                outputs = tf.nn.dropout(outputs, rate=self.dropout)
            outputs += inputs
            outputs = self.layer_norm_1.call(outputs)
            # Position-wise feed forward
            inputs = outputs
            outputs = self.feed_forward.call(inputs)
            if self.is_training and self.dropout > 0.0:
                outputs = tf.nn.dropout(outputs, rate=self.dropout)
            outputs += inputs
            outputs = self.layer_norm_2.call(outputs)
        return outputs


class TransformerEncoder(object):
    Config = namedtuple('TransformerEncoderConfig', ['ndims', 'nheads', 'nlayers', 'ff_ndims', 'vocab_size', 'maxlen', 'dropout'])

    def __init__(self, ndims=512, nheads=8, nlayers=6, ff_ndims=2048, vocab_size=2**13, maxlen=128, dropout=0.5, is_training=True, reuse=tf.AUTO_REUSE, name='TransformerEncoder'):
        """
            ndims: number of dimensions
            nheads: number of attention heads
            nlayers: number of layers
            ff_dims: feed forward inner layer dimension
            dropout: dropout rate
            is_training: is training or not
            reuse: reuse variable or not
            name: model name
        """
        self.ndims = ndims
        self.nheads = nheads
        self.nlayers = nlayers
        self.ff_ndims = ff_ndims
        self.is_training = is_training
        self.dropout = dropout
        self.name = name
        self.reuse = reuse
        self.vocab_size = vocab_size
        self.maxlen = maxlen

    def build(self, input_shape):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # Embedding layer
            with tf.variable_scope('EmbeddingLayer', reuse=self.reuse):
                self.embedding_weight = tf.get_variable(name='embedding_weight', shape=(self.vocab_size, self.ndims), initializer=tf.random_uniform_initializer(-1.0, 1.0))
                self.pe_weight = sinusoid_positional_encoding(self.maxlen, self.ndims)
            # Stack of layer
            self.layers = []
            with tf.variable_scope('EncoderLayers', reuse=self.reuse):
                for i in range(self.nlayers):
                    layer = EncoderLayer(self.ndims, self.nheads, self.ff_ndims, self.dropout, self.is_training, self.reuse, 'EncoderLayer_{}'.format(i))
                    layer.build(tf.TensorShape((None, None, self.ndims)))
                    self.layers.append(layer)

    def call(self, inputs, seq_lens):
        """
            inputs: [batch_size, seq_len] of word indices.
            seq_lens: [batch_size] of sequence lengths
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('EmbeddingLayer', reuse=self.reuse):
                embedding = tf.nn.embedding_lookup(self.embedding_weight, inputs)
                input_len = tf.shape(inputs)[1]
                pe = tf.nn.embedding_lookup(self.pe_weight, tf.range(0, input_len, 1, dtype=tf.int32))[tf.newaxis, :]
                outputs = embedding + pe
            with tf.variable_scope('EncoderLayers', reuse=self.reuse):
                mask = tf.sequence_mask(seq_lens, dtype=tf.float32)
                for layer in self.layers:
                    outputs = layer.call(outputs, mask)
        return outputs


class TransformerEncoderClassifier(TransformerEncoder):
    Config = namedtuple('TransformerEncoderClassifierConfig', TransformerEncoder.Config._fields + ('n_classes',))

    def __init__(self, n_classes, ndims=512, nheads=8, nlayers=6, ff_ndims=2048, vocab_size=2**13, maxlen=128, dropout=0.5, is_training=True, reuse=tf.AUTO_REUSE, name='TransformerEncoderClassifier'):
        super().__init__(ndims, nheads, nlayers, ff_ndims, vocab_size, maxlen, dropout, is_training, reuse, name)
        self.n_classes = n_classes

    def build(self, input_shape):
        super().build(input_shape)
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('Classifier', reuse=self.reuse):
                self.cls_weight = {}
                self.cls_weight['W'] = tf.get_variable(name='W', shape=(self.ndims, self.n_classes))
                self.cls_weight['b'] = tf.get_variable(name='b', shape=(self.n_classes,))

    def call(self, inputs, seq_lens):
        outputs = super().call(inputs, seq_lens)
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('Classifier', reuse=self.reuse):
                # use the output at the first time step for classification
                outputs = tf.nn.xw_plus_b(outputs[:, 0], self.cls_weight['W'], self.cls_weight['b'])
        return outputs
