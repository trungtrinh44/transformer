from collections import namedtuple

import numpy as np
import tensorflow as tf


def scaled_dot_attention(Q, K, V, mask=None):
    """
        Q: [..., query_lengh, query_key_dim]
        K: [..., key_value_length, query_key_dim]
        V: [..., key_value_length, value_dim]
        mask: broadcastable to [..., query_length, key_value_length] use in the decoder to mask subsequent tokens.
    """
    assert Q.shape[-1] == K.shape[-1], "Q and K must have a last dimension of same size"
    d_k = tf.cast(Q.shape[-1], tf.float32)
    att = tf.matmul(Q, K, transpose_b=True) / d_k  # [..., query_length, key_value_length]
    if mask is not None:
        att += mask * -1e9
    att = tf.nn.softmax(att, axis=-1)  # [..., query_length, key_value_length]
    att = tf.matmul(att, V)  # [..., query_lengh, value_dim]
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
    pe = np.concatenate((np.sin(pe[:, 0::2]), np.cos(pe[:, 1::2])), axis=1)
    return tf.convert_to_tensor(pe[np.newaxis, :], dtype=tf.float32)


class MultiheadAttention(object):
    def __init__(self, nheads, qk_dims, v_dims, ndims, is_training=True, reuse=tf.AUTO_REUSE, name='MultiheadAttention'):
        """
            nheads: number of attention heads
            qk_dims: query and key dimension size
            v_dims: value dimension size
            ndims: output dimension size
            is_training: is training or not
            reuse: reuse variable or not
            name: model name
        """
        self.nheads = nheads
        self.qk_dims = qk_dims
        self.v_dims = v_dims
        self.ndims = ndims
        self.is_training = is_training
        self.reuse = reuse
        self.name = name

    def build(self, q_shape, k_shape, v_shape):
        with tf.variable_scope(self.name, reuse=self.reuse):
            weights = {}
            with tf.variable_scope('Query', reuse=self.reuse):
                weights['w_q'] = tf.get_variable(name='weight', shape=(1, q_shape[-1], self.qk_dims * self.nheads))
            with tf.variable_scope('Key', reuse=self.reuse):
                weights['w_k'] = tf.get_variable(name='weight', shape=(1, k_shape[-1], self.qk_dims * self.nheads))
            with tf.variable_scope('Value', reuse=self.reuse):
                weights['w_v'] = tf.get_variable(name='weight', shape=(1, v_shape[-1], self.v_dims * self.nheads))
            with tf.variable_scope('Linear', reuse=self.reuse):
                weights['linear'] = tf.get_variable(name='weight', shape=(1, self.v_dims * self.nheads, self.ndims))
            self.weights = weights

    def split_heads(self, x, batch_size, att_dims):
        """
            x: Tensor of shape [batch_size, seq_len, nheads * att_dims]
        """
        x = tf.reshape(x, (batch_size, -1, self.nheads, att_dims))
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        return x

    def call(self, Q, K, V, mask=None):
        weights = self.weights
        with tf.name_scope(self.name):
            batch_size = tf.shape(Q)[0]
            with tf.name_scope('Query'):
                Q = tf.nn.conv1d(Q, weights['w_q'], 1, 'VALID')  # [batch_size, q_lens, self.qk_dims * self.nheads]
                Q = self.split_heads(Q, batch_size, self.qk_dims)  # [batch_size, nheads, q_lens, qk_dims]
            with tf.name_scope('Key'):
                K = tf.nn.conv1d(K, weights['w_k'], 1, 'VALID')  # [batch_size, k_lens, self.qk_dims * self.nheads]
                K = self.split_heads(K, batch_size, self.qk_dims)  # [batch_size, nheads, k_lens, qk_dims]
            with tf.name_scope('Value'):
                V = tf.nn.conv1d(V, weights['w_v'], 1, 'VALID')  # [batch_size, k_lens, self.v_dims * self.nheads]
                V = self.split_heads(V, batch_size, self.v_dims)  # [batch_size, nheads, k_lens, v_dims]
            with tf.name_scope('Scaled_Dot_Attention'):
                att_heads = scaled_dot_attention(Q, K, V, mask)  # [batch_size, nheads, q_lens, v_dims]
                att_heads = tf.transpose(att_heads, perm=(0, 2, 1, 3))  # [batch_size, q_lens, nheads, v_dims]
                att_heads = tf.reshape(att_heads, (batch_size, -1, self.v_dims * self.nheads))  # [batch_size, q_lens, v_dims * nheads]
            with tf.name_scope('Linear'):
                outputs = tf.nn.conv1d(att_heads, weights['linear'], 1, 'VALID')
        return outputs


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
                    prev_shape = layer['size']

    def call(self, inputs):
        with tf.name_scope(self.name):
            outputs = inputs
            for idx, layer in enumerate(self.layers):
                with tf.name_scope('layer_{}'.format(idx)):
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

    def call(self, inputs, variance_epsilon=1e-12):
        with tf.name_scope(self.name):
            inputs_shape = inputs.shape
            inputs_rank = inputs_shape.ndims
            begin_norm_axis = self.begin_norm_axis
            if begin_norm_axis < 0:
                begin_norm_axis = inputs_rank + begin_norm_axis
            norm_axes = list(range(begin_norm_axis, inputs_rank))
            mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)
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

            self.mult_att = MultiheadAttention(self.nheads, self.att_dims, self.att_dims, self.ndims, self.is_training, self.reuse)
            self.mult_att.build(input_shape, input_shape, input_shape)

            self.layer_norm_1 = LayerNorm(begin_norm_axis=-1, trainable=self.is_training, reuse=self.reuse, name='LayerNorm_1')
            self.layer_norm_1.build(ts)

            self.feed_forward = PositionwiseFF([
                {'size': self.ff_ndims, 'activation': tf.nn.relu},
                {'size': self.ndims}
            ], is_training=self.is_training, reuse=self.reuse)
            self.feed_forward.build(ts)

            self.layer_norm_2 = LayerNorm(begin_norm_axis=-1, trainable=self.is_training, reuse=self.reuse, name='LayerNorm_2')
            self.layer_norm_2.build(ts)

    def call(self, inputs, mask):
        with tf.name_scope(self.name):
            # Multi-Head Attention
            outputs = self.mult_att.call(inputs, inputs, inputs, mask)
            if self.is_training and self.dropout > 0.0:
                outputs = tf.nn.dropout(outputs, keep_prob=1-self.dropout)
            outputs += inputs
            outputs = self.layer_norm_1.call(outputs, 1e-6)
            # Position-wise feed forward
            inputs = outputs
            outputs = self.feed_forward.call(inputs)
            if self.is_training and self.dropout > 0.0:
                outputs = tf.nn.dropout(outputs, keep_prob=1-self.dropout)
            outputs += inputs
            outputs = self.layer_norm_2.call(outputs, 1e-6)
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

    def build(self, input_shape, pretrain_wv=None, train_wv=True):
        """
        pretrain_wv: None of a numpy array containing pretrain word vector
        train_wv: use only when pretrain_wv is available
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            # Embedding layer
            with tf.variable_scope('EmbeddingLayer', reuse=self.reuse):
                if pretrain_wv is None:
                    self.embedding_weight = tf.get_variable(name='embedding_weight', shape=(self.vocab_size, self.ndims), initializer=tf.random_uniform_initializer(-1.0, 1.0))
                else:
                    assert pretrain_wv.shape[0] == self.vocab_size and pretrain_wv.shape[1] == self.ndims, "Pretrained word vec shape is incompatible with the model configuration"
                    self.embedding_weight = tf.get_variable(name='embedding_weight', shape=(self.vocab_size, self.ndims), initializer=tf.constant_initializer(pretrain_wv), trainable=train_wv)
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
        with tf.name_scope(self.name):
            with tf.name_scope('EmbeddingLayer'):
                embedding = tf.nn.embedding_lookup(self.embedding_weight, inputs)
                input_len = tf.shape(inputs)[1]
                outputs = embedding * (self.ndims**0.5) + self.pe_weight[:, :input_len, :]
            with tf.name_scope('EncoderLayers'):
                mask = tf.cast(tf.equal(inputs, 0), tf.float32, name='mask')  # 0 is the padding value
                mask = mask[:, tf.newaxis, tf.newaxis, :]
                for layer in self.layers:
                    outputs = layer.call(outputs, mask)
        return outputs


class TransformerEncoderClassifier(TransformerEncoder):
    Config = namedtuple('TransformerEncoderClassifierConfig', TransformerEncoder.Config._fields + ('n_classes',))

    def __init__(self, n_classes, ndims=512, nheads=8, nlayers=6, ff_ndims=2048, vocab_size=2**13, maxlen=128, dropout=0.5, is_training=True, reuse=tf.AUTO_REUSE, name='TransformerEncoderClassifier'):
        super().__init__(ndims, nheads, nlayers, ff_ndims, vocab_size, maxlen, dropout, is_training, reuse, name)
        self.n_classes = n_classes

    def build(self, input_shape, pretrained_wv=None, train_wv=True):
        super().build(input_shape, pretrained_wv, train_wv)
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('Classifier', reuse=self.reuse):
                self.cls_weight = {}
                self.cls_weight['W'] = tf.get_variable(name='W', shape=(self.ndims, self.n_classes))
                self.cls_weight['b'] = tf.get_variable(name='b', shape=(self.n_classes,))

    def call(self, inputs, seq_lens):
        outputs = super().call(inputs, seq_lens)
        with tf.name_scope(self.name):
            with tf.name_scope('Classifier'):
                # use the output at the first time step for classification
                outputs = tf.nn.xw_plus_b(outputs[:, 0], self.cls_weight['W'], self.cls_weight['b'])
        return outputs
