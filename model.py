import tensorflow as tf
import numpy as np

def scaled_dot_attention(Q, K, V, mask=None):
    """
        Q: [batch_size, query_lengh, query_key_dim]
        K: [batch_size, key_value_length, query_key_dim]
        V: [batch_size, key_value_length, value_dim]
        mask: broadcastable to [batch_size, query_length, key_value_length] use in the decoder to mask subsequent tokens.
    """
    assert Q.shape[-1] == K.shape[-1], "Q and K must have a last dimension of same size"
    d_k = Q.shape[-1]
    att = tf.matmul(Q, tf.transpose(K, (0,2,1))) / d_k # [batch_size, query_length, key_value_length]
    if mask is not None:
        att = att * mask + (1 - mask) * 1e-20
    att = tf.nn.softmax(att, axis=-1) # [batch_size, query_length, key_value_length]
    att = tf.matmul(att, V) # [batch_size, query_lengh, value_dim]
    return att

def sinusoid_positional_encoding(npos, ndims):
    """
        npos: number of positions
        ndims: number of dimensions 
    """
    cols = np.arange(0, ndims, 1) # [ndims]
    cols = cols // 2 # [ndims]
    cols = 2 * cols / ndims # [ndims]
    cols = np.power(10000, cols) # [ndims]
    pe = np.arange(0, npos, 1)[:, np.newaxis] # [npos, 1]
    pe = pe / cols
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return tf.convert_to_tensor(pe, dtype=tf.float32)

# class Transformer(object):
#     def __init__(self, ndims, nhead, nlayers, )