from collections import namedtuple

import tensorflow as tf


class TransformerOptimizer(object):
    Config = namedtuple('TransformerOptimizerConfig', ['warmup_steps', 'd_model'])

    def __init__(self, warmup_steps, d_model):
        self.warmup_steps = warmup_steps
        self.d_model = d_model

    def call(self, step_num):
        d_model = self.d_model
        warmup_steps = self.warmup_steps
        step_num = tf.cast(step_num, tf.float32)
        d_model = tf.convert_to_tensor(d_model, dtype=tf.float32)
        warmup_steps = tf.convert_to_tensor(warmup_steps, dtype=tf.float32)
        self.learning_rate = learning_rate = tf.pow(d_model, -0.5) * tf.minimum(tf.pow(step_num, -0.5), step_num * tf.pow(warmup_steps, -1.5))
        return tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-09)

    def summary(self):
        return tf.summary.scalar('learning_rate', self.learning_rate)
