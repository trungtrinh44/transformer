import os
import time
from collections import namedtuple

import tensorflow as tf

from model import TransformerEncoderClassifier
from utils import get_logger


def get_learning_rate(step_num, warmup_steps, d_model):
    step_num = tf.cast(step_num, tf.float32)
    d_model = tf.convert_to_tensor(d_model, dtype=tf.float32)
    warmup_steps = tf.convert_to_tensor(warmup_steps, dtype=tf.float32)
    return d_model * tf.minimum(tf.pow(step_num, -0.5), step_num * tf.pow(warmup_steps, -1.5))


def get_optimizer(learning_rate):
    return tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-09)


class ClassifyTrainer(object):
    Config = namedtuple('ClassifyTrainerConfig', ['path', 'warmup_steps', 'keep_n_train', 'keep_n_test', 'save_freq'])

    def __init__(self, model_config, trainer_config, name='ClassifyTrainer'):
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.name = name

    def build(self, restore_checkpoint=True):
        path = self.path = self.trainer_config.path
        train_path = self.train_path = os.path.join(path, 'train')
        self.test_path = os.path.join(path, 'test')
        self.log_path = os.path.join(path, 'log')
        self.logger = get_logger(self.log_path)
        # Create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        session = self.session = tf.Session(config=config)

        # Init x and y
        with tf.variable_scope(self.name):
            x = self.x = tf.placeholder(shape=(None, None), dtype=tf.int32, name='x')
            y = self.y = tf.placeholder(shape=(None,), dtype=tf.int32, name='y')
            seq_lens = self.seq_lens = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_lens')
            # Build optimizer
            global_step = self.global_step = tf.Variable(0, trainable=False, name='global_step')
            lr = self.lr = get_learning_rate(global_step, self.trainer_config.warmup_steps, self.model_config.ndims)
            optimizer = get_optimizer(lr)
        # Build train model
        model_train = self.model_train = TransformerEncoderClassifier(**self.model_config._asdict(), is_training=True, reuse=False)
        model_train.build(x.shape)
        output_train = self.output_train = model_train.call(x, seq_lens)

        # Build test model
        model_test = self.model_test = TransformerEncoderClassifier(**self.model_config._asdict(), is_training=False, reuse=True)
        model_test.build(x.shape)
        output_test = self.output_test = model_test.call(x, seq_lens)

        # Build loss value
        train_loss = self.train_loss = tf.losses.sparse_softmax_cross_entropy(y, output_train)
        self.test_loss = tf.losses.sparse_softmax_cross_entropy(y, output_test)

        # Build acc value
        self.train_acc = tf.reduce_mean(tf.cast(
            tf.equal(y, tf.argmax(output_train, axis=1, output_type=tf.int32)),
            dtype=tf.float32
        ))
        self.test_acc = tf.reduce_mean(tf.cast(
            tf.equal(y, tf.argmax(output_test, axis=1, output_type=tf.int32)),
            dtype=tf.float32
        ))
        self.train_op = optimizer.minimize(train_loss)

        # Train saver
        train_saver = self.train_saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=self.trainer_config.keep_n_train)
        # Test saver
        self.test_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_train.name),
                                         max_to_keep=self.trainer_config.keep_n_test)

        session.run(tf.global_variables_initializer())
        if restore_checkpoint:
            latest_checkpoint = tf.train.latest_checkpoint(train_path)
            if latest_checkpoint is not None:
                train_saver.restore(session, latest_checkpoint)

    def train_step(self, train_iter):
        """
            train_iter: iterator that return ((indices, seq_lens), labels) to feed to the model
        """
        t0 = time.time()
        for (indices, seq_lens), labels in train_iter:
            _, loss, acc, step = self.session.run([self.train_op, self.train_loss, self.train_acc, self.global_step],
                                                  feed_dict={self.x: indices, self.seq_lens: seq_lens, self.y: labels})
            self.logger.info("Step {:4d}: loss: {:05.5f}, acc: {:05.5f}, time {:05.2f}".format(step, loss, acc, time.time()-t0))
            if step % self.trainer_config.save_freq == 0:
                self.train_saver.save(self.session, os.path.join(self.train_path, 'model.cpkt'), step)

    def test_step(self, test_iter):
        """
            test_iter: iterator that return ((indices, seq_lens), labels) to feed to the model
        """
        t0 = time.time()
        total_loss, total_acc, total_len = 0.0, 0.0, 0
        step = self.session.run(self.global_step)
        self.test_saver.save(self.session, os.path.join(self.test_path, 'model.cpkt'), step)
        for (indices, seq_lens), labels in test_iter:
            loss, acc = self.session.run([self.test_loss, self.test_acc],
                                         feed_dict={self.x: indices, self.seq_lens: seq_lens, self.y: labels})
            total_loss += loss * len(labels)
            total_acc += acc * len(labels)
            total_len += len(labels)
            self.logger.info("Evaluate step: loss: {:05.5f}, acc: {:05.5f}, time {:05.2f}".format(loss, acc, time.time()-t0))
        total_loss /= total_len
        total_acc /= total_len
        self.logger.info("Evaluate step: total loss: {:05.5f}, total acc: {:05.5f}, time {:05.2f}".format(total_loss, total_acc, time.time()-t0))
