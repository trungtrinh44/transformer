import logging

import numpy as np


def get_logger(filename):
    """Return a logger instance that writes in filename
    Args:
        filename: (string) path to log.txt
    Returns:
        logger: (instance of logger)
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    if logger.handlers:
        [handler.close() for handler in logger.handlers]
        logger.handlers = []
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def get_batch_classifier(data, batch_size, shuffle):
    """
        data: a numpy array of (features, label)
    """
    if shuffle:
        np.random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        seq_lens = np.array([len(x) for x, _ in batch], dtype=np.int32)
        labels = np.array([x for _, x in batch], dtype=np.int32)
        indices = np.zeros(shape=(len(batch), seq_lens.max()), dtype=np.int32)
        for seq, _ in batch:
            indices[:len(seq)] = seq
        yield (indices, seq_lens), labels
