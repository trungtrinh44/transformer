import logging


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
