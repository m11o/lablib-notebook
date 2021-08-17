import logging
from time import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s]%(asctime)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S'
)


logger = logging.getLogger(__name__)


def elapsed_time_logger(func):
    start = time()
    result = func()
    elapsed_time = time() - start

    logger.info('elapsed time[{0}]: {1}'.format(func.__name__, elapsed_time))

    return result

