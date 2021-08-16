import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(levelname)s]%(asctime)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S'
)

logger = logging.getLogger('LOGGER_NAME')

