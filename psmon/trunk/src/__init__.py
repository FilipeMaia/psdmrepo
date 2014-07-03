import logging as pslogging
from psmon import config

# initialize logging for the package
pslogging.basicConfig(format=config.LOG_FORMAT, level=config.LOG_LEVEL_ROOT)
