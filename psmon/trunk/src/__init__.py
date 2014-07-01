import logging as pslogging
from psmon import psconfig

# initialize logging for the package
pslogging.basicConfig(format=psconfig.LOG_FORMAT, level=psconfig.LOG_LEVEL_ROOT)
