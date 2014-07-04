import logging

### CONFIG KEYS FOR SERVER ###
RESET_REQ_STR = 'reset signal - %s'
RESET_REP_STR = 'reset signal recieved from %s'
ZMQ_TOPIC_DELIM_CHAR = '\0'
### CONFIG KEYS FOR LOGGING ###
LOG_LEVEL = 'INFO'
LOG_LEVEL_ROOT = logging.WARN
LOG_FORMAT = '[%(levelname)-8s] %(message)s' #'%(asctime)s:%(levelname)s:%(message)s'
### GENERAL APP CONFIG ###
APP_SERVER = 'localhost'
APP_PORT = 12301
APP_COMM_OFFSET = 1
APP_RATE = 5.0
APP_BUFFER = 10
APP_CLIENT = 'pyqt'
APP_RUN_DEFAULT = '12'
APP_EXP_DEFAULT = 'xppb0114'
APP_BIND_ATTEMPT = 20
### PYQT DEFAULT APPEARANCE CONFIG ###
PYQT_BORDERS = {'color': (150, 150, 150), 'width': 1.0}
PYQT_PLOT_PEN = None
PYQT_PLOT_SYMBOL = 'o'
PYQT_COLOR_PALETTE = 'thermal'
### MPL DEFAULT APPEARANCE CONFIG ###
MPL_COLOR_PALETTE = 'hot'
MPL_AXES_BKG_COLOR = 'w'
