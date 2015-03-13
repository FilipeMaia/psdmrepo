import logging

from job import *
#from event_process import *
#from event_process_lib import *
#from output_html import *
from evplib import *
from device_config import *

logger = logging.getLogger('data_summary')
logger.setLevel(logging.DEBUG)

#fh = logging.FileHandler('data_summary.log')
#fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
#fh.setFormatter(formatter)
ch.setFormatter(formatter)

#logger.addHandler(fh)
logger.addHandler(ch)


def set_logger_level(lvl):
    logger.setLevel( getattr(logging,lvl) )
#    fh.setLevel( getattr(logging,lvl) )
    ch.setLevel( getattr(logging,lvl) )
    return

def logger_flush():
#    fh.flush()
    return
