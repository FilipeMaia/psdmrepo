__all__ = []

from job                  import *
from acqiris              import *
from add_all_devices      import *
from add_available_data   import *
from add_elog             import *
from build_html           import *
from counter              import *
from cspad                import *
from epics_trend          import *
from evr                  import *
from ipimb                import *
from simple_stats         import *
from simple_trends        import *
from time_fiducials       import *
from store_report_results import *
from offbyone             import *

__version__ = '00.00.06'

import logging

#from event_process import *
#from event_process_lib import *
#from output_html import *
from device_config import *

logger = logging.getLogger('DataSummary')
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


