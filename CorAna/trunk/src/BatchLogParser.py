#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module BatchLogParser...
#
#------------------------------------------------------------------------

"""Extracts required information from batch log files

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

from ConfigParametersCorAna import confpars as cp
from Logger                 import logger
from FileNameManager        import fnm

#-----------------------------

class BatchLogParser :
    """Extracts required information from batch log files
    """

    def __init__ (self) :
        """
        @param path         path to the input log file
        @param dictionary   dictionary of searched items and associated parameters
        @param keys         keys from the dictionary       
        """
        path = None 
        dict = None
        keys = None 

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def parse_batch_log_peds_scan (self) :
        self.path = fnm.path_peds_scan_batch_log()
        self.dict   = {'BATCH_FRAME_TIME_INTERVAL_AVE'  : cp.bat_dark_dt_ave,
                       'BATCH_FRAME_TIME_INTERVAL_RMS'  : cp.bat_dark_dt_rms,
                       #'BATCH_FRAME_TIME_INDEX_MAX'    : time_ind_max,
                       #'BATCH_RUN_NUMBER'              : run_num,
                       #'BATCH_SEC_PER_EVENT'           : rate_sec_per_evt, 
                       #'BATCH_EVENTS_PER_SEC'          : rate_evt_per_sec,
                       'BATCH_NUMBER_OF_EVENTS'         : cp.bat_dark_total,
                       'BATCH_PROCESSING_TIME'          : cp.bat_dark_time
                       }

        self.print_dict()
        self.parse_log_file()

        if  cp.bat_dark_end.value() == cp.bat_dark_end.value_def() and cp.bat_dark_total.value() != cp.bat_dark_total.value_def():
            cp.bat_dark_end.setValue(cp.bat_dark_total.value())

#-----------------------------

    def parse_batch_log_data_scan (self) :
        self.path = fnm.path_data_scan_batch_log()
        self.dict   = {'BATCH_FRAME_TIME_INTERVAL_AVE'  : cp.bat_data_dt_ave,
                       'BATCH_FRAME_TIME_INTERVAL_RMS'  : cp.bat_data_dt_rms,
                       #'BATCH_FRAME_TIME_INDEX_MAX'    : time_ind_max,
                       #'BATCH_RUN_NUMBER'              : run_num,
                       #'BATCH_SEC_PER_EVENT'           : rate_sec_per_evt, 
                       #'BATCH_EVENTS_PER_SEC'          : rate_evt_per_sec,
                       'BATCH_NUMBER_OF_EVENTS'         : cp.bat_data_total,
                       'BATCH_PROCESSING_TIME'          : cp.bat_data_time
                       }

        self.print_dict()
        self.parse_log_file()

        if  cp.bat_data_end.value() == cp.bat_data_end.value_def() and cp.bat_data_total.value() != cp.bat_data_total.value_def():
            cp.bat_data_end.setValue(cp.bat_data_total.value())

#-----------------------------

    def parse_batch_log_data_aver (self) :
        self.path = fnm.path_data_aver_batch_log()
        self.dict   = {
                       'BATCH_IMG_ROWS'                 : cp.bat_img_rows,
                       'BATCH_IMG_COLS'                 : cp.bat_img_cols,
                       'BATCH_IMG_SIZE'                 : cp.bat_img_size
                       #'BATCH_NUMBER_OF_EVENTS'         : cp.bat_data_total,
                       }

        self.print_dict()
        self.parse_log_file()

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def print_dict (self) :
        logger.debug('Parser search dictionary:',__name__)
        for k,v in self.dict.iteritems() :
            msg = '%s : %s' % (k.ljust(32), str(v.value()))
            logger.debug(msg)

#-----------------------------

    def parse_log_file (self) :

        logger.debug('Log file to parse: ' + self.path)
        self.keys   = self.dict.keys()

        if not os.path.lexists(self.path) :
            logger.debug('The requested file: ' + self.path + ' is not available.', __name__)         
            #for val in self.dict.values() :
            #    val.setDefault()
            return

        fin  = open(self.path, 'r')
        for line in fin :
            if line[0:6] == 'BATCH_' :
                fields = line.split()
                key = fields[0]
                if key in self.keys :
                    str_value = fields[1].strip(' ')
                    logger.debug(key+': '+str_value)
                    self.dict[key].setValueFromString ( str_value )
        fin.close() 

#-----------------------------

blp = BatchLogParser ()

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    blp.parse_batch_log_peds_scan()
    blp.parse_batch_log_data_scan()

    sys.exit ( 'End of test for BatchLogParser' )

#-----------------------------
