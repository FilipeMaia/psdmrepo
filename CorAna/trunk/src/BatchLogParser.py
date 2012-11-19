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

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
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

    def parse_batch_log_pedestals_tahometer (self) :
        self.path = fnm.path_pedestals_tahometer_batch_log()
        self.dict   = {'BATCH_PROCESSING_TIME'  : cp.bat_dark_time,
                       'BATCH_NUMBER_OF_EVENTS' : cp.bat_dark_total
                       #'BATCH_SEC_PER_EVENT'    : 
                       #'BATCH_EVENTS_PER_SEC'   : 
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
            for val in self.dict.values() :
                val.setDefault()
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

    blp.parse_batch_log_pedestals_tahometer()

    sys.exit ( 'End of test for BatchLogParser' )

#-----------------------------
