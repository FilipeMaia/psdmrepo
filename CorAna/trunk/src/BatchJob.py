#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module BatchJob...
#
#------------------------------------------------------------------------

"""Base class with common methods for batch jobs

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

from ConfigParametersCorAna   import confpars as cp
from Logger                   import logger
from ConfigFileGenerator      import cfg
from FileNameManager          import fnm
import GlobalUtils            as     gu

#-----------------------------

class BatchJob :
    """Base class with common methods for batch jobs.
    """

    def __init__ (self) :
        """Constructor.
        @param fname the file name for ...
        """

        self.time_interval_sec      = 100
        self.dict_status = {True  : ' is available',
                            False : ' is not available'}

#-----------------------------

    def job_can_be_submitted(self, job_id, t_sub, comment='') :
        if self.job_was_recently_submitted(t_sub, comment) and \
           (self.get_batch_job_status(job_id, comment) != 'DONE') :

            msg = 'Batch job can be re-resubmitted when timeout ' \
                  + str(self.time_interval_sec) + ' sec is expired' \
                  + ' or the job ' + str(job_id) + ' is DONE'
            logger.info(msg, __name__)             
            return False
        else :
            return True

#-----------------------------

    def job_was_recently_submitted(self, t_sub, comment='') :

        if t_sub == None : return False

        if gu.get_time_sec() - t_sub > self.time_interval_sec :
            return False
        else :
            msg = 'Sorry, but '+ comment +' job has already been submitted less then ' + \
            str(self.time_interval_sec) + ' sec ago and has not been completed yet... Be patient, relax and wait...'
            logger.warning(msg, __name__)         
            return True

#-----------------------------

    def check_batch_job(self, job_id, comment='') :

        if job_id == None :
            logger.info('Batch job for ' + comment + ' was not submitted in this session.', __name__) 
            return

        lines = gu.batch_job_check(job_id, cp.bat_queue.value())
        msg = 'Check batch status for ' + comment + ':\n'
        for line in lines :
            msg += line
        logger.info(msg, __name__) 

#-----------------------------

    def get_batch_job_status(self, job_id, comment='') :

        if job_id == None :
            self.batch_job_status = None
        else :
            self.batch_job_status = gu.batch_job_status(job_id, cp.bat_queue.value())

        logger.info('Status for ' + comment + ': ' + str(self.batch_job_status), __name__) 
        return self.batch_job_status

#-----------------------------

    def print_files_for_list(self, list_of_files, comment='') :
        logger.info('Print files for list ' + comment, __name__)         
        for fname in list_of_files :
            logger.info(fname)         

#-----------------------------

    def check_files_for_list(self, list_of_files, comment='') :
        logger.info('Check files for list ' + comment, __name__)         
        for fname in list_of_files :
            msg = '%s %s' % ( fname.ljust(100), self.dict_status[os.path.lexists(fname)] )
            logger.info(msg)         

#-----------------------------

    def remove_files_for_list(self, list_of_files, comment='') :
        logger.info('Remove files for list ' + comment, __name__)
        for fname in list_of_files :
            if os.path.lexists(fname) :
                gu.remove_file(fname)
                logger.info('Removed   : ' + fname)
            else :
                logger.info('Not found : ' + fname)

#-----------------------------

if __name__ == "__main__" :
    sys.exit ( 'This is a base class... Not supposed to be run by itself.' )

#-----------------------------
