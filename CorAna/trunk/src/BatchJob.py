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
        self.dict_status = {True  : 'available',
                            False : 'not available'}

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

        if t_sub is None : return False

        if gu.get_time_sec() - t_sub > self.time_interval_sec :
            return False
        else :
            msg = 'Sorry, but '+ comment +' job has already been submitted less then ' + \
            str(self.time_interval_sec) + ' sec ago and has not been completed yet... Be patient, relax and wait...'
            logger.warning(msg, __name__)         
            return True

#-----------------------------

    def check_batch_job(self, job_id, comment='') :

        if job_id is None :
            logger.info('Batch job for ' + comment + ' was not submitted in this session.', __name__) 
            return

        lines = gu.batch_job_check(job_id, cp.bat_queue.value())
        msg = 'Check batch status for ' + comment + ':\n'
        for line in lines :
            msg += line
        logger.info(msg, __name__) 

#-----------------------------

    def kill_batch_job(self, job_id, comment='') :

        if job_id is None :
            #logger.info('Batch job for ' + comment + ' was not submitted in this session.', __name__) 
            return

        lines = gu.batch_job_kill(job_id)
        msg = 'Kill batch job ' + job_id + ' ' + comment + ':\n'
        for line in lines :
            msg += line
        logger.info(msg, __name__) 

#-----------------------------

    def get_batch_job_status(self, job_id, comment='') :

        if job_id is None :
            self.batch_job_status = None
        else :
            self.batch_job_status = gu.batch_job_status(job_id, cp.bat_queue.value())

        if comment != '' :
            logger.info('Status for ' + comment + ': ' + str(self.batch_job_status), __name__) 
        return self.batch_job_status

#-----------------------------

    def get_batch_job_status_and_string(self, job_id, time_sec, comment='') :

        if job_id is None :
            return 'None', 'Batch job was not submitted in this session.'

        time_str = gu.get_local_time_str(time_sec, fmt='%Y-%m-%d %H:%M:%S')
        status = self.get_batch_job_status(job_id, comment)

        msg = 'Job Id: ' + str(job_id) + \
              ' was submitted at ' + str(time_str) + \
              '    Status: ' + str(status)
        return status, msg

#-----------------------------

    def print_files_for_list(self, list_of_files, comment='') :
        logger.info('Print files for list ' + comment, __name__)         
        for fname in list_of_files :
            logger.info(fname)         

#-----------------------------

    def check_files_for_list(self, list_of_files, comment='') :
        logger.info('Check files for list ' + comment, __name__)         
        for fname in list_of_files :
            msg = '%s is %s' % ( fname.ljust(100), self.dict_status[os.path.lexists(fname)] )
            logger.info(msg)         

#-----------------------------

    def status_for_files(self, list_of_files, comment='') :
        status = True
        for fname in list_of_files :
            if not os.path.lexists(fname) :
                status = False
                break
            if os.path.getsize(fname) < 1 :
                status = False
                break
        if comment != '' :
            logger.info('Check file existence and size for the list ' + comment + str(status), __name__)         
        return status

#-----------------------------

    def status_and_string_for_files(self, list_of_files, comment='') :
        status = self.status_for_files(list_of_files, comment)
        return status, 'Files are ' + self.dict_status[status] + '.'

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
