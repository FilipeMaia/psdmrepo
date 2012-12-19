#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module BatchJobData...
#
#------------------------------------------------------------------------

"""Deals with batch jobs for data

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

class BatchJobData :
    """Deals with batch jobs for data.
    """

    def __init__ (self) :
        """Constructor.
        @param fname the file name for ...
        """
        self.job_id_data_aver = None
        self.job_id_data_scan = None
        #self.path_peds_cfg   = fnm.path_data_aver_psana_cfg()

        self.time_aver_job_submitted = None
        self.time_scan_job_submitted = None

        self.time_interval_sec      = 100
        self.dict_status = {True  : ' is available',
                            False : ' is not available'}

#-----------------------------

    def     make_psana_cfg_file_for_data_scan(self) :
        cfg.make_psana_cfg_file_for_data_scan()

    def     make_psana_cfg_file_for_data_aver(self) :
        cfg.make_psana_cfg_file_for_data_aver()

#-----------------------------

    def submit_batch_for_data_scan(self) :

        if not self.job_can_be_submitted(self.job_id_data_scan, self.time_scan_job_submitted, 'data scan') : return
        self.time_scan_job_submitted = gu.get_time_sec()

        self.make_psana_cfg_file_for_data_scan()

        command      = 'psana -c ' + fnm.path_data_scan_psana_cfg() + ' ' + fnm.path_data_xtc_cond()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_data_scan_batch_log()
        if os.path.lexists(bat_log_file) : gu.remove_file(bat_log_file)

        self.job_id_data_scan, out, err = gu.batch_job_submit(command, queue, bat_log_file)

        if err != '' : logger.warning( err, __name__) 
        logger.info(out, __name__) 
        #logger.debug('Submit batch for scan on data run, job Id: ' + self.job_id_data_scan, __name__)

#-----------------------------

    def submit_batch_for_data_aver(self) :

        if not self.job_can_be_submitted(self.job_id_data_aver, self.time_aver_job_submitted, 'data aver') : return        
        self.time_aver_job_submitted = gu.get_time_sec()

        self.make_psana_cfg_file_for_data_aver()

        command      = 'psana -c ' + fnm.path_data_aver_psana_cfg() + ' ' + fnm.path_data_xtc_cond()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_data_aver_batch_log()
        if os.path.lexists(bat_log_file) : gu.remove_file(bat_log_file)

        self.job_id_data_aver, out, err = gu.batch_job_submit(command, queue, bat_log_file)

        if err != '' : logger.warning(err, __name__) 
        logger.info(out, __name__) 
        #logger.debug('   Submit batch for pedestals on data run, job Id: ' + self.job_id_data_aver, __name__) 

#-----------------------------

    def job_can_be_submitted(self, job_id, t_sub, comment='') :
        if self.job_was_recently_submitted(t_sub, comment) and \
           (self.get_batch_job_status(job_id, comment) != 'DONE') :

            msg = 'Batch job can be re-resubmitted when timeout ' \
                  + str(self.time_interval_sec) + ' sec is expired' \
                  + ' or the job ' + job_id + ' is DONE'
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
            msg = 'Sorry, but '+ comment +' job was already submitted less then ' + \
            str(self.time_interval_sec) + ' sec ago... Be patient, just relax and wait...'
            logger.warning(msg, __name__)         
            return True

#-----------------------------

    def check_batch_job_for_data_aver(self) :
        self.check_batch_job(self.job_id_data_aver, 'data average')

#-----------------------------

    def check_batch_job_for_data_scan(self) :
        self.check_batch_job(self.job_id_data_scan, 'data scan')

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

    def print_work_files_for_data_aver(self) :
        logger.info('Print work files for data scan / average:', __name__)         
        for fname in fnm.get_list_of_files_data_aver() :
            logger.info(fname)         

#-----------------------------

    def check_work_files_for_data_aver(self) :
        logger.info('Check work files for data scan / average:', __name__)         
        for fname in fnm.get_list_of_files_data_aver() :
            msg = '%s %s' % ( fname.ljust(100), self.dict_status[os.path.lexists(fname)] )
            logger.info(msg)         

#-----------------------------

    def remove_files_data_aver(self) :
        logger.info('Remove data_average work files for selected run:', __name__)
        for fname in fnm.get_list_of_files_data_aver() :
            if os.path.lexists(fname) :
                gu.remove_file(fname)
                logger.info('Removed: ' + fname)

#-----------------------------

    def status_for_data_aver_file(self) :
        fname  = fnm.path_data_ave()
        status = os.path.lexists(fname)
        logger.info('Status: data file ' + fname + self.dict_status[status], __name__) 
        return status

#-----------------------------

    def get_data_aver_from_file(self) :
        return gu.get_array_from_file( fnm.path_data_aver() )

#-----------------------------

bjdata = BatchJobData ()

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    #bjdata.submit_batch_for_data_aver()
    #gu.sleep_sec(5)
    #bjdata.check_batch_job_for_data_aver()

    #bjdata.submit_batch_for_data_scan_on_data_xtc()
    #bjdata.print_work_files_for_data_aver()
    bjdata.check_work_files_for_data_aver()

    sys.exit ( 'End of test for BatchJobData' )

#-----------------------------
