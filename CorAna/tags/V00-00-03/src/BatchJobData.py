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
#import sys
#import os

from BatchJob import *

#from ConfigParametersCorAna   import confpars as cp
#from Logger                   import logger
#from ConfigFileGenerator      import cfg
#from FileNameManager          import fnm
#import GlobalUtils            as     gu

#-----------------------------

class BatchJobData(BatchJob) :
    """Deals with batch jobs for data.
    """

    def __init__ (self) :
        """Constructor.
        @param fname the file name for ...
        """

        BatchJob.__init__(self)

        self.job_id_data_aver = None
        self.job_id_data_scan = None

        self.time_aver_job_submitted = None
        self.time_scan_job_submitted = None

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

        self.job_id_data_scan, out, err = gu.batch_job_submit(command, queue, bat_log_file)

#-----------------------------

    def submit_batch_for_data_aver(self) :

        if not self.job_can_be_submitted(self.job_id_data_aver, self.time_aver_job_submitted, 'data aver') : return        
        self.time_aver_job_submitted = gu.get_time_sec()

        self.make_psana_cfg_file_for_data_aver()

        command      = 'psana -c ' + fnm.path_data_aver_psana_cfg() + ' ' + fnm.path_data_xtc_cond()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_data_aver_batch_log()

        self.job_id_data_aver, out, err = gu.batch_job_submit(command, queue, bat_log_file)

#-----------------------------

    def check_batch_job_for_data_aver(self) :
        self.check_batch_job(self.job_id_data_aver, 'data average')

#-----------------------------

    def check_batch_job_for_data_scan(self) :
        self.check_batch_job(self.job_id_data_scan, 'data scan')

#-----------------------------
#-----------------------------

    def print_work_files_for_data_aver(self) :
        self.print_files_for_list(fnm.get_list_of_files_data_aver(),'of data scan / average:')

#-----------------------------

    def check_work_files_for_data_aver(self) :
        self.check_files_for_list(fnm.get_list_of_files_data_aver(),'of data scan / average:')

#-----------------------------

    def remove_files_data_aver(self) :
        self.remove_files_for_list(fnm.get_list_of_files_data_aver(),'of data scan / average:')

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
