#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module BatchJobPedestals...
#
#------------------------------------------------------------------------

"""Deals with batch jobs for pedestals

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

class BatchJobPedestals (BatchJob) :
    """Deals with batch jobs for pedestals.
    """

    def __init__ (self) :
        """Constructor.
        @param fname  the file name for output log file
        """

        BatchJob.__init__(self)

        self.job_id_peds_str = None
        self.job_id_scan_str = None

        self.time_peds_job_submitted = None
        self.time_scan_job_submitted = None

#-----------------------------

    def     make_psana_cfg_file_for_peds_aver(self) :
        cfg.make_psana_cfg_file_for_peds_aver()

    def     make_psana_cfg_file_for_peds_scan(self) :
        cfg.make_psana_cfg_file_for_peds_scan()

#-----------------------------

    def submit_batch_for_peds_scan(self) :

        if not self.job_can_be_submitted(self.job_id_scan_str, self.time_scan_job_submitted, 'scan') : return
        self.time_scan_job_submitted = gu.get_time_sec()

        self.make_psana_cfg_file_for_peds_scan()

        command      = 'psana -c ' + fnm.path_peds_scan_psana_cfg() + ' ' + fnm.path_dark_xtc_cond()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_peds_scan_batch_log()

        self.job_id_scan_str, out, err = gu.batch_job_submit(command, queue, bat_log_file)

#-----------------------------

    def submit_batch_for_peds_aver(self) :

        if not self.job_can_be_submitted(self.job_id_peds_str, self.time_peds_job_submitted, 'peds') : return        
        self.time_peds_job_submitted = gu.get_time_sec()

        self.make_psana_cfg_file_for_peds_aver()

        command      = 'psana -c ' + fnm.path_peds_aver_psana_cfg() + ' ' + fnm.path_dark_xtc_cond()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_peds_aver_batch_log()

        self.job_id_peds_str, out, err = gu.batch_job_submit(command, queue, bat_log_file)

#-----------------------------

    def check_batch_job_for_peds_aver(self) :
        self.check_batch_job(self.job_id_peds_str, 'peds')

#-----------------------------

    def check_batch_job_for_peds_scan(self) :
        self.check_batch_job(self.job_id_scan_str, 'scan')

#-----------------------------
#-----------------------------

    def print_work_files_for_pedestals(self) :
        self.print_files_for_list(fnm.get_list_of_files_pedestals(),'of dark run / pedestals:')

#-----------------------------

    def check_work_files_for_pedestals(self) :
        self.check_files_for_list(fnm.get_list_of_files_pedestals(),'of dark run / pedestals:')

#-----------------------------

    def remove_files_pedestals(self) :
        self.remove_files_for_list(fnm.get_list_of_files_pedestals(),'of dark run / pedestals:')

#-----------------------------

    def status_for_pedestal_file(self) :
        fname  = fnm.path_pedestals_ave()
        status = os.path.lexists(fname)
        logger.info('Status: pedestal file ' + fname + self.dict_status[status], __name__) 
        return status

#-----------------------------

    def get_pedestals_from_file(self) :
        return gu.get_array_from_file( fnm.path_pedestals_ave() )

#-----------------------------

bjpeds = BatchJobPedestals ()

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    #bjpeds.submit_batch_for_peds_aver()
    #gu.sleep_sec(5)
    #bjpeds.check_batch_job_for_peds_scan()

    #bjpeds.submit_batch_for_peds_scan_on_dark_xtc()
    #bjpeds.print_work_files_for_pedestals()
    bjpeds.check_work_files_for_pedestals()

    sys.exit ( 'End of test for BatchJobPedestals' )

#-----------------------------
