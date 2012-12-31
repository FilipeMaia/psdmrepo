#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module BatchJobCorAna...
#
#------------------------------------------------------------------------

"""Deals with batch jobs for correlation analysis

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

class BatchJobCorAna(BatchJob) :
    """Deals with batch jobs for correlation analysis.
    """

    def __init__ (self) :
        """Constructor.
        @param fname the file name for ...
        """

        BatchJob.__init__(self)

        self.job_id_cora_split = None

        self.time_sub1 = None

#-----------------------------

    def     make_psana_cfg_file_for_cora_split(self) :
        cfg.make_psana_cfg_file_for_cora_split()

#-----------------------------

    def submit_batch_for_cora_split(self) :

        if not self.job_can_be_submitted(self.job_id_cora_split, self.time_sub1, 'correlation analysis') : return
        self.time_sub1 = gu.get_time_sec()

        self.make_psana_cfg_file_for_cora_split()

        command  = 'psana -c ' + fnm.path_cora_split_psana_cfg() + ' ' + fnm.path_data_xtc_cond()
        queue    = cp.bat_queue.value()
        log_file = fnm.path_cora_split_batch_log()

        self.job_id_cora_split, out, err = gu.batch_job_submit(command, queue, log_file)

#-----------------------------

#    def submit_batch_for_data_aver(self) :

#        if not self.job_can_be_submitted(self.job_id_data_aver, self.time_aver_job_submitted, 'data aver') : return        
#        self.time_aver_job_submitted = gu.get_time_sec()

#        self.make_psana_cfg_file_for_data_aver()

#        command      = 'psana -c ' + fnm.path_data_aver_psana_cfg() + ' ' + fnm.path_data_xtc_cond()
#        queue        = cp.bat_queue.value()
#        bat_log_file = fnm.path_data_aver_batch_log()

#        self.job_id_data_aver, out, err = gu.batch_job_submit(command, queue, bat_log_file)

#-----------------------------

    def check_batch_job_for_cora_split(self) :
        self.check_batch_job(self.job_id_cora_split, 'correlation analysis')

#-----------------------------

    def status_for_cora_split_files(self) :
        return self.status_and_string_for_files(fnm.get_list_of_files_cora_split_all(), 'of split: ' )

#-----------------------------

    def status_batch_job_for_cora_split(self) :
        return self.get_batch_job_status_and_string(self.job_id_cora_split, self.time_sub1)

#-----------------------------

#    def print_work_files_for_data_aver(self) :
#        self.print_files_for_list(fnm.get_list_of_files_cora_split(),'of correlation analysis:')

#-----------------------------

    def check_work_files_cora(self) :
        self.check_files_for_list(fnm.get_list_of_files_cora_split(),'of correlation analysis:')

#-----------------------------

    def remove_files_cora(self) :
        self.remove_files_for_list(fnm.get_list_of_files_cora_split_all(),'of correlation analysis:')

#-----------------------------

    def get_batch_job_id_cora_split(self) :
        return self.job_id_cora_split

#-----------------------------

#    def get_data_aver_from_file(self) :
#        return gu.get_array_from_file( fnm.path_data_aver() )

#-----------------------------

bjcora = BatchJobCorAna ()

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    bjcora.submit_batch_for_cora()
    #gu.sleep_sec(5)

    #bjcora.submit_batch_for_data_scan_on_data_xtc()
    #bjcora.print_work_files_for_data_aver()
    #bjcora.check_work_files_for_data_aver()

    sys.exit ( 'End of test for BatchJobCorAna' )

#-----------------------------
