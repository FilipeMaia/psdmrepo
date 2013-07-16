#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module FileNameManager...
#
#------------------------------------------------------------------------

"""Dynamically generates the file names from the confoguration parameters

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@author Mikhail S. Dubrovin
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

from   ConfigParametersForApp import cp
from   Logger                 import logger
import GlobalUtils            as     gu

#-----------------------------

class FileNameManager :
    """Dynamically generates the file names from the confoguration parameters.
    """

    def __init__ (self) :
        """Constructor.
        @param fname  the file name for output log file
        """

#-----------------------------

#    def log_file(self) :
#        return cp.dir_work.value() + '/' + logger.getLogFileName()

#    def log_file_total(self) :
#        return cp.dir_work.value() + '/' + logger.getLogTotalFileName()

#-----------------------------

    def path_dir_work(self) :
        #return "%s/work/" % os.path.dirname(sys.argv[0])
        path = "%s/work/" % os.path.dirname(sys.argv[0])
        print 'path_dir_work:', path
        return path


    def str_exp_run_for_xtc_path(self, path) :
        instrument, experiment, run_str, run_num = gu.parse_xtc_path(path)
        if experiment == None : return 'exp-run-'
        else                  : return experiment + '-' + run_str + '-'


    def str_run_for_xtc_path(self, path) :
        instrument, experiment, run_str, run_num = gu.parse_xtc_path(path)
        if run_str == None : return 'run-'
        else               : return run_str + '-'

#-----------------------------

#    def str_exp_run_data(self) :
#        return self.str_exp_run_for_xtc_path(self.path_data_xtc())

#-----------------------------

#    def path_prefix(self) :
#        return self.path_dir_work() + '/' + cp.fname_prefix.value() 

    def path_prefix_data(self) :
        #return self.path_prefix() + self.str_exp_run_data()
        return './'

#-----------------------------

    def path_gui_image(self) :
        return self.path_prefix_data() + 'gui-image.png'

#    def path_config_pars(self) :
#        return cp.fname_cp.value()

#-----------------------------

#    def path_dark_xtc(self) :
#        return cp.in_dir_dark.value() + '/' + cp.in_file_dark.value()

#    def path_data_xtc(self) :
#        return cp.in_dir_data.value() + '/' + cp.in_file_data.value()


#    def path_dark_xtc_all_chunks(self) :
#        return cp.in_dir_dark.value() + '/' + gu.xtc_fname_for_all_chunks(cp.in_file_dark.value())

#    def path_data_xtc_all_chunks(self) :
#        return cp.in_dir_data.value() + '/' + gu.xtc_fname_for_all_chunks(cp.in_file_data.value())

#    def path_dark_xtc_cond(self) :
#        if cp.use_dark_xtc_all.value() : return self.path_dark_xtc_all_chunks()
#        else                           : return self.path_dark_xtc()

#    def path_data_xtc_cond(self) :
#        if cp.use_data_xtc_all.value() : return self.path_data_xtc_all_chunks()
#        else                           : return self.path_data_xtc()

#    def str_exp_run_dark(self) :
#        return self.str_exp_run_for_xtc_path(self.path_dark_xtc())
  
#    def str_exp_run_data(self) :
#        return self.str_exp_run_for_xtc_path(self.path_data_xtc())

#    def str_run_data(self) :
#        return self.str_run_for_xtc_path(self.path_data_xtc())

#    def str_exp_run_for_xtc_path(self, path) :
#        instrument, experiment, run_str, run_num = gu.parse_xtc_path(path)
#        if experiment == None : return 'exp-run-'
#        else                  : return experiment + '-' + run_str + '-'

#    def str_run_for_xtc_path(self, path) :
#        instrument, experiment, run_str, run_num = gu.parse_xtc_path(path)
#        if run_str == None : return 'run-'
#        else               : return run_str + '-'

#-----------------------------

#    def path_prefix(self) :
#        return cp.dir_work.value() + '/' + cp.fname_prefix.value() 

#    def path_prefix_dark(self) :
#        return self.path_prefix() + self.str_exp_run_dark()

#    def path_prefix_data(self) :
#        return self.path_prefix() + self.str_exp_run_data()

#    def path_prefix_cora(self) :
#        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + 'cora'

#-----------------------------

#    def path_data_scan_psana_cfg(self) :
#        return self.path_prefix_data()  + 'data-scan.cfg'

#    def path_data_scan_batch_log(self) :
#        return self.path_prefix_data() + 'data-scan-batch-log.txt'

#    def path_data_scan_monitors_data(self) :
#        return self.path_prefix_data() + 'data-scan-mons-data.txt'

#    def path_data_scan_monitors_commments(self) :
#        return self.path_prefix_data() + 'data-scan-mons-comments.txt'

#    def path_data_scan_tstamp_list(self) :
#        return self.path_prefix_data() + 'data-scan-tstamp-list.txt'

#    def path_data_scan_tstamp_list_tmp(self) :
#        return  self.path_data_scan_tstamp_list() + '-tmp'

#-----------------------------

#    def  get_list_of_files_data(self) :
#        self.list_of_files_data  = self.get_list_of_files_data_scan()
#        self.list_of_files_data += self.get_list_of_files_data_aver()
#        #self.list_of_files_data.append(fnm.path_data_xtc())
#        return self.list_of_files_data

#-----------------------------

fnm = FileNameManager ()

#-----------------------------

if __name__ == "__main__" :

#    print '\nfnm.get_list_of_files_cora_proc_check():' 
#    list =   fnm.get_list_of_files_cora_proc_check()
#    for fname in list : print fname

#    print 'fnm.path_hotpix_mask() : ', fnm.path_hotpix_mask()
#    print 'fnm.path_satpix_mask() : ', fnm.path_satpix_mask()
    
    sys.exit ( 'End of test for FileNameManager' )

#-----------------------------
