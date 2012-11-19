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
import GlobalUtils            as   gu
#-----------------------------

class FileNameManager :
    """Dynamically generates the file names from the confoguration parameters.
    """

    def __init__ (self) :
        """Constructor.
        @param fname  the file name for output log file
        """
#-----------------------------
#-----------------------------

    def log_file(self) :
        return cp.dir_work.value() + '/' + logger.getLogFileName()

    def log_file_total(self) :
        return cp.dir_work.value() + '/' + logger.getLogTotalFileName()

#-----------------------------

    def path_dark_xtc(self) :
        return cp.in_dir_dark.value() + '/' + cp.in_file_dark.value()

    def path_flat_xtc(self) :
        return cp.in_dir_flat.value() + '/' + cp.in_file_flat.value()

    def path_data_xtc(self) :
        return cp.in_dir_data.value() + '/' + cp.in_file_data.value()

    def str_exp_run_dark(self) :
        instrument, experiment, run_str, run_num = gu.parse_xtc_path(self.path_dark_xtc())
        if experiment == None : return 'exp-run-'
        else                  : return experiment + '-' + run_str + '-'

    def str_exp_run_flat(self) :
        instrument, experiment, run_str, run_num = gu.parse_xtc_path(self.path_flat_xtc())
        if experiment == None : return 'exp-run-'
        else                  : return experiment + '-' + run_str + '-'

    def str_exp_run_data(self) :
        instrument, experiment, run_str, run_num = gu.parse_xtc_path(self.path_data_xtc())
        if experiment == None : return 'exp-run-'
        else                  : return experiment + '-' + run_str + '-'

#-----------------------------

    def path_pedestals_psana_cfg(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + self.str_exp_run_dark() + 'peds.cfg'

    def path_pedestals_ave(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + self.str_exp_run_dark() + 'peds-ave.txt'

    def path_pedestals_rms(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + self.str_exp_run_dark() + 'peds-rms.txt'

    def path_pedestals_batch_log(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + self.str_exp_run_dark() + 'peds-batch-log.txt'

    def path_peds_scan_batch_log(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + self.str_exp_run_dark() + 'peds-scan-batch-log.txt'

    def path_peds_scan_psana_cfg(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + self.str_exp_run_dark() + 'peds-scan.cfg'

    def path_pedestals_plot(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + self.str_exp_run_dark() + 'peds-plot.png'

#-----------------------------

    def  get_list_of_files_pedestals(self) :
        self.list_of_files_pedestals = []
        #self.list_of_files_pedestals.append(self.path_dark_xtc())
        self.list_of_files_pedestals.append(self.path_pedestals_psana_cfg())
        self.list_of_files_pedestals.append(self.path_pedestals_batch_log())
        self.list_of_files_pedestals.append(self.path_pedestals_ave())
        self.list_of_files_pedestals.append(self.path_pedestals_rms())
        self.list_of_files_pedestals.append(self.path_peds_scan_psana_cfg())
        self.list_of_files_pedestals.append(self.path_peds_scan_batch_log())
        self.list_of_files_pedestals.append(self.path_pedestals_plot())
        return self.list_of_files_pedestals

#-----------------------------

    def  get_list_of_files_flatfield(self) :
        self.list_of_files_flatfield = []
        return self.list_of_files_flatfield

#-----------------------------

    def  get_list_of_files_data(self) :
        self.list_of_files_data = []
        return self.list_of_files_data
    
#-----------------------------

#-----------------------------

fnm = FileNameManager ()

#-----------------------------

if __name__ == "__main__" :

    print 'path_pedestals_xtc()       : ', fnm.path_dark_xtc()
    print 'path_pedestals_psana_cfg() : ', fnm.path_pedestals_psana_cfg()
    print 'path_pedestals_ave()       : ', fnm.path_pedestals_ave()
    print 'path_pedestals_rms()       : ', fnm.path_pedestals_rms()
    print 'path_pedestals_batch_log() : ', fnm.path_pedestals_batch_log()
    print 'path_peds_scan_psana_cfg() : ', fnm.path_peds_scan_psana_cfg()
    print 'path_peds_scan_batch_log() : ', fnm.path_peds_scan_batch_log()
    print '\n',
    print '\n',
    print 'str_exp_run_dark()   : ', fnm.str_exp_run_dark()
    print 'str_exp_run_flat()   : ', fnm.str_exp_run_flat()
    print 'str_exp_run_data()   : ', fnm.str_exp_run_data()
    
    sys.exit ( 'End of test for FileNameManager' )

#-----------------------------
