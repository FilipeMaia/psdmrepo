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
import RegDBUtils             as     ru

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

    def path_dark_xtc(self) :
        #return cp.in_dir_dark.value() + '/' + cp.in_file_dark.value()
        return self.path_to_xtc_files_for_run()

    def path_dark_xtc_all_chunks(self) :
        #return cp.in_dir_dark.value() + '/' + gu.xtc_fname_for_all_chunks(cp.in_file_dark.value())
        return self.path_to_xtc_files_for_run()

    def path_dark_xtc_cond(self) :
        if cp.use_dark_xtc_all.value() : return self.path_dark_xtc_all_chunks()
        else                           : return self.path_dark_xtc()


#    def path_data_xtc(self) :
#        return cp.in_dir_data.value() + '/' + cp.in_file_data.value()


#    def path_data_xtc_all_chunks(self) :
#        return cp.in_dir_data.value() + '/' + gu.xtc_fname_for_all_chunks(cp.in_file_data.value())

#    def path_data_xtc_cond(self) :
#        if cp.use_data_xtc_all.value() : return self.path_data_xtc_all_chunks()
#        else                           : return self.path_data_xtc()

    def str_exp_run_dark(self) :
        return self.str_exp_run_for_xtc_path(self.path_dark_xtc())
  
#    def str_exp_run_data(self) :
#        return self.str_exp_run_for_xtc_path(self.path_data_xtc())

#    def str_run_data(self) :
#        return self.str_run_for_xtc_path(self.path_data_xtc())

    def str_exp_run_for_xtc_path(self, path) :
        instrument, experiment, run_str, run_num = gu.parse_xtc_path(path)
        if experiment == None : return 'exp-run-'
        else                  : return experiment + '-' + run_str + '-'

#    def str_run_for_xtc_path(self, path) :
#        instrument, experiment, run_str, run_num = gu.parse_xtc_path(path)
#        if run_str == None : return 'run-'
#        else               : return run_str + '-'

#-----------------------------

    def path_to_calib_dir_custom(self):
        """Returns path to the user selected (non-default) calib dir, for example /reg/neh/home1/<user-name>/<further-path>/calib"""
        return cp.calib_dir.value()

#-----------------------------

    def path_to_calib_dir_default(self):
        """Returns somthing like /reg/d/psdm/CXI/cxitut13/calib or None"""
        if cp.instr_dir .value() is None : return None
        if cp.instr_name.value() is None : return None
        if cp.exp_name  .value() is None : return None
        return cp.instr_dir.value() + '/' + cp.instr_name.value() + '/' + cp.exp_name.value() + '/calib'

#-----------------------------

    def path_to_calib_dir(self):
        if cp.calib_dir.value() != 'None' : return self.path_to_calib_dir_custom()
        else                              : return self.path_to_calib_dir_default()

#-----------------------------

    def path_to_xtc_dir(self):
        """Returns somthing like /reg/d/psdm/CXI/cxitut13/xtc/ or None"""
        if cp.instr_dir.value()  is None : return None
        if cp.instr_name.value() is None : return None
        if cp.exp_name.value()   is None : return None
        return cp.instr_dir.value() + '/' + cp.instr_name.value() + '/' + cp.exp_name.value() + '/xtc/'


    def get_list_of_xtc_files(self):
        dir = self.path_to_xtc_dir()
        return gu.get_list_of_files_in_dir_for_ext(dir, '.xtc')


    def get_list_of_xtc_runs(self):
        """Returns the list of xtc runs as string, for example:  ['0001', '0202', '0203', '0204',...]
        """
        list_of_xtc_files = self.get_list_of_xtc_files()
        list_of_xtc_runs = []
        for fname in list_of_xtc_files :
            exp, run, stream, chunk, ext = gu.parse_xtc_file_name(fname)
            if run in list_of_xtc_runs : continue
            list_of_xtc_runs.append(run)
        return list_of_xtc_runs


    def get_list_of_xtc_run_nums(self):
        """Returns the list of xtc integer run numbers:  [1, 202, 203, 204,...]
        """
        list_of_xtc_runs = self.get_list_of_xtc_runs()
        list_of_xtc_run_nums = []
        for run in list_of_xtc_runs :
            list_of_xtc_run_nums.append(int(run))
        return list_of_xtc_run_nums


    def path_to_xtc_files_for_run(self):
        """Returns somthing like /reg/d/psdm/CXI/cxitut13/xtc/e304-r0022-*.xtc"""

        if cp.str_run_number.value() == 'None' : return None
        pattern = '-r' + cp.str_run_number.value()

        for fname in self.get_list_of_xtc_files() :
            if fname.find(pattern) != -1 :
                #print fname
                expnum, runnum, stream, chunk, ext = gu.parse_xtc_file_name(fname) # Parse: e170-r0003-s00-c00.xtc
                return self.path_to_xtc_dir() + 'e' + expnum + '-r' + runnum + '-*.xtc' 

        return None

#-----------------------------

    def path_prefix(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() 

    def path_prefix_dark(self) :
        return self.path_prefix() + self.str_exp_run_dark()

#    def path_prefix_data(self) :
#        return self.path_prefix() + self.str_exp_run_data()

#    def path_prefix_cora(self) :
#        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + 'cora'


#-----------------------------

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

    def log_file(self) :
        return cp.dir_work.value() + '/' + logger.getLogFileName()

#-----------------------------

    def path_peds_scan_psana_cfg(self) :
        return self.path_prefix_dark() + 'peds-scan.cfg'

    def path_peds_scan_batch_log(self) :
        return self.path_prefix_dark() + 'peds-scan-batch-log.txt'


    def path_peds_aver_psana_cfg(self) :
        return self.path_prefix_dark() + 'peds-aver.cfg'

    def path_peds_aver_batch_log(self) :
        return self.path_prefix_dark() + 'peds-aver-batch-log.txt'

    def path_peds_ave(self) :
        return self.path_prefix_dark() + 'peds-ave.txt'

    def path_peds_rms(self) :
        return self.path_prefix_dark() + 'peds-rms.txt'

    def path_peds_aver_plot(self) :
        return self.path_prefix_dark() + 'peds-aver-plot.png'

#-----------------------------

    def path_hotpix_mask(self) :
        return self.path_prefix_dark() + 'hotpix-mask-thr-' \
               + str(cp.mask_hot_thr.value()) + 'ADU.txt'

    def path_hotpix_mask_prefix(self) :
        return os.path.splitext(self.path_hotpix_mask())[0]

    def path_hotpix_mask_plot(self) :
        return self.path_hotpix_mask_prefix() + '-plot.png' 

#-----------------------------

    def  get_list_of_files_peds_scan(self) :
        self.list_of_files_peds_scan = []
        #self.list_of_files_peds_scan.append(self.path_dark_xtc())
        self.list_of_files_peds_scan.append(self.path_peds_scan_psana_cfg())
        self.list_of_files_peds_scan.append(self.path_peds_scan_batch_log())
        return self.list_of_files_peds_scan


    def  get_list_of_files_peds_aver(self) :
        self.list_of_files_peds_aver = []
        self.list_of_files_peds_aver.append(self.path_peds_aver_psana_cfg())
        self.list_of_files_peds_aver.append(self.path_peds_aver_batch_log())
        #self.list_of_files_peds_aver.append(self.path_peds_ave())
        #self.list_of_files_peds_aver.append(self.path_peds_rms())
        #self.list_of_files_peds_aver.append(self.path_hotpix_mask())
        return self.list_of_files_peds_aver


    def  get_list_of_files_peds(self) :
        self.list_of_files_peds = self.get_list_of_files_peds_scan()
        self.list_of_files_peds+= self.get_list_of_files_peds_aver()
        self.list_of_files_peds.append(self.path_peds_aver_plot())
        #self.list_of_files_peds.append(self.path_dark_xtc())
        return self.list_of_files_peds


#-----------------------------

    def get_list_of_enumerated_file_names(self, path1='file.dat', len_of_list=0) :
        return gu.get_list_of_enumerated_file_names(path1, len_of_list)

#-----------------------------

    def get_list_of_files_for_all_sources(self, path1='file.dat', list_of_insets=[]) :
        """Returns the list of file names, where the file name is a combination of path1 and inset from list
        """
        if list_of_insets == [] : return [] # [path1]
        name, ext = os.path.splitext(path1)
        return ['%s-%s%s' % (name, src, ext) for src in list_of_insets]


    def get_list_of_files_for_detector(self, path1='work/file.dat', det_name='') :
        """From pattern of the path it makes a list of files with names for all sources."""
        if det_name == '' : return path1
        lst = ru.list_of_detectors_in_run_for_selected(cp.instr_name.value(), cp.exp_name.value(), int(cp.str_run_number.value()), det_name)
        return self.get_list_of_files_for_all_sources(path1, lst)

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
