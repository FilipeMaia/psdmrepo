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

    def path_dir_work(self) :
        path = cp.dir_work.value()
        #print 'path_dir_work:', path
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

    def path_prefix_data(self) :
        #return self.path_prefix() + self.str_exp_run_data()
        return './'

#-----------------------------

    def path_gui_image(self) :
        return self.path_prefix_data() + 'gui-image.png'

#-----------------------------

    def path_dark_xtc(self) :
        return self.path_to_xtc_files_for_run()

    def path_dark_xtc_all_chunks(self) :
        return self.path_to_xtc_files_for_run()

    def path_dark_xtc_cond(self) :
        if cp.use_dark_xtc_all.value() : return self.path_dark_xtc_all_chunks()
        else                           : return self.path_dark_xtc()

    def str_exp_run_dark(self) :
        return self.str_exp_run_for_xtc_path(self.path_dark_xtc())
  
    def str_exp_run_for_xtc_path(self, path) :
        instrument, experiment, run_str, run_num = gu.parse_xtc_path(path)
        if experiment == None : return 'exp-run-'
        else                  : return experiment + '-' + run_str + '-'

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
        return [self.path_peds_scan_psana_cfg(),
                self.path_peds_scan_batch_log()]


    def  get_list_of_files_peds_aver(self) :
        return [self.path_peds_aver_psana_cfg(),
                self.path_peds_aver_batch_log()]


    def  get_list_of_files_peds(self) :
        self.list_of_files_peds = self.get_list_of_files_peds_scan()
        self.list_of_files_peds+= self.get_list_of_files_peds_aver()
        self.list_of_files_peds.append(self.path_peds_aver_plot())
        #self.list_of_files_peds.append(self.path_dark_xtc())
        return self.list_of_files_peds

#-----------------------------
# Interaction with RegDB

    def txt_of_sources_in_run(self, run_number) :
        return ru.txt_of_sources_in_run(cp.instr_name.value(), cp.exp_name.value(), run_number)


    def list_of_sources_in_run(self, run_number) :
        return ru.list_of_sources_in_run(cp.instr_name.value(), cp.exp_name.value(), run_number)


    def  list_of_sources_in_run_for_selected_detector(self, det_name) :
        ins, exp, run_number = cp.instr_name.value(), cp.exp_name.value(), int(cp.str_run_number.value())
        return ru.list_of_sources_in_run_for_selected_detector(ins, exp, run_number, det_name)

#-----------------------------

    def get_list_of_files_for_detector(self, path1='file.dat', det_name='') :
        """For specified file name pattern and detector returns the list of files for all sources.
        """
        if det_name == '' : return path1
        lst = self.list_of_sources_in_run_for_selected_detector(det_name)
        return gu.get_list_of_files_for_list_of_insets(path1, lst)


    def get_list_of_files_for_all_detectors_and_sources(self, path1='file.dat') :
        """For specified file name pattern returns the list of file names for all current detectors and all their sources
        """
        list_of_files = []
        for det_name in cp.list_of_dets_selected() :
            lst_src = self.list_of_sources_in_run_for_selected_detector(det_name)
            list_of_files += gu.get_list_of_files_for_list_of_insets(path1, lst_src)
        return list_of_files

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
