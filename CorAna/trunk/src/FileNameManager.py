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
import GlobalUtils          as     gu

#-----------------------------

class FileNameManager :
    """Dynamically generates the file names from the confoguration parameters.
    """

    def __init__ (self) :
        """Constructor.
        @param fname  the file name for output log file
        """

#-----------------------------

    def log_file(self) :
        return cp.dir_work.value() + '/' + logger.getLogFileName()

    def log_file_total(self) :
        return cp.dir_work.value() + '/' + logger.getLogTotalFileName()

#-----------------------------

    def path_config_pars(self) :
        return cp.fname_cp.value()

#-----------------------------

    def path_dark_xtc(self) :
        return cp.in_dir_dark.value() + '/' + cp.in_file_dark.value()

    #def path_flat_xtc(self) :
    #    return cp.in_dir_flat.value() + '/' + cp.in_file_flat.value()

    def path_data_xtc(self) :
        return cp.in_dir_data.value() + '/' + cp.in_file_data.value()


    def path_dark_xtc_all_chunks(self) :
        return cp.in_dir_dark.value() + '/' + gu.xtc_fname_for_all_chunks(cp.in_file_dark.value())

    def path_data_xtc_all_chunks(self) :
        return cp.in_dir_data.value() + '/' + gu.xtc_fname_for_all_chunks(cp.in_file_data.value())

    def path_dark_xtc_cond(self) :
        if cp.use_dark_xtc_all.value() : return self.path_dark_xtc_all_chunks()
        else                           : return self.path_dark_xtc()

    def path_data_xtc_cond(self) :
        if cp.use_data_xtc_all.value() : return self.path_data_xtc_all_chunks()
        else                           : return self.path_data_xtc()

    def str_exp_run_dark(self) :
        return self.str_exp_run_for_xtc_path(self.path_dark_xtc())

    #def str_exp_run_flat(self) :
    #    return self.str_exp_run_for_xtc_path(self.path_flat_xtc())

    def str_exp_run_data(self) :
        return self.str_exp_run_for_xtc_path(self.path_data_xtc())

    def str_run_data(self) :
        return self.str_run_for_xtc_path(self.path_data_xtc())

    def str_exp_run_for_xtc_path(self, path) :
        instrument, experiment, run_str, run_num = gu.parse_xtc_path(path)
        if experiment == None : return 'exp-run-'
        else                  : return experiment + '-' + run_str + '-'

    def str_run_for_xtc_path(self, path) :
        instrument, experiment, run_str, run_num = gu.parse_xtc_path(path)
        if run_str == None : return 'run-'
        else               : return run_str + '-'

#-----------------------------

    def path_prefix(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() 

    def path_prefix_dark(self) :
        return self.path_prefix() + self.str_exp_run_dark()

    def path_prefix_data(self) :
        return self.path_prefix() + self.str_exp_run_data()

#-----------------------------

    def path_blam(self) :
        return cp.dname_blam.value() + '/' + cp.fname_blam.value()

    def path_flat(self) :
        return cp.dname_flat.value() + '/' + cp.fname_flat.value()

    def path_flat_plot(self) :
        return self.path_prefix() + self.str_exp_run_data() + 'flat-plot.png'

    def path_blam_plot(self) :
        return self.path_prefix() + self.str_exp_run_data() + 'blam-plot.png'


#-----------------------------

    def path_peds_scan_batch_log(self) :
        return self.path_prefix_dark() + 'peds-scan-batch-log.txt'

    def path_peds_scan_psana_cfg(self) :
        return self.path_prefix_dark() + 'peds-scan.cfg'

    def path_peds_scan_tstamp_list(self) :
        return self.path_prefix_dark() + 'peds-scan-tstamp-list.txt'

    def path_peds_scan_tstamp_list_tmp(self) :
        return  self.path_peds_scan_tstamp_list() + '-tmp'



    def path_peds_aver_psana_cfg(self) :
        return self.path_prefix_dark()  + 'peds.cfg'

    def path_peds_aver_batch_log(self) :
        return self.path_prefix_dark()  + 'peds-batch-log.txt'

    def path_pedestals_ave(self) :
        return self.path_prefix_dark()  + 'peds-ave.txt'

    def path_pedestals_rms(self) :
        return self.path_prefix_dark()  + 'peds-rms.txt'

    def path_peds_aver_plot(self) :
        return self.path_prefix_dark()  + 'peds-aver-plot.png'

#-----------------------------

    def path_data_scan_psana_cfg(self) :
        return self.path_prefix_data()  + 'data-scan.cfg'

    def path_data_scan_batch_log(self) :
        return self.path_prefix_data() + 'data-scan-batch-log.txt'

    def path_data_scan_monitors_data(self) :
        return self.path_prefix_data() + 'data-scan-mons-data.txt'

    def path_data_scan_monitors_commments(self) :
        return self.path_prefix_data() + 'data-scan-mons-comments.txt'

    def path_data_scan_tstamp_list(self) :
        return self.path_prefix_data() + 'data-scan-tstamp-list.txt'

    def path_data_scan_tstamp_list_tmp(self) :
        return  self.path_data_scan_tstamp_list() + '-tmp'

#-----------------------------

    def path_data_aver_psana_cfg(self) :
        return self.path_prefix_data() + 'data-aver.cfg'

    def path_data_aver_batch_log(self) :
        return self.path_prefix_data() + 'data-aver-batch-log.txt'

    def path_data_ave(self) :
        return self.path_prefix_data() + 'data-ave.txt'

    def path_data_rms(self) :
        return self.path_prefix_data() + 'data-rms.txt'

    def path_data_aver_plot(self) :
        return self.path_prefix_data() + 'data-aver-plot.png'

    def path_data_time_plot(self) :
        return self.path_prefix_data() + 'data-time-plot.png'

    def path_data_mons_plot(self) :
        return self.path_prefix_data() + 'data-mons-plot.png'

    def path_gui_image(self) :
        return self.path_prefix_data() + 'gui-image.png'

#-----------------------------

    def path_prefix_cora(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix_cora.value() 

    def path_cora_split_psana_cfg(self) :
        return self.path_prefix_cora() + '-' + self.str_exp_run_data() + 'split.cfg'

    def path_cora_split_batch_log(self) :
        return self.path_prefix_cora() + '-' + self.str_exp_run_data() + 'split-log.txt'

    def path_cora_split_med(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'med.txt'

    def path_cora_split_time(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'time.txt'

    def path_cora_split_time_ind(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'time-ind.txt'

    def path_cora_split_files(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'b*.txt'

    def path_cora_proc_tau_in(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'tau-in.txt'

    def path_cora_proc_tau_out(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'tau.txt'




    def path_cora_merge_result(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'image-result.txt'

    def path_cora_merge_batch_log(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'merge-log.txt'

#-----------------------------

    def  get_list_of_files_cora_work(self, format='b%04d.bin') :
        self.list_of_files_cora_work = []
        for i in range(cp.bat_img_nparts.value()) :
            suffix = format % (i)
            fname = self.path_prefix_cora() + '-' + self.str_run_data() + suffix
            self.list_of_files_cora_work.append(fname)
        return self.list_of_files_cora_work

#-----------------------------

    def  get_list_of_files_cora_split_work(self) :
        return self.get_list_of_files_cora_work(format='b%04d.bin')

    def get_list_of_files_cora_split_all(self) :
        return self.get_list_of_files_cora_split() + \
               self.get_list_of_files_cora_split_work()

    def  get_list_of_files_cora_split(self) :
        self.list_of_files_cora_split = []
        self.list_of_files_cora_split.append(fnm.path_cora_split_psana_cfg())
        self.list_of_files_cora_split.append(fnm.path_cora_split_batch_log())
        self.list_of_files_cora_split.append(fnm.path_cora_split_med())
        self.list_of_files_cora_split.append(fnm.path_cora_split_time())
        self.list_of_files_cora_split.append(fnm.path_cora_split_time_ind())
        #self.list_of_files_cora_split.append(fnm.path_cora_split_files())
        #self.list_of_files_cora_split.append(fnm.)

        return self.list_of_files_cora_split

#-----------------------------

    def get_list_of_files_cora_proc_work(self) :
        return self.get_list_of_files_cora_work(format='b%04d-result.bin')

    def get_list_of_files_cora_proc_work_log(self) :
        return self.get_list_of_files_cora_work(format='b%04d-result.log')

    def get_list_of_files_cora_proc_main(self) :
        return self.get_list_of_files_cora_proc() + \
               self.get_list_of_files_cora_proc_work()

    def get_list_of_files_cora_proc_all(self) :
        return self.get_list_of_files_cora_proc() + \
               self.get_list_of_files_cora_proc_work() + \
               self.get_list_of_files_cora_proc_work_log()

    def get_list_of_files_cora_proc_check(self) :
        return [fnm.path_cora_proc_tau_out()] + \
               self.get_list_of_files_cora_proc_work() + \
               self.get_list_of_files_cora_proc_work_log()

    def get_list_of_files_cora_proc_browser(self) :
        return self.get_list_of_files_cora_proc() + \
               self.get_list_of_files_cora_proc_work_log()

    def  get_list_of_files_cora_proc(self) :
        self.list_of_files_cora_proc = []
        self.list_of_files_cora_proc.append(fnm.path_cora_proc_tau_in())
        self.list_of_files_cora_proc.append(fnm.path_cora_proc_tau_out())
        return self.list_of_files_cora_proc

#-----------------------------

    def  get_list_of_files_cora_merge(self) :
        self.list_of_files_cora_merge = []
        self.list_of_files_cora_merge.append(fnm.path_cora_merge_result())
        self.list_of_files_cora_merge.append(fnm.path_cora_merge_batch_log())
        return self.list_of_files_cora_merge

#-----------------------------

    def  get_list_of_files_data_aver(self) :
        self.list_of_files_data_aver  = []
        #self.list_of_files_data_aver.append(fnm.path_data_xtc())
        self.list_of_files_data_aver.append(fnm.path_data_scan_psana_cfg())
        self.list_of_files_data_aver.append(fnm.path_data_scan_batch_log())
        self.list_of_files_data_aver.append(fnm.path_data_scan_monitors_data())
        self.list_of_files_data_aver.append(fnm.path_data_scan_monitors_commments())
        self.list_of_files_data_aver.append(fnm.path_data_scan_tstamp_list())
        self.list_of_files_data_aver.append(fnm.path_data_scan_tstamp_list_tmp())

        self.list_of_files_data_aver.append(fnm.path_data_aver_psana_cfg())
        self.list_of_files_data_aver.append(fnm.path_data_aver_batch_log())
        self.list_of_files_data_aver.append(fnm.path_data_ave())
        self.list_of_files_data_aver.append(fnm.path_data_rms())

        self.list_of_files_data_aver.append(fnm.path_data_aver_plot())
        self.list_of_files_data_aver.append(fnm.path_data_time_plot())
        self.list_of_files_data_aver.append(fnm.path_data_mons_plot())
        return self.list_of_files_data_aver

#-----------------------------

    def  get_list_of_files_data(self) :
        self.list_of_files_data  = []
        self.list_of_files_data.append(fnm.path_data_xtc())
        return self.list_of_files_data


#-----------------------------

    def  get_list_of_files_pedestals(self) :
        self.list_of_files_pedestals = []
        #self.list_of_files_pedestals.append(self.path_dark_xtc())
        self.list_of_files_pedestals.append(self.path_peds_scan_psana_cfg())
        self.list_of_files_pedestals.append(self.path_peds_scan_batch_log())
        self.list_of_files_pedestals.append(self.path_peds_scan_tstamp_list())
        self.list_of_files_pedestals.append(self.path_peds_scan_tstamp_list_tmp())
        
        self.list_of_files_pedestals.append(self.path_peds_aver_psana_cfg())
        self.list_of_files_pedestals.append(self.path_peds_aver_batch_log())
        self.list_of_files_pedestals.append(self.path_pedestals_ave())
        self.list_of_files_pedestals.append(self.path_pedestals_rms())

        self.list_of_files_pedestals.append(self.path_peds_aver_plot())
        return self.list_of_files_pedestals

#-----------------------------

    def  get_list_of_files_flatfield(self) :
        self.list_of_files_flatfield = []
        self.list_of_files_flatfield.append(fnm.path_flat())
        self.list_of_files_flatfield.append(fnm.path_flat_plot())
        return self.list_of_files_flatfield

#-----------------------------

    def  get_list_of_files_blamish(self) :
        self.list_of_files_blamish = []
        self.list_of_files_blamish.append(fnm.path_blam())
        self.list_of_files_blamish.append(fnm.path_blam_plot())
        return self.list_of_files_blamish

#-----------------------------

    def  get_list_of_files_service(self) :
        self.list_of_files_service = []
        self.list_of_files_service.append(fnm.path_gui_image())
        return self.list_of_files_service

#-----------------------------

    def get_list_of_files_total(self) :
        self.list_of_files_total  = []
        self.list_of_files_total.append(fnm.path_config_pars())
        self.list_of_files_total += fnm.get_list_of_files_pedestals()
        self.list_of_files_total += fnm.get_list_of_files_flatfield()
        self.list_of_files_total += fnm.get_list_of_files_blamish()
        self.list_of_files_total += fnm.get_list_of_files_data_aver()
        self.list_of_files_total += fnm.get_list_of_files_service()
        return self.list_of_files_total

#-----------------------------

#-----------------------------

fnm = FileNameManager ()

#-----------------------------

if __name__ == "__main__" :

    print 'path_pedestals_xtc()       : ', fnm.path_dark_xtc()
    print 'path_peds_aver_psana_cfg() : ', fnm.path_peds_aver_psana_cfg()
    print 'path_pedestals_ave()       : ', fnm.path_pedestals_ave()
    print 'path_pedestals_rms()       : ', fnm.path_pedestals_rms()
    print 'path_peds_aver_batch_log() : ', fnm.path_peds_aver_batch_log()
    print 'path_peds_scan_psana_cfg() : ', fnm.path_peds_scan_psana_cfg()
    print 'path_peds_scan_batch_log() : ', fnm.path_peds_scan_batch_log()
    print '\n',
    print '\n',
    print 'str_exp_run_dark()   : ', fnm.str_exp_run_dark()
    #print 'str_exp_run_flat()   : ', fnm.str_exp_run_flat()
    print 'str_exp_run_data()   : ', fnm.str_exp_run_data()

    print 'path_data_xtc_all_chunks() : ', fnm.path_data_xtc_all_chunks()
    print 'path_dark_xtc_all_chunks() : ', fnm.path_dark_xtc_all_chunks()

    #list = fnm.get_list_of_files_cora_split_work()
    print '\nfnm.get_list_of_files_cora_split_work():'    
    list =   fnm.get_list_of_files_cora_split_work()
    for fname in list : print fname

    print '\nfnm.get_list_of_files_cora_proc_all():'    
    list =   fnm.get_list_of_files_cora_proc_all()
    for fname in list : print fname

    print '\nfnm.get_list_of_files_cora_merge():'    
    list =   fnm.get_list_of_files_cora_merge()
    for fname in list : print fname

    print '\nfnm.get_list_of_files_cora_proc_check():' 
    list =   fnm.get_list_of_files_cora_proc_check()
    for fname in list : print fname
    
    sys.exit ( 'End of test for FileNameManager' )

#-----------------------------
