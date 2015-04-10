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
        if experiment is None : return 'exp-run-'
        else                  : return experiment + '-' + run_str + '-'

    def str_run_for_xtc_path(self, path) :
        instrument, experiment, run_str, run_num = gu.parse_xtc_path(path)
        if run_str is None : return 'run-'
        else               : return run_str + '-'

#-----------------------------

    def path_prefix(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() 

    def path_prefix_dark(self) :
        return self.path_prefix() + self.str_exp_run_dark()

    def path_prefix_data(self) :
        return self.path_prefix() + self.str_exp_run_data()

    def path_prefix_cora(self) :
        #return cp.dir_work.value() + '/' + cp.fname_prefix_cora.value() 
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + 'cora'

#-----------------------------

    def path_blem(self) :
        return cp.dname_blem.value() + '/' + cp.fname_blem.value()

    def path_blem_prefix(self) :
        return os.path.splitext(self.path_blem())[0]

    def path_blem_plot(self) :
        return self.path_prefix() + self.str_exp_run_data() + 'blem-plot.png'

    def path_flat(self) :
        return cp.dname_flat.value() + '/' + cp.fname_flat.value()

    def path_flat_plot(self) :
        return self.path_prefix() + self.str_exp_run_data() + 'flat-plot.png'

#-----------------------------

    def path_tau_list(self) :
        return  cp.ana_tau_list_dname.value() + '/' + cp.ana_tau_list_fname.value() 

#-----------------------------

    def path_roi_mask(self) :
        return  cp.ana_mask_dname.value() + '/' + cp.ana_mask_fname.value() 

    def path_roi_mask_prefix(self) :
        return os.path.splitext(self.path_roi_mask())[0]

    def path_roi_mask_plot(self) :
        return self.path_roi_mask_prefix() + '-plot.png'

#-----------------------------

    def path_hotpix_frac(self) :
        return self.path_prefix() + self.str_exp_run_dark() + 'hotpix-frac.txt'

    def path_hotpix_mask(self) :
        return self.path_prefix() + self.str_exp_run_dark() + 'hotpix-mask-thr-' \
               + str(cp.mask_hot_thr.value()) + 'ADU.txt'

    def path_hotpix_mask_prefix(self) :
        return os.path.splitext(self.path_hotpix_mask())[0]

    def path_hotpix_mask_plot(self) :
        return self.path_hotpix_mask_prefix() + '-plot.png'

#-----------------------------

    def path_satpix_frac(self) :
        return self.path_prefix() + self.str_exp_run_data() + 'satpix-frac-level-' \
               + str(cp.ccdset_adcsatu.value()) + 'ADU.txt' 

    def path_satpix_mask(self) :
        return self.path_prefix() + self.str_exp_run_data() + 'satpix-mask-level-' \
               + str(cp.ccdset_adcsatu.value()) + 'ADU.txt' 

    def path_satpix_mask_prefix(self) :
        return os.path.splitext(self.path_satpix_mask())[0]

    def path_satpix_mask_plot(self) :
        return self.path_satpix_mask_prefix() + '-plot.png'

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
        return self.path_prefix_dark()  + 'peds-aver.cfg'

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

    def path_data_raw_ave(self) :
        return self.path_prefix_data() + 'data-raw-ave.txt'

    def path_data_raw_rms(self) :
        return self.path_prefix_data() + 'data-raw-rms.txt'

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

    def path_cora_split_psana_cfg(self) :
        return self.path_prefix_cora() + '-' + self.str_exp_run_data() + 'split.cfg'

    def path_cora_split_batch_log(self) :
        return self.path_prefix_cora() + '-' + self.str_exp_run_data() + 'split-log.txt'

    def path_cora_split_imon_cfg(self) :
        return self.path_prefix_cora() + '-' + self.str_exp_run_data() + 'imon-cfg.txt'

    def path_cora_split_med(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'med.txt'

    def path_cora_split_time(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'time.txt'

    def path_cora_split_time_ind(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'time-ind.txt'

    def path_cora_split_files(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'b*.txt'

    def path_cora_split_map_static_q(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'map-static-q.txt'

    def path_cora_split_q_ave_static(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'q-ave-static.txt'

    def path_cora_split_int_static_q(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'int-static-q.txt'

    def path_cora_proc_tau_in(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'tau-in.txt'

    def path_cora_proc_tau_out(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'tau.txt'

    def path_cora_merge_tau(self) :
        return self.path_cora_proc_tau_out()

    def path_cora_merge_result(self) :
        return self.path_prefix_cora() + '-' + self.str_run_data() + 'image-result.bin'

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
               #self.get_list_of_files_cora_split_intensity()

    def  get_list_of_files_cora_split(self) :
        self.list_of_files_cora_split = []
        self.list_of_files_cora_split.append(fnm.path_cora_split_imon_cfg())
        self.list_of_files_cora_split.append(fnm.path_cora_split_psana_cfg())
        self.list_of_files_cora_split.append(fnm.path_cora_split_batch_log())
        self.list_of_files_cora_split.append(fnm.path_cora_split_med())
        self.list_of_files_cora_split.append(fnm.path_cora_split_time())
        self.list_of_files_cora_split.append(fnm.path_cora_split_time_ind())
        self.list_of_files_cora_split.append(fnm.path_data_ave())
        self.list_of_files_cora_split.append(fnm.path_data_rms())
        #self.list_of_files_cora_split.append(fnm.path_cora_split_files())
        #self.list_of_files_cora_split.append(fnm.)
        return self.list_of_files_cora_split

    def  get_list_of_files_cora_split_intensity(self) :
        self.list_of_files_cora_split_intensity = []
        self.list_of_files_cora_split_intensity.append(fnm.path_cora_split_map_static_q())
        self.list_of_files_cora_split_intensity.append(fnm.path_cora_split_int_static_q())
        return self.list_of_files_cora_split_intensity

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
        self.list_of_files_cora_merge.append(fnm.path_cora_merge_tau())
        self.list_of_files_cora_merge.append(fnm.path_cora_merge_result())
        self.list_of_files_cora_merge.append(fnm.path_cora_merge_batch_log())
        return self.list_of_files_cora_merge

    def  get_list_of_files_cora_merge_main(self) :
        self.list_of_files_cora_merge_main = []
        self.list_of_files_cora_merge_main.append(fnm.path_cora_merge_result())
        self.list_of_files_cora_merge_main.append(fnm.path_cora_merge_batch_log())
        return self.list_of_files_cora_merge_main

#-----------------------------

    def  get_list_of_files_data_scan(self) :
        self.list_of_files_data_scan = []
        self.list_of_files_data_scan.append(fnm.path_data_scan_psana_cfg())
        self.list_of_files_data_scan.append(fnm.path_data_scan_batch_log())
        self.list_of_files_data_scan.append(fnm.path_data_scan_monitors_data())
        self.list_of_files_data_scan.append(fnm.path_data_scan_monitors_commments())
        self.list_of_files_data_scan.append(fnm.path_data_scan_tstamp_list())
        self.list_of_files_data_scan.append(fnm.path_data_scan_tstamp_list_tmp())
        return self.list_of_files_data_scan


    def  get_list_of_files_data_aver_short(self) :
        self.list_of_files_data_aver_short = []
        self.list_of_files_data_aver_short.append(fnm.path_data_aver_psana_cfg())
        self.list_of_files_data_aver_short.append(fnm.path_data_aver_batch_log())
        self.list_of_files_data_aver_short.append(fnm.path_data_raw_ave())
        self.list_of_files_data_aver_short.append(fnm.path_data_raw_rms())
        return self.list_of_files_data_aver_short


    def  get_list_of_files_data_aver(self) :
        self.list_of_files_data_aver = self.get_list_of_files_data_aver_short()
        self.list_of_files_data_aver.append(fnm.path_satpix_frac())
        self.list_of_files_data_aver.append(fnm.path_satpix_mask())
        self.list_of_files_data_aver.append(fnm.path_data_aver_plot())
        self.list_of_files_data_aver.append(fnm.path_data_time_plot())
        self.list_of_files_data_aver.append(fnm.path_data_mons_plot())
        #self.list_of_files_data_aver.append(fnm.path_hotpix_frac())
        #self.list_of_files_data_aver.append(fnm.path_hotpix_mask())
        #self.list_of_files_data_aver.append(fnm.path_data_xtc())
        return self.list_of_files_data_aver

#-----------------------------

    def  get_list_of_files_data(self) :
        self.list_of_files_data  = self.get_list_of_files_data_scan()
        self.list_of_files_data += self.get_list_of_files_data_aver()
        #self.list_of_files_data.append(fnm.path_data_xtc())
        return self.list_of_files_data

#-----------------------------

    def  get_list_of_files_peds_scan(self) :
        self.list_of_files_peds_scan = []
        #self.list_of_files_peds_scan.append(self.path_dark_xtc())
        self.list_of_files_peds_scan.append(self.path_peds_scan_psana_cfg())
        self.list_of_files_peds_scan.append(self.path_peds_scan_batch_log())
        self.list_of_files_peds_scan.append(self.path_peds_scan_tstamp_list())
        self.list_of_files_peds_scan.append(self.path_peds_scan_tstamp_list_tmp())
        return self.list_of_files_peds_scan


    def  get_list_of_files_peds_aver(self) :
        self.list_of_files_peds_aver = []
        self.list_of_files_peds_aver.append(self.path_peds_aver_psana_cfg())
        self.list_of_files_peds_aver.append(self.path_peds_aver_batch_log())
        self.list_of_files_peds_aver.append(self.path_pedestals_ave())
        self.list_of_files_peds_aver.append(self.path_pedestals_rms())
        self.list_of_files_peds_aver.append(self.path_hotpix_mask())
        return self.list_of_files_peds_aver


    def  get_list_of_files_pedestals(self) :
        self.list_of_files_pedestals = self.get_list_of_files_peds_scan()
        self.list_of_files_pedestals+= self.get_list_of_files_peds_aver()
        self.list_of_files_pedestals.append(self.path_peds_aver_plot())
        #self.list_of_files_pedestals.append(self.path_dark_xtc())
        return self.list_of_files_pedestals

#-----------------------------

    def  get_list_of_files_flatfield(self) :
        self.list_of_files_flatfield = []
        self.list_of_files_flatfield.append(fnm.path_flat())
        self.list_of_files_flatfield.append(fnm.path_flat_plot())
        return self.list_of_files_flatfield

#-----------------------------

    def  get_list_of_files_blemish(self) :
        self.list_of_files_blemish = []
        self.list_of_files_blemish.append(fnm.path_blem())
        self.list_of_files_blemish.append(fnm.path_blem_plot())
        return self.list_of_files_blemish

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
        self.list_of_files_total += fnm.get_list_of_files_blemish()
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

    print 'fnm.path_hotpix_mask() : ', fnm.path_hotpix_mask()
    print 'fnm.path_satpix_mask() : ', fnm.path_satpix_mask()
    
    sys.exit ( 'End of test for FileNameManager' )

#-----------------------------
