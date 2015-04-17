#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module BatchJobPedestals...
#
#------------------------------------------------------------------------

"""Deals with batch jobs for dark runs (pedestals)

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

from BatchJob import *
from FileNameManager          import fnm
from ConfigFileGenerator      import cfg
from ConfigParametersForApp   import cp

#-----------------------------

class BatchJobPedestals (BatchJob) : 
    """Deals with batch jobs for dark runs (pedestals).
    """

    def __init__ (self, run_number) :
        """
        @param fname  the file name for output log file
        """

        self.run_number = run_number

        BatchJob.__init__(self)

        self.job_id_peds_str = None
        self.job_id_scan_str = None

        self.time_peds_job_submitted = None
        self.time_scan_job_submitted = None

        self.procDarkStatus  = 0 # 0=inactive, 1=scan, 2=averaging, 3=both

        #self.opt = ' -o psana.l3t-accept-only=0'
        self.opt = ''

#-----------------------------

    def getParent(self):
        # parent depends on self.run_number
        item, widg = cp.dict_guidarklistitem[self.run_number]
        return widg.gui_run

#-----------------------------

    def exportLocalPars(self):
        #self.getParent().exportLocalPars()
        cp.str_run_number.setValue('%04d' % self.run_number)

#-----------------------------

    def submit_batch_for_peds_scan(self) :
        self.exportLocalPars()

        if not self.job_can_be_submitted(self.job_id_scan_str, self.time_scan_job_submitted, 'scan') : return
        self.time_scan_job_submitted = gu.get_time_sec()

        cfg.make_psana_cfg_file_for_peds_scan()

        command      = 'psana -c ' + fnm.path_peds_scan_psana_cfg() + ' ' + fnm.path_to_xtc_files_for_run() # fnm.path_dark_xtc_cond()
        queue        = self.queue.value()
        bat_log_file = fnm.path_peds_scan_batch_log()

        #print 'command     :', command
        #print 'queue       :', queue
        #print 'bat_log_file:', bat_log_file

        self.job_id_scan_str, out, err = gu.batch_job_submit(command, queue, bat_log_file)
        self.procDarkStatus ^= 1 # set bit to 1

        if err != '' :
            self.stop_auto_processing(is_stop_on_button_click=False)        
            logger.warning('Autoprocessing for run %s is stopped due to batch submission error!!!' % self.str_run_number, __name__)
        #print 'self.procDarkStatus: ', self.procDarkStatus

#-----------------------------

    def command_for_peds_scan(self) :

        cfg.make_psana_cfg_file_for_peds_scan()

        #command = 'env'
        command = 'psana -c ' + fnm.path_peds_scan_psana_cfg() + self.opt 
        command_seq = command.split()

        msg = 'Scan xtc file(s) using command:\n%s' % command \
            + '\nand save results in the log-file: %s' % fnm.path_peds_scan_batch_log()
        logger.info(msg, __name__)
        
        err = gu.subproc_in_log(command_seq, fnm.path_peds_scan_batch_log()) # , shell=True)
        if err != '' :
            logger.error('\nerr: %s' % (err), __name__)
            self.stop_auto_processing(is_stop_on_button_click=False)
            logger.warning('Autoprocessing for run %s is stopped due to error at execution of the scan command' % self.str_run_number, __name__)
        else :
            logger.info('Scan for run %s is completed' % self.str_run_number, __name__)

#-----------------------------

    def command_for_peds_aver(self) :

        if not cfg.make_psana_cfg_file_for_peds_aver() :            
            logger.warning('INTERACTIVE JOB IS NOT STARTED !!!', __name__)
            return False

        command = 'psana -c ' + fnm.path_peds_aver_psana_cfg() # + ' ' + fnm.path_to_xtc_files_for_run() # fnm.path_dark_xtc_cond()
        command_seq = command.split()

        msg = 'Avereging xtc file(s) using command:\n%s' % command \
            + '\nand save results in the log-file: %s' % fnm.path_peds_aver_batch_log()
        logger.info(msg, __name__)
        
        err = gu.subproc_in_log(command_seq, fnm.path_peds_aver_batch_log()) # , shell=True)
        if err != '' :
            logger.warning('\nWarning/error message from subprocess:\n%s' % (err), __name__)
            return False
        else :
            logger.info('Avereging for run %s is completed' % self.str_run_number, __name__)
            return True

#-----------------------------

    def is_good_lsf(self) :
        """Checks and returns LSF status"""
        queue = self.queue.value()
        farm = cp.dict_of_queue_farm[queue]
        msg, status = gu.msg_and_status_of_lsf(farm)

        ###status = False # FOR TEST PURPOSE ONLY!!!

        if status :
            logger.info('LSF status is ok for queue: %s on farm: %s' % (queue, farm), __name__)
        else :
            msgi = '\nLSF status for queue: %s on farm: %s\n%s\n' % (queue, farm, msg)
            msgw = 'LSF farm for queue %s IS BUSY OR DOES NOT WORK !!!\n' % (queue)
            logger.info(msgi, __name__)
            logger.warning(msgw, __name__)

        msg, status = gu.msg_and_status_of_queue(queue)
        #logger.info(msg, __name__)
        
        return status

#-----------------------------

    def submit_batch_for_peds_aver(self) :
        self.exportLocalPars() # export run_number to cp.str_run_number

        if not self.job_can_be_submitted(self.job_id_peds_str, self.time_peds_job_submitted, 'peds') : return        
        self.time_peds_job_submitted = gu.get_time_sec()

        self.command_for_peds_scan()

        if not cfg.make_psana_cfg_file_for_peds_aver() :            
            self.stop_auto_processing(is_stop_on_button_click=False)
            logger.warning('BATCH JOB IS NOT SUBMITTED !!!', __name__)
            return False

        if not self.is_good_lsf() :
            self.stop_auto_processing(is_stop_on_button_click=False)
            logger.warning('BATCH JOB IS NOT SUBMITTED !!!', __name__)
            return False

        #command      = 'psana -c ' + fnm.path_peds_aver_psana_cfg() + ' ' + fnm.path_to_xtc_files_for_run() # fnm.path_dark_xtc_cond()
        #command      = 'psana -c ' + fnm.path_peds_aver_psana_cfg() + self.opt + ' ' + fnm.path_to_xtc_files_for_run() # fnm.path_dark_xtc_cond()
        command      = 'psana -c ' + fnm.path_peds_aver_psana_cfg() + self.opt
        queue        = self.queue.value()
        bat_log_file = fnm.path_peds_aver_batch_log()

        self.job_id_peds_str, out, err = gu.batch_job_submit(command, queue, bat_log_file)
        self.procDarkStatus ^= 2 # set bit to 1

        if err != 'Warning: job being submitted without an AFS token.' :
            #logger.info('This job is running on LCLS NFS, it does not need in AFS, ignore warning and continue.', __name__)
            return True

        elif err != '' :
            self.stop_auto_processing(is_stop_on_button_click=False)
            logger.warning('Autoprocessing for run %s is stopped due to batch submission error!!!' % self.str_run_number, __name__)
            logger.warning('BATCH JOB IS NOT SUBMITTED !!!', __name__)
            return False

        return True

#-----------------------------

    def kill_batch_job_for_peds_scan(self) :
        self.kill_batch_job(self.job_id_scan_str, 'for peds scan')

    def kill_batch_job_for_peds_aver(self) :
        self.kill_batch_job(self.job_id_peds_str, 'for peds aver')

    def kill_all_batch_jobs(self):
        logger.debug('kill_all_batch_jobs', __name__)
        self.kill_batch_job_for_peds_scan()
        self.kill_batch_job_for_peds_aver()

#-----------------------------

    def remove_files_pedestals(self) :
        self.remove_files_for_list(self.get_list_of_files_peds(), 'of dark run / pedestals:')

#-----------------------------

    def get_list_of_files_peds(self) :
        self.exportLocalPars() # export run_number to cp.str_run_number        
        return fnm.get_list_of_files_peds()

#-----------------------------

    def get_list_of_files_peds_scan(self) :
        self.exportLocalPars() # export run_number to cp.str_run_number        
        return fnm.get_list_of_files_peds_scan()

#-----------------------------

    def get_list_of_files_peds_aver(self) :
        self.exportLocalPars() # export run_number to cp.str_run_number        
        lst_of_srcs = cp.blsp.list_of_sources_for_selected_detectors ()
        list_of_fnames = fnm.get_list_of_files_peds_aver() \
             + gu.get_list_of_files_for_list_of_insets(fnm.path_peds_ave(), lst_of_srcs) \
             + gu.get_list_of_files_for_list_of_insets(fnm.path_peds_rms(), lst_of_srcs)

        return list_of_fnames

#-----------------------------

    def get_list_of_files_peds_essential(self) :
        self.exportLocalPars() # export run_number to cp.str_run_number
        return   self.get_list_of_files_peds_scan() \
               + self.get_list_of_files_peds_aver()

#-----------------------------

    def status_for_peds_files_essential(self) :
        return self.status_for_files(self.get_list_of_files_peds_essential(), 'of dark scan and ave essential')

#-----------------------------

    def status_for_peds_scan_files(self, comment='') :
        stat, msg = self.status_and_string_for_files(self.get_list_of_files_peds_scan(), comment)
        if stat and self.procDarkStatus & 1 : self.procDarkStatus ^= 1 # set bit to 0
        return stat, msg
    
    def status_for_peds_aver_files(self, comment='') :
        stat, msg = self.status_and_string_for_files(self.get_list_of_files_peds_aver(), comment)
        if stat and self.procDarkStatus & 2 : self.procDarkStatus ^= 2 # set bit to 0
        return stat, msg

    def status_batch_job_for_peds_scan(self) :
        return self.get_batch_job_status_and_string(self.job_id_scan_str, self.time_scan_job_submitted)

    def status_batch_job_for_peds_aver(self) :
        return self.get_batch_job_status_and_string(self.job_id_peds_str, self.time_peds_job_submitted)

    def job_id_peds_aver_str(self) :
        return str(self.job_id_peds_str)

#-----------------------------
#-----------------------------
#----- AUTO-PROCESSING -------
#-----------------------------
#-----------------------------

    def on_auto_processing_start(self):
        logger.info('on_auto_processing_start()', __name__)
        #self.onRunScan() # scan is not needed if info is available form RegDB
        self.onRunAver()
        pass


    def on_auto_processing_stop(self):
        logger.info('on_auto_processing_stop()', __name__)
        self.kill_all_batch_jobs()


    def switch_stop_go_button(self):
        logger.debug('switch_stop_go_button', __name__)
        try :
            self.getParent().onStop()
        except :
            logger.warning('Lost connection to the object for run %d. Click on run string to reset buttons....' % self.run_number, __name__)
        #try : self.parent.onStop() # <- but this is not necessarily a parent !!!!
        #except : pass

#-----------------------------

    def on_auto_processing_status(self):

        self.exportLocalPars() # export run_number to cp.str_run_number        

        if self.autoRunStage == 1 :

            self.status_bj_scan, str_bj_scan = self.status_batch_job_for_peds_scan()
            #print 'self.status_bj_scan, str_bj_scan =', str(self.status_bj_scan), str_bj_scan
            msg = 'Stage %s for run %s, %s' % (self.autoRunStage, self.str_run_number, str_bj_scan)
            logger.info(msg, __name__)

            if self.status_bj_scan == 'EXIT' :
                self.stop_auto_processing( is_stop_on_button_click=False )
                logger.warning('PROCESSING IS STOPPED for run %s due to status: %s - CHECK LSF!!!' % (self.str_run_number, self.status_bj_scan), __name__)

            self.status_scan, fstatus_str_scan = self.status_for_peds_scan_files(comment='')
            #print 'self.status_scan, fstatus_str_scan = ', self.status_scan, fstatus_str_scan

            if self.status_scan :            
                logger.info('on_auto_processing_status: Scan is completed, begin averaging', __name__)
                
                cp.blsp.print_list_of_types_and_sources()
                
                if cp.blsp.get_list_of_sources() == [] :
                    self.stop_auto_processing( is_stop_on_button_click=False )
                    logger.warning('on_auto_processing_status: Scan for run %s did not find data in xtc file for this detector. PROCESSING IS STOPPED!!!' % self.str_run_number, __name__)
                    return
                
                self.onRunAver()

        elif self.autoRunStage == 2 :

            self.status_bj_aver, str_bj_aver = self.status_batch_job_for_peds_aver()
            msg = 'Stage %s for run %s, %s' % (self.autoRunStage, self.str_run_number, str_bj_aver)
            logger.info(msg, __name__)

            if self.status_bj_aver == 'EXIT' :
                self.stop_auto_processing( is_stop_on_button_click=False )
                logger.warning('PROCESSING IS STOPPED for run %s due to status: %s - CHECK LSF!!!' % (self.str_run_number,self.status_bj_aver), __name__)

            self.status_aver, fstatus_str_aver = self.status_for_peds_aver_files(comment='')
            #print 'self.status_aver, fstatus_str_aver = ', self.status_aver, fstatus_str_aver

            if self.status_aver : 
                logger.info('on_auto_processing_status: Averaging is completed, stop processing for run %s.' % self.str_run_number, __name__)
                self.stop_auto_processing( is_stop_on_button_click=False )

        else :
            msg = 'NONRECOGNIZED PROCESSING STAGE %s for run %s !!!' % (self.autoRunStage,self.str_run_number)
            logger.warning(msg, __name__)
            

#-----------------------------

    def onRunScan(self):
        logger.debug('onRunScan', __name__)
        self.submit_batch_for_peds_scan()
        self.autoRunStage = 1

#-----------------------------

    def onRunAver(self):
        logger.debug('onRunAver', __name__)
        if self.submit_batch_for_peds_aver() : self.autoRunStage = 2
        else : self.autoRunStage = 0

#-----------------------------

#bjpeds = BatchJobPedestals (1)

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
    #bjpeds.check_work_files_for_pedestals()

    sys.exit ( 'End of test for BatchJobPedestals' )

#-----------------------------
