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

from BatchJob import *

#-----------------------------

class BatchJobPedestals (BatchJob) : 
    """Deals with batch jobs for dark runs (pedestals).
    """

    def __init__ (self) :
        """
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

        command      = 'psana -c ' + fnm.path_peds_scan_psana_cfg() + ' ' + fnm.path_to_xtc_files_for_run() # fnm.path_dark_xtc_cond()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_peds_scan_batch_log()

        #print 'command     :', command
        #print 'queue       :', queue
        #print 'bat_log_file:', bat_log_file

        self.job_id_scan_str, out, err = gu.batch_job_submit(command, queue, bat_log_file)
        cp.procDarkStatus ^= 1 # set bit to 1

        if err != '' :
            self.stop_auto_processing(is_stop_on_button_click=False)        
            logger.warning('Autoprocessing is stopped due to batch submission error!!!', __name__)
        #print 'cp.procDarkStatus: ', cp.procDarkStatus

#-----------------------------

    def submit_batch_for_peds_aver(self) :

        if not self.job_can_be_submitted(self.job_id_peds_str, self.time_peds_job_submitted, 'peds') : return        
        self.time_peds_job_submitted = gu.get_time_sec()

        self.make_psana_cfg_file_for_peds_aver()

        command      = 'psana -c ' + fnm.path_peds_aver_psana_cfg() + ' ' + fnm.path_to_xtc_files_for_run() # fnm.path_dark_xtc_cond()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_peds_aver_batch_log()

        self.job_id_peds_str, out, err = gu.batch_job_submit(command, queue, bat_log_file)
        cp.procDarkStatus ^= 2 # set bit to 1

        if err != '' :
            self.stop_auto_processing(is_stop_on_button_click=False)
            logger.warning('Autoprocessing is stopped due to batch submission error!!!', __name__)
        #print 'cp.procDarkStatus: ', cp.procDarkStatus

#-----------------------------

    def check_batch_job_for_peds_aver(self) :
        self.check_batch_job(self.job_id_peds_str, 'peds')

    def check_batch_job_for_peds_scan(self) :
        self.check_batch_job(self.job_id_scan_str, 'scan')

#-----------------------------

    def kill_batch_job_for_peds_scan(self) :
        self.kill_batch_job(self.job_id_scan_str, 'for peds scan')

    def kill_batch_job_for_peds_aver(self) :
        self.kill_batch_job(self.job_id_peds_str, 'for peds aver')

#-----------------------------

    def print_work_files_for_pedestals(self) :
        self.print_files_for_list(fnm.get_list_of_files_peds(),'of dark run / pedestals:')

#-----------------------------

    def check_work_files_for_pedestals(self) :
        self.check_files_for_list(fnm.get_list_of_files_peds(),'of dark run / pedestals:')

#-----------------------------

    def remove_files_pedestals(self) :
        self.remove_files_for_list(fnm.get_list_of_files_peds(),'of dark run / pedestals:')

#-----------------------------

    def get_list_of_files_peds_aver(self) :
        list_of_fnames = fnm.get_list_of_files_peds_aver() \
             + blsp.get_list_of_files_for_all_sources(fnm.path_peds_ave()) \
             + blsp.get_list_of_files_for_all_sources(fnm.path_peds_rms())
        return list_of_fnames

    def status_for_pedestal_file(self) :
        fname  = fnm.path_peds_ave()
        status = os.path.lexists(fname)
        logger.info('Status: pedestal file ' + fname + ' ' + self.dict_status[status], __name__) 
        return status

    def status_for_peds_aver_files(self) :
        stat = self.status_for_files(self.get_list_of_files_peds_aver(), comment='of peds average: ')
        if stat and cp.procDarkStatus & 1 : cp.procDarkStatus ^= 1 # set bit to 0
        return stat

    def status_for_peds_scan_files(self) :
        stat = self.status_for_files(fnm.get_list_of_files_peds_scan(), comment='of peds scan: ')
        if stat and cp.procDarkStatus & 2 : cp.procDarkStatus ^= 2 # set bit to 0
        return stat

#-----------------------------

    def status_for_peds_scan_files(self, comment='') :
        stat, msg = self.status_and_string_for_files(fnm.get_list_of_files_peds_scan(), comment)
        if stat and cp.procDarkStatus & 1 : cp.procDarkStatus ^= 1 # set bit to 0
        return stat, msg
    
    def status_for_peds_aver_files(self, comment='') :
        stat, msg = self.status_and_string_for_files(self.get_list_of_files_peds_aver(), comment)
        if stat and cp.procDarkStatus & 2 : cp.procDarkStatus ^= 2 # set bit to 0
        return stat, msg

    def status_batch_job_for_peds_scan(self) :
        return self.get_batch_job_status_and_string(self.job_id_scan_str, self.time_scan_job_submitted)

    def status_batch_job_for_peds_aver(self) :
        return self.get_batch_job_status_and_string(self.job_id_peds_str, self.time_peds_job_submitted)

#-----------------------------
#-----------------------------
#----- AUTO-PROCESSING -------
#-----------------------------
#-----------------------------

    def on_auto_processing_start(self):
        logger.info('on_auto_processing_start()', __name__)
        #self.remove_files_peds_all()
        self.onRunScan()
        pass


    def on_auto_processing_stop(self):
        logger.info('on_auto_processing_stop()', __name__)
        self.kill_all_batch_jobs()

#-----------------------------

    def on_auto_processing_status_v1(self):
        logger.info('Auto run stage '+str(cp.autoRunStatus), __name__)

        self.status_scan, fstatus_str_scan = self.status_for_peds_scan_files(comment='')
        self.status_aver, fstatus_str_aver = self.status_for_peds_aver_files(comment='')

        #print 'self.status_scan, fstatus_str_scan = ', self.status_scan, fstatus_str_scan
        #print 'self.status_aver, fstatus_str_aver = ', self.status_aver, fstatus_str_aver

        if   cp.autoRunStatus == 1 and self.status_scan :            
            logger.info('on_auto_processing_status: Scan is completed, begin averaging', __name__)

            blsp.parse_batch_log_peds_scan() # defines the blsp.list_of_sources
            
            if blsp.list_of_sources == [] :
                self.stop_auto_processing( is_stop_on_button_click=False )
                logger.warning('on_auto_processing_status: Scan did not find data in xtc file for this detector. PROCESSING IS STOPPED!!!', __name__)
                return

            self.onRunAver()

        elif cp.autoRunStatus == 2 and self.status_aver : 
            logger.info('on_auto_processing_status: Averaging is completed, stop processing.', __name__)
            self.stop_auto_processing( is_stop_on_button_click=False )

#-----------------------------

    def on_auto_processing_status(self):

        if cp.autoRunStatus == 1 :

            self.status_bj_scan, str_bj_scan = self.status_batch_job_for_peds_scan()
            #print 'self.status_bj_scan, str_bj_scan =', str(self.status_bj_scan), str_bj_scan
            msg = 'Stage %s, %s' % (cp.autoRunStatus, str_bj_scan)
            logger.info(msg, __name__)

            if self.status_bj_scan == 'EXIT' :
                self.stop_auto_processing( is_stop_on_button_click=False )
                logger.warning('PROCESSING IS STOPPED due to status: %s - CHECK LSF!!!' % self.status_bj_scan, __name__)

            self.status_scan, fstatus_str_scan = self.status_for_peds_scan_files(comment='')
            #print 'self.status_scan, fstatus_str_scan = ', self.status_scan, fstatus_str_scan

            if self.status_scan :            
                logger.info('on_auto_processing_status: Scan is completed, begin averaging', __name__)
                
                blsp.parse_batch_log_peds_scan() # defines the blsp.list_of_sources
                
                if blsp.list_of_sources == [] :
                    self.stop_auto_processing( is_stop_on_button_click=False )
                    logger.warning('on_auto_processing_status: Scan did not find data in xtc file for this detector. PROCESSING IS STOPPED!!!', __name__)
                    return
                
                self.onRunAver()

        elif cp.autoRunStatus == 2 :

            self.status_bj_aver, str_bj_aver = self.status_batch_job_for_peds_aver()
            msg = 'Stage %s, %s' % (cp.autoRunStatus, str_bj_aver)
            logger.info(msg, __name__)

            if self.status_bj_aver == 'EXIT' :
                self.stop_auto_processing( is_stop_on_button_click=False )
                logger.warning('PROCESSING IS STOPPED due to status: %s - CHECK LSF!!!' % self.status_bj_aver, __name__)

            self.status_aver, fstatus_str_aver = self.status_for_peds_aver_files(comment='')
            #print 'self.status_aver, fstatus_str_aver = ', self.status_aver, fstatus_str_aver

            if self.status_aver : 
                logger.info('on_auto_processing_status: Averaging is completed, stop processing.', __name__)
                self.stop_auto_processing( is_stop_on_button_click=False )

        else :
            msg = 'NONRECOGNIZED PROCESSING STAGE %s !!!' % cp.autoRunStatus
            logger.warning(msg, __name__)
            

#-----------------------------


    def kill_all_batch_jobs(self):
        logger.debug('kill_all_batch_jobs', __name__)
        self.kill_batch_job_for_peds_scan()
        self.kill_batch_job_for_peds_aver()

#-----------------------------

    def onRunScan(self):
        logger.debug('onRunScan', __name__)
        self.submit_batch_for_peds_scan()
        cp.autoRunStatus = 1

#-----------------------------

    def onRunAver(self):
        logger.debug('onRunAver', __name__)
        self.submit_batch_for_peds_aver()
        cp.autoRunStatus = 2

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
