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

from BatchJob import *
from PyQt4 import QtGui, QtCore # need it in order to use QtCore.QObject for connect

#-----------------------------

class BatchJobCorAna( BatchJob, QtCore.QObject ) : # need in QtCore.QObject in order to connect to signals
    """Deals with batch jobs for correlation analysis.
    """

    def __init__ (self) :
        """Constructor.
        @param fname the file name for ...
        """

        BatchJob.__init__(self)
        QtCore.QObject.__init__(self, None)

        self.job_id_cora_split = None
        self.time_sub_split    = None

        self.job_id_cora_merge = None
        self.time_sub_merge    = None

        self.nparts            = None
        self.init_list_for_proc()
        
#-----------------------------

    def init_list_for_proc(self) :
        """Creates the empty list for proc. containing ing, jobid and time for all processes"""
        if cp.bat_img_nparts.value() == self.nparts : return
        self.nparts = cp.bat_img_nparts.value()
        #print 'self.nparts:', self.nparts

        self.list_for_proc = []
        for i in range(self.nparts) :
            self.list_for_proc.append([i, None, None])

        #print 'self.list_for_proc =', self.list_for_proc

#-----------------------------

    def     make_psana_cfg_file_for_cora_split(self) :
        cfg.make_psana_cfg_file_for_cora_split()

#-----------------------------

    def submit_batch_for_cora_split(self) :

        if not self.job_can_be_submitted(self.job_id_cora_split, self.time_sub_split, 'cor. ana. split') : return
        self.time_sub_split = gu.get_time_sec()

        self.make_psana_cfg_file_for_cora_split()

        command  = 'psana -c ' + fnm.path_cora_split_psana_cfg() + ' ' + fnm.path_data_xtc_cond()
        queue    = cp.bat_queue.value()
        log_file = fnm.path_cora_split_batch_log()

        #print "!!!!!!!!! WARNING gu.batch_job_submit(...) IS COMMENTED in BatchJobCorAna.py !!!!!!!!! "
        self.job_id_cora_split, out, err = gu.batch_job_submit(command, queue, log_file)

#-----------------------------

    def submit_batch_for_cora_proc(self, ind) :

        self.init_list_for_proc()

        i, job_id, time_sub = self.list_for_proc[ind] 

        if not self.job_can_be_submitted(job_id, time_sub, 'cor. ana. proc') : return
        time_sub = gu.get_time_sec()

        fname    = fnm.get_list_of_files_cora_split_work()[ind]
        #tname    = fnm.path_cora_proc_tau_in()
        tname    = fnm.path_tau_list()
        log_file = fnm.get_list_of_files_cora_proc_work_log()[ind]

        command  = 'corana -f ' + fname # + ' -l ' + log_file
        if cp.ana_tau_list_type.value() == 'file' and os.path.exists(tname) : command +=   ' -t ' + tname
        queue    = cp.bat_queue.value()

        #print 'command  =', command
        #print 'log_file =', log_file, '\n'
  
        job_id, out, err = gu.batch_job_submit(command, queue, log_file)
        self.list_for_proc[ind] = [i, job_id, time_sub]

#-----------------------------

    def submit_batch_for_cora_merge(self) :

        if not self.job_can_be_submitted(self.job_id_cora_merge, self.time_sub_merge, 'cor. ana. merge') : return
        self.time_sub_merge = gu.get_time_sec()

        fname    = fnm.get_list_of_files_cora_proc_work()[0]
        tname    = fnm.path_cora_merge_tau()
        log_file = fnm.path_cora_merge_batch_log()

        command  = 'corana_merge -f ' + fname + ' -t ' + tname
        queue    = cp.bat_queue.value()

        #print 'command =', command
        self.job_id_cora_merge, out, err = gu.batch_job_submit(command, queue, log_file)

#-----------------------------

    def check_batch_job_for_cora_split(self) :
        self.check_batch_job(self.job_id_cora_split, 'split')

    def kill_batch_job_for_cora_split(self) :
        self.kill_batch_job(self.job_id_cora_split, 'for split')

    def status_for_cora_split_files(self, comment='of split: ') :
        return self.status_and_string_for_files(fnm.get_list_of_files_cora_split_all(), comment )

    def status_batch_job_for_cora_split(self) :
        return self.get_batch_job_status_and_string(self.job_id_cora_split, self.time_sub_split)

#-----------------------------

    def status_for_cora_proc_files(self, comment='of proc: ') :
        return self.status_and_string_for_files(fnm.get_list_of_files_cora_proc_check(), comment )

    def status_batch_job_for_cora_proc(self, ind) :
        i, job_id, time_sub =  self.list_for_proc[ind]
        return self.get_batch_job_status(job_id, '')

    def kill_batch_job_for_cora_proc(self, ind) :
        i, job_id, time_sub =  self.list_for_proc[ind]
        return self.kill_batch_job(job_id, 'for proc')

    def status_batch_job_for_cora_proc_all(self) :
        ind = 0
        i, job_id, time_sub =  self.list_for_proc[ind]
        return self.get_batch_job_status_and_string(job_id, time_sub)

#-----------------------------

    def check_batch_job_for_cora_merge(self) :
        self.check_batch_job(self.job_id_cora_merge, 'merge')

    def kill_batch_job_for_cora_merge(self) :
        self.kill_batch_job(self.job_id_cora_merge, 'for merge')

    def status_for_cora_merge_files(self, comment='of merge: ' ) :
        fstatus, fstatus_str = self.status_and_string_for_files(fnm.get_list_of_files_cora_merge(), comment )
        if fstatus : cp.res_fname.setValue(fnm.path_cora_merge_result())
        return fstatus, fstatus_str

    def status_batch_job_for_cora_merge(self) :
        return self.get_batch_job_status_and_string(self.job_id_cora_merge, self.time_sub_merge)

#-----------------------------

#    def print_work_files_for_data_aver(self) :
#        self.print_files_for_list(fnm.get_list_of_files_cora_split(),'of correlation analysis:')

#-----------------------------

    def check_work_files_cora(self) :
        self.check_files_for_list(fnm.get_list_of_files_cora_split(),'of correlation analysis:')

#-----------------------------

    def remove_files_cora_split(self) :
        self.remove_files_for_list(fnm.get_list_of_files_cora_split_all(),'of split:')

#-----------------------------

    def remove_files_cora_proc(self, ind=None) :

        if ind is None :
            self.list_of_files_to_remove = fnm.get_list_of_files_cora_proc_work() + \
                                           fnm.get_list_of_files_cora_proc_work_log()
            self.list_of_files_to_remove.append(fnm.path_cora_proc_tau_out()) 

        else :
            self.list_of_files_to_remove = [fnm.get_list_of_files_cora_proc_work()[ind], \
                                           fnm.get_list_of_files_cora_proc_work_log()[ind]]

        #print 'self.list_of_files_to_remove =\n', self.list_of_files_to_remove
        self.remove_files_for_list(self.list_of_files_to_remove,'of proc:')

#-----------------------------

    def remove_files_cora_merge(self) :
        self.remove_files_for_list(fnm.get_list_of_files_cora_merge_main(),'of merge:')

#-----------------------------

    def get_batch_job_id_cora_split(self) :
        return self.job_id_cora_split

#-----------------------------

    def get_batch_job_cora_split_time_string(self) :
        return gu.get_local_time_str(self.time_sub_split, fmt='%Y-%m-%d %H:%M:%S')

#-----------------------------

    def get_batch_job_id_cora_proc(self, ind) :
        return self.list_for_proc[ind][1]

#-----------------------------

    def get_batch_job_cora_proc_time_string(self, ind) :
        #print 'ind:', ind
        time_sub_sec = self.list_for_proc[ind][2]
        if time_sub_sec is None : return 'Time N/A'
        return gu.get_local_time_str(time_sub_sec, fmt='%Y-%m-%d %H:%M:%S')

#-----------------------------

    def get_batch_job_id_cora_merge(self) :
        return self.job_id_cora_merge

#-----------------------------

    def get_batch_job_cora_merge_time_string(self) :
        return gu.get_local_time_str(self.time_sub_merge, fmt='%Y-%m-%d %H:%M:%S')

#-----------------------------

    def remove_files_cora_all(self):
        logger.debug('remove_files_cora_all', __name__)
        self.remove_files_cora_split()
        self.remove_files_cora_merge()
        for i in range(self.nparts) :
            self.remove_files_cora_proc(i)

#-----------------------------
#-----------------------------
#----- AUTO-PROCESSING -------
#-----------------------------
#-----------------------------

    def connectToThread1(self):
        try : self.connect( cp.thread1, QtCore.SIGNAL('update(QString)'), self.updateStatus )
        except : logger.warning('connectToThread1 IS FAILED !!!', __name__)


    def disconnectFromThread1(self):
        try : self.disconnect( cp.thread1, QtCore.SIGNAL('update(QString)'), self.updateStatus )
        except : logger.warning('disconnectFromThread1 IS FAILED !!!', __name__)


    def updateStatus(self, text):
        #print 'BatchJobCorAna: Signal is recieved ' + str(text)
        self.auto_processing_status()

#-----------------------------

    def stop_auto_processing(self) :
        cp.autoRunStatus = 0            
        self.kill_all_batch_jobs()
        logger.info('Auto-processing IS STOPPED', __name__)
        self.disconnectFromThread1()


    def kill_all_batch_jobs(self):
        logger.debug('kill_all_batch_jobs', __name__)
        self.kill_batch_job_for_cora_split()
        self.kill_batch_job_for_cora_merge()
        for i in range(self.nparts) :
            self.kill_batch_job_for_cora_proc(i)

#-----------------------------

    def start_auto_processing(self) :
        if cp.autoRunStatus != 0 :            
            logger.warning('Auto-processing procedure is already active in stage '+str(cp.autoRunStatus), __name__)
        else :
            self.connectToThread1()
            self.remove_files_cora_all()
            self.onRunSplit()

#-----------------------------

    def auto_processing_status(self):
        if cp.autoRunStatus : self.updateRunState()

#-----------------------------

    def updateRunState(self):
        logger.info('Auto run stage '+str(cp.autoRunStatus), __name__)

        self.status_split, fstatus_str_split = bjcora.status_for_cora_split_files(comment='')
        self.status_proc,  fstatus_str_proc  = bjcora.status_for_cora_proc_files (comment='')
        self.status_merge, fstatus_str_merge = bjcora.status_for_cora_merge_files(comment='')

        if   cp.autoRunStatus == 1 and self.status_split :            
            logger.info('updateRunState: Split is completed, begin processing', __name__)
            self.onRunProc()

        elif cp.autoRunStatus == 2 and self.status_proc : 
            logger.info('updateRunState: Processing is completed, begin merging', __name__)
            self.onRunMerge()

        elif cp.autoRunStatus == 3 and self.status_merge : 
            logger.info('updateRunState: Merging is completed, stop auto-run', __name__)
            cp.autoRunStatus = 0            
            self.disconnectFromThread1()
        
#-----------------------------

    def onRunSplit(self):
        logger.debug('onRunSplit', __name__)
        if self.isReadyToStartRunSplit() :
            self.submit_batch_for_cora_split()
            cp.autoRunStatus = 1


    def isReadyToStartRunSplit(self):
        msg1 = 'JOB IS NOT SUBMITTED !!!\nFirst, set the number of events for data.'
        if  (cp.bat_data_end.value() == cp.bat_data_end.value_def()) :
            logger.warning(msg1, __name__)
            return False

        elif(cp.bat_data_start.value() >= cp.bat_data_end.value()) :
            logger.warning(msg1, __name__)
            return False

        else :
            return True

#-----------------------------

    def onRunProc(self):
        logger.debug('onRunProc', __name__)

        for i in range(self.nparts) :
            if self.isReadyToStartRunProc(i) :
                self.submit_batch_for_cora_proc(i)
                cp.autoRunStatus = 2


    def isReadyToStartRunProc(self, ind):

        fname = fnm.get_list_of_files_cora_split_work()[ind]
        if not os.path.exists(fname) :
            msg1 = 'JOB IS NOT SUBMITTED !!!\nThe file ' + str(fname) + ' does not exist'
            logger.warning(msg1, __name__)
            return False

        fsize = os.path.getsize(fname)
        if fsize < 1 :
            msg2 = 'JOB IS NOT SUBMITTED !!!\nThe file ' + str(fname) + ' has wrong size(Byte): ' + str(fsize) 
            logger.warning(msg2, __name__)
            return False

        msg3 = 'The file ' + str(fname) + ' exists and its size(Byte): ' + str(fsize) 
        logger.info(msg3, __name__)
        return True

#-----------------------------

    def onRunMerge(self):
        logger.debug('onRunMerge', __name__)
        if self.isReadyToStartRunMerge() :
            self.submit_batch_for_cora_merge()
            cp.autoRunStatus = 3


    def isReadyToStartRunMerge(self):
        fstatus, fstatus_str = bjcora.status_for_cora_proc_files()
        if fstatus : 
            logger.info(fstatus_str, __name__)
            return True
        else :
            msg = 'JOB IS NOT SUBMITTED !!!' + fstatus_str
            logger.warning(msg, __name__)
            return False

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
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
