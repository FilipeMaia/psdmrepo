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

from BatchJob import *

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

        self.job_id_cora_split, out, err = gu.batch_job_submit(command, queue, log_file)

#-----------------------------

    def submit_batch_for_cora_proc(self, ind) :

        self.init_list_for_proc()

        i, job_id, time_sub = self.list_for_proc[ind] 

        if not self.job_can_be_submitted(job_id, time_sub, 'cor. ana. proc') : return
        time_sub = gu.get_time_sec()

        fname    = fnm.get_list_of_files_cora_split_work()[ind]
        tname    = fnm.path_cora_proc_tau_in()
        log_file = fnm.get_list_of_files_cora_proc_work_log()[ind]

        command  = 'corana -f ' + fname # + ' -l ' + log_file
        if os.path.exists(tname) : command +=   '-t ' + tname
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

    def status_for_cora_split_files(self) :
        return self.status_and_string_for_files(fnm.get_list_of_files_cora_split_all(), 'of split: ' )

    def status_batch_job_for_cora_split(self) :
        return self.get_batch_job_status_and_string(self.job_id_cora_split, self.time_sub_split)

#-----------------------------

    def status_for_cora_proc_files(self) :
        return self.status_and_string_for_files(fnm.get_list_of_files_cora_proc_check(), 'of proc: ' )

    def status_batch_job_for_cora_proc(self, ind) :
        i, job_id, time_sub =  self.list_for_proc[ind]
        return self.get_batch_job_status(job_id, '')

#-----------------------------

    def check_batch_job_for_cora_merge(self) :
        self.check_batch_job(self.job_id_cora_merge, 'merge')

    def status_for_cora_merge_files(self) :
        return self.status_and_string_for_files(fnm.get_list_of_files_cora_merge(), 'of merge: ' )

    def status_batch_job_for_cora_merge(self) :
        return self.get_batch_job_status_and_string(self.job_id_cora_merge, self.time_sub_merge)

#-----------------------------

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

        if ind == None :
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
        if time_sub_sec == None : return 'Time N/A'
        return gu.get_local_time_str(time_sub_sec, fmt='%Y-%m-%d %H:%M:%S')

#-----------------------------

    def get_batch_job_id_cora_merge(self) :
        return self.job_id_cora_merge

#-----------------------------

    def get_batch_job_cora_merge_time_string(self) :
        return gu.get_local_time_str(self.time_sub_merge, fmt='%Y-%m-%d %H:%M:%S')

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
