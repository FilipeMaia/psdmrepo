#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module BatchJobPedestals...
#
#------------------------------------------------------------------------

"""Deals with batch jobs for pedestals

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

from ConfigParametersCorAna   import confpars as cp
from Logger                   import logger
from ConfigFileGenerator      import cfg
from FileNameManager          import fnm
import GlobalUtils            as     gu

#-----------------------------

class BatchJobPedestals :
    """Deals with batch jobs for pedestals.
    """

    def __init__ (self) :
        """Constructor.
        @param fname  the file name for output log file
        """
        self.job_id_ped_str = None
        self.job_id_tah_str = None
        #self.path_ped_cfg   = fnm.path_pedestals_psana_cfg()

        self.time_ped_job_submitted = None
        self.time_tah_job_submitted = None

        self.time_interval_sec      = 100
        self.dict_status = {True  : ' is available',
                            False : ' is not available'}

#-----------------------------

    def     make_psana_cfg_file_for_pedestals(self) :
        cfg.make_psana_cfg_file_for_pedestals()

    def     make_psana_cfg_file_for_tahometer(self) :
        cfg.make_psana_cfg_file_for_tahometer()

#-----------------------------

    def submit_batch_for_tahometer(self) :

        if self.job_was_recently_submitted(self.time_tah_job_submitted, 'scanner on dark') : return
        self.time_tah_job_submitted = gu.get_time_sec()

        self.make_psana_cfg_file_for_tahometer()

        command      = 'psana -c ' + fnm.path_pedestals_tahometer_psana_cfg() + ' ' + fnm.path_dark_xtc()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_pedestals_tahometer_batch_log()

        self.job_id_tah_str, out, err = gu.batch_job_submit(command, queue, bat_log_file)

        if err != '' : logger.warning( err, __name__) 
        logger.info(out, __name__) 
        logger.info('Submit batch for tahometer on dark, job Id: ' + self.job_id_tah_str) 

#-----------------------------

    def submit_batch_for_pedestals(self) :

        if self.job_was_recently_submitted(self.time_ped_job_submitted, 'pedestals') : return
        self.time_ped_job_submitted = gu.get_time_sec()

        self.make_psana_cfg_file_for_pedestals()

        command      = 'psana -c ' + fnm.path_pedestals_psana_cfg() + ' ' + fnm.path_dark_xtc()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_pedestals_batch_log()

        self.job_id_ped_str, out, err = gu.batch_job_submit(command, queue, bat_log_file)

        if err != '' : logger.warning(err, __name__) 
        logger.info(out, __name__) 
        logger.info('   Submit batch for pedestals on dark, job Id: ' + self.job_id_ped_str) 

#-----------------------------

    def job_was_recently_submitted(self, t_sub, comment='') :

        if t_sub == None : return False

        if gu.get_time_sec() - t_sub > self.time_interval_sec :
            return False
        else :
            msg = 'Sorry, but '+ comment +' job was already submitted less then ' + \
            str(self.time_interval_sec) + ' sec ago... Be patient, just relax and wait...'
            logger.warning(msg, __name__)         
            return True

#-----------------------------

    def check_batch_status_for_pedestals(self) :

        if self.job_id_ped_str == None :
            logger.info('Batch job for pedestals was not submitted in this session.', __name__) 
            return

        #status, nodename = gu.batch_job_status_and_nodename(self.job_id_ped_str, cp.bat_queue.value())
        #msg = 'Batch job Id: ' + self.job_id_ped_str + ' on node: ' + str(nodename) + ' status: ' + str(status)
        lines = gu.batch_job_check(self.job_id_ped_str, cp.bat_queue.value())
        msg = 'Check batch status for pedestals:\n'
        for line in lines :
            msg += line
        logger.info(msg, __name__) 

#-----------------------------

    def check_batch_status_for_pedestals_tahometer(self) :

        if self.job_id_tah_str == None :
            logger.info('Batch job for scanner on dark was not submitted in this session.', __name__) 
            return

        lines = gu.batch_job_check(self.job_id_tah_str, cp.bat_queue.value())
        msg = 'Check batch status for scanner on dark:\n'
        for line in lines :
            msg += line
        logger.info(msg, __name__) 

#-----------------------------

    def print_work_files_for_pedestals(self) :
        logger.info('Print work files for dark run / pedestals:', __name__)         
        for fname in fnm.get_list_of_files_pedestals() :
            logger.info(fname)         

#-----------------------------

    def check_work_files_for_pedestals(self) :
        logger.info('Check work files for dark run / pedestals:', __name__)         
        for fname in fnm.get_list_of_files_pedestals() :
            msg = '%s %s' % ( fname.ljust(100), self.dict_status[os.path.lexists(fname)] )
            logger.info(msg)         

#-----------------------------

    def remove_files_pedestals(self) :
        logger.info('Remove pedestal work files for selected run:', __name__)
        for fname in fnm.get_list_of_files_pedestals() :
            if os.path.lexists(fname) :
                gu.remove_file(fname)
                logger.info('Removed: ' + fname)

#-----------------------------

    def status_for_pedestals(self) :
        logger.info('Status for pedestals:', __name__)         
        fname  = fnm.path_pedestals_ave()
        status = os.path.lexists(fname)
        logger.info(fname + self.dict_status[status]) 
        return status

#-----------------------------

    def get_pedestals_from_file(self) :
        fname = fnm.path_pedestals_ave()
        if os.path.lexists(fname) :
            return gu.get_array_from_file( fname )
        else :
            logger.warning(fname + ' is not available', __name__)         
            return None

#-----------------------------

bjpeds = BatchJobPedestals ()

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    #bjpeds.submit_batch_for_pedestals()
    #gu.sleep_sec(5)
    #bjpeds.check_batch_status_for_pedestals()

    #bjpeds.submit_batch_for_tahometer_on_dark_xtc()
    #bjpeds.print_work_files_for_pedestals()
    bjpeds.check_work_files_for_pedestals()

    sys.exit ( 'End of test for BatchJobPedestals' )

#-----------------------------
