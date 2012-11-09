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
        self.job_id_str = None
        self.path_cfg   = fnm.path_pedestals_psana_cfg()

#-----------------------------

    def     make_psana_cfg_file_for_pedestals(self) :
        cfg.make_psana_cfg_file_for_pedestals()

    def     make_psana_cfg_file_for_tahometer(self) :
        cfg.make_psana_cfg_file_for_tahometer()

#-----------------------------

    def submit_batch_for_tahometer_on_dark_xtc(self) :

        self.make_psana_cfg_file_for_tahometer()

        command      = 'psana -c ' + fnm.path_tahometer_psana_cfg() + ' ' + fnm.path_dark_xtc()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_pedestals_tahometer_batch_log()

        self.job_id_str = gu.batch_job_submit(command, queue, bat_log_file)
        logger.info('Submit batch for tahometer on dark - job Id: ' + self.job_id_str, __name__) 

#-----------------------------

    def submit_batch_for_pedestals(self) :

        self.make_psana_cfg_file_for_pedestals()

        command      = 'psana -c ' + fnm.path_pedestals_psana_cfg() + ' ' + fnm.path_dark_xtc()
        queue        = cp.bat_queue.value()
        bat_log_file = fnm.path_pedestals_batch_log()

        self.job_id_str = gu.batch_job_submit(command, queue, bat_log_file)
        logger.info('submit_batch_for_pedestals() - job Id: ' + self.job_id_str, __name__) 

#-----------------------------

    def check_batch_status_for_pedestals(self) :

        if self.job_id_str == None :
            logger.info('Batch job was not submitted in this session.', __name__) 
            return

        #status, nodename = gu.batch_job_status_and_nodename(self.job_id_str, cp.bat_queue.value())
        #msg = 'Batch job Id: ' + self.job_id_str + ' on node: ' + str(nodename) + ' status: ' + str(status)
        lines = gu.batch_job_check(self.job_id_str, cp.bat_queue.value())
        msg = 'Check batch status for pedestals:\n'
        for line in lines :
            msg += line
        logger.info(msg, __name__) 

#-----------------------------

    def print_work_files_for_pedestals(self) :
        logger.info('Print work files for dark run / pedestals:', __name__)         
        for fname in fnm.get_list_of_files_pedestals() :
            logger.info(fname, __name__)         

#-----------------------------

    def check_work_files_for_pedestals(self) :
        logger.info('Check work files for dark run / pedestals:', __name__)         
        for fname in fnm.get_list_of_files_pedestals() :
            msg = '%s %s' % ( fname.ljust(100), str(os.path.lexists(fname)) )
            logger.info(msg, __name__)         

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
