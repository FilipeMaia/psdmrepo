#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CommandLineCalib...
#
#------------------------------------------------------------------------

"""CommandLineCalib is intended for command line calibration of dark runs

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

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
import sys
import os
from time import sleep

from Logger                   import logger

from FileNameManager          import fnm
from ConfigFileGenerator      import cfg
from ConfigParametersForApp   import cp

from BatchJobPedestals        import *
from BatchLogScanParser       import blsp # Just in order to instatiate it

import FileDeployer           as     fdmets
from NotificationDBForCL      import *

#------------------------------
class CommandLineCalib () :
    """Command line calibration of dark runs

    @see FileNameManager, ConfigFileGenerator, ConfigParametersForApp, BatchJobPedestals, BatchLogScanParser, FileDeployer, Logger
    """

    sep = '\n' + 60*'-' + '\n'

    def __init__ (self, args, opts) :

        #print '__name__', __name__ # CalibManager.CommandLineCalib
        cp.commandlinecalib = self 

        self.args = args
        self.opts = opts
        self.count_msg = 0

        if not self.set_pars() : return

        self.print_command_line()
        self.print_local_pars()
        self.print_list_of_detectors()
        self.print_list_of_xtc_files()
        try :
            self.print_list_of_sources_from_regdb()
        except :
            pass

        gu.create_directory(cp.dir_work.value())

        if self.queue is None :
            self.proc_dark_run_interactively()
        else :
            if not self.get_print_lsf_status() : return
            self.proc_dark_run_in_batch()
            self.print_list_of_types_and_sources_from_xtc()

        self.print_list_of_files_dark_in_work_dir()

        self.deploy_calib_files()
        
        self.save_log_file()
        self.add_record_in_db()

#------------------------------

    def set_pars(self) :

        self.print_bits = self.opts['print_bits']
        logger.setPrintBits(self.print_bits)

	docfg = self.loadcfg = self.opts['loadcfg']

	if self.opts['runnum'] is None :
            appname = os.path.basename(sys.argv[0])
	    msg = self.sep + 'This command line calibration interface should be launched with parameters.'\
                  +'\nTo see the list of parameters use command: %s -h' % appname\
                  +'\nIf the "%s" is launched after "calibman" most of parameters may be already set.' % appname\
	          +'\nBut, at least run number must be specified as an optional parameter, try command:\n    %s -r <number> -L'%(appname)\
                  + self.sep
            self.log(msg,4)
	    return False
        self.runnum = self.opts['runnum']
        self.str_run_number = '%04d' % self.runnum

	if self.opts['runrange'] is None :
            self.str_run_range = '%s-end' % self.runnum
        else :
            self.str_run_range = self.opts['runrange'] 

        self.exp_name = cp.exp_name.value_def()
	self.exp_name = cp.exp_name.value() if docfg and self.opts['exp'] is None else self.opts['exp']
        if self.exp_name is None or self.exp_name == cp.exp_name.value_def() :
	    self.log('\nWARNING: EXPERIMENT NAME IS NOT DEFINED...'\
                     + '\nAdd optional parameter -e <exp-name>',4)
	    return False

        if self.opts['detector'] is None :
            self.det_name = cp.det_name.value() if docfg else cp.det_name.value_def()
        else :
            self.det_name = self.opts['detector'].replace(","," ")

        list_of_dets_sel = self.det_name.split()
        list_of_dets_sel_lower = [det.lower() for det in list_of_dets_sel]

        #msg = self.sep + 'List of detectors:'
        for det, par in zip(cp.list_of_dets_lower, cp.det_cbx_states_list) :
            par.setValue(det in list_of_dets_sel_lower)
            #msg += '\n%s %s' % (det.ljust(10), par.value())
        #self.log(msg,1)

        if self.det_name == cp.det_name.value_def() :
	    self.log('\nWARNING: DETECTOR NAMES ARE NOT DEFINED...'\
                     + '\nAdd optional parameter -d <det-names>, ex.: -d CSPAD,CSPAD2x2 etc',4)
	    return False

        self.event_code  = cp.bat_dark_sele.value()  if self.opts['event_code']  is None else self.opts['event_code']
        self.scan_events = cp.bat_dark_scan.value()  if self.opts['scan_events'] is None else self.opts['scan_events']
        self.skip_events = cp.bat_dark_start.value() if self.opts['skip_events'] is None else self.opts['skip_events']
        self.num_events  = cp.bat_dark_end.value() - cp.bat_dark_start.value() if self.opts['num_events'] is None else self.opts['num_events']
        self.thr_rms_min = cp.mask_rms_thr_min.value() if self.opts['thr_rms_min'] is None else self.opts['thr_rms_min']
        self.thr_rms     = cp.mask_rms_thr.value() if self.opts['thr_rms'] is None else self.opts['thr_rms']
        self.workdir     = cp.dir_work.value()  if self.opts['workdir'] is None else self.opts['workdir']
	#self.queue       = cp.bat_queue.value() if self.opts['queue'] is None else self.opts['queue']
	self.queue       = self.opts['queue']
	#self.logfile     = cp.log_file.value()  if self.opts['logfile']  is None else self.opts['logfile']

	self.process     = self.opts['process'] 
	self.deploy      = self.opts['deploy'] 
        self.instr_name  = self.exp_name[:3]

        cp.str_run_number.setValue(self.str_run_number)
        cp.exp_name      .setValue(self.exp_name)
        cp.instr_name    .setValue(self.instr_name)

        self.calibdir     = cp.calib_dir.value() if docfg and self.opts['calibdir'] is None else self.opts['calibdir']
        if self.calibdir == cp.calib_dir.value_def() or self.calibdir is None :
            self.calibdir = fnm.path_to_calib_dir_default()

        self.xtcdir       = cp.xtc_dir_non_std.value_def() if self.opts['xtcdir'] is None else self.opts['xtcdir']

        cp.xtc_dir_non_std .setValue(self.xtcdir)
        cp.calib_dir       .setValue(self.calibdir)
        cp.dir_work        .setValue(self.workdir)
        cp.bat_queue       .setValue(self.queue)
        cp.bat_dark_sele   .setValue(self.event_code)
        cp.bat_dark_scan   .setValue(self.scan_events)
        cp.bat_dark_start  .setValue(self.skip_events)
        cp.bat_dark_end    .setValue(self.num_events+self.skip_events)
        cp.mask_rms_thr_min.setValue(self.thr_rms_min)
        cp.mask_rms_thr    .setValue(self.thr_rms)
	cp.det_name        .setValue(self.det_name)

        #cp.log_file      .setValue(self.logfile)          

        return True

#------------------------------

    def print_local_pars(self) :
        msg = self.sep \
        + 'print_local_pars(): Combination of command line parameters and' \
        + '\nconfiguration parameters from file %s (if available after "calibman")' % cp.getParsFileName() \
        + '\n     str_run_number: %s' % self.str_run_number\
        + '\n     runrange      : %s' % self.str_run_range\
        + '\n     exp_name      : %s' % self.exp_name\
        + '\n     instr_name    : %s' % self.instr_name\
        + '\n     workdir       : %s' % self.workdir\
        + '\n     calibdir      : %s' % self.calibdir\
        + '\n     xtcdir        : %s' % self.xtcdir\
        + '\n     det_name      : %s' % self.det_name\
        + '\n     queue         : %s' % self.queue\
        + '\n     num_events    : %d' % self.num_events\
        + '\n     skip_events   : %d' % self.skip_events\
        + '\n     scan_events   : %d' % self.scan_events\
        + '\n     thr_rms_min   : %f' % self.thr_rms_min\
        + '\n     thr_rms       : %f' % self.thr_rms\
        + '\n     process       : %s' % self.process\
        + '\n     deploy        : %s' % self.deploy\
        + '\n     loadcfg       : %s' % self.loadcfg\
        + '\n     print_bits    : %s' % self.print_bits
        #+ '\nself.logfile       : ' % self.logfile     

        self.log(msg,1)

#------------------------------

    def print_list_of_detectors(self) :
        msg = self.sep + 'List of detectors:'
        for det, par in zip(cp.list_of_dets_lower, cp.det_cbx_states_list) :
            msg += '\n%s %s' % (det.ljust(10), par.value())
        self.log(msg,1)

#------------------------------

    def print_command_line(self) :
        msg = 'Command line for book-keeping:\n%s' % (' '.join(sys.argv))
        self.log(msg,1)

#------------------------------

    def print_command_line_pars(self, args, opts) :

        msg = '\nprint_command_line_pars(...):\n  args: %s\n  opts: %s' % (args,opts)
        self.log(msg,1)

#------------------------------

    def proc_dark_run_interactively(self) :

        if self.process :
            self.log(self.sep + 'Begin dark run data processing interactively',1)
        else :
            self.log(self.sep + '\nWARNING: FILE PROCESSING OPTION IS TURNED OFF...'\
                  + '\nAdd "-P" option in the command line to process files\n',4) 
            return

        self.bjpeds = BatchJobPedestals(self.runnum)
        self.bjpeds.command_for_peds_scan()

        self.print_list_of_types_and_sources_from_xtc()

        if not self.bjpeds.command_for_peds_aver() :
            msg = self.sep + 'Subprocess for averaging is completed with warning/error message(s);'\
                  +'\nsee details in the logfile(s).'
            self.log(msg,4)
            #return

        self.print_dark_ave_batch_log()
        return

#------------------------------

    def proc_dark_run_in_batch(self) :

        if self.process :
            self.log(self.sep + 'Begin dark run data processing in batch queue %s' % self.queue,1)
        else :
            self.log(self.sep + '\nWARNING: FILE PROCESSING OPTION IS TURNED OFF...'\
                  + '\nAdd "-P" option in the command line to process files\n',4)
            return

        self.bjpeds = BatchJobPedestals(self.runnum)
        self.bjpeds.start_auto_processing()

        sum_dt=0
        dt = 10 # sec
        for i in range(50) :
            sleep(dt)
            sum_dt += dt
            status = self.bjpeds.status_for_peds_files_essential()
            str_bj_stat, msg_bj_stat = self.bjpeds.status_batch_job_for_peds_aver() 

            self.log('%3d sec: Files %s available. %s' % (sum_dt, {False:'ARE NOT', True:'ARE'}[status], msg_bj_stat), 1)

            if status :
                self.print_dark_ave_batch_log()
                return

        print 'WARNING: Too many check cycles. Probably LSF is dead...'
        
        #if self.bjpeds.autoRunStage :            
        #self.bjpeds.stop_auto_processing()


#------------------------------

    def deploy_calib_files(self) :
        #list_of_deploy_commands, list_of_sources = fdmets.get_list_of_deploy_commands_and_sources_dark(self.str_run_number, self.str_run_range)
        #msg = self.sep + 'Tentative deployment commands:\n' + '\n'.join(list_of_deploy_commands)
        #self.log(msg,1)

        if self.deploy :
            self.log(self.sep + 'Begin deployment of calibration files',1) 
            fdmets.deploy_calib_files(self.str_run_number, self.str_run_range, mode='calibrun-dark', ask_confirm=False)
            self.log('\nDeployment of calibration files is completed',1)
        else :
            self.log(self.sep + '\nWARNING: FILE DEPLOYMENT OPTION IS TURNED OFF...'\
                     +'\nAdd "-D" option in the command line to deploy files\n',4)

#------------------------------
#------------------------------

    def save_log_file(self) :
        logfname = fnm.log_file()
        msg = 'See details in log-file: %s' % logfname
        #self.log(msg,4) # set it 4-critical - always print
        logger.critical(msg) # critical - always print
        logger.saveLogInFile(logfname)


    def add_record_in_db(self) :
        try :
            ndb = NotificationDBForCL()
            ndb.insert_record(mode='enabled')
            ndb.close()
            #ndb.add_record()
        except :
            pass

 
    def print_list_of_files_dark_in_work_dir(self) :
        lst = self.get_list_of_files_dark_in_work_dir()
        msg = self.sep + 'List of files in work directory for command "ls %s*"' % fnm.path_prefix_dark()
        if lst == [] : msg += ' is empty'
        else         : msg += ':\n' + '\n'.join(lst)
        self.log(msg,1)


    def get_list_of_files_dark_in_work_dir(self) :
        path_prexix = fnm.path_prefix_dark()
        dir, prefix = os.path.split(path_prexix)
        return gu.get_list_of_files_in_dir_for_part_fname(dir, pattern=prefix)


    def get_list_of_files_dark_expected(self) :
        lst_of_srcs = cp.blsp.list_of_sources_for_selected_detectors()
        return fnm.get_list_of_files_peds() \
             + gu.get_list_of_files_for_list_of_insets(fnm.path_peds_ave(),    lst_of_srcs) \
             + gu.get_list_of_files_for_list_of_insets(fnm.path_peds_rms(),    lst_of_srcs) \
             + gu.get_list_of_files_for_list_of_insets(fnm.path_hotpix_mask(), lst_of_srcs)


    def print_list_of_types_and_sources_from_xtc(self) :
        txt = self.sep + 'Data Types and Sources from xtc scan of the\n' \
            + cp.blsp.txt_list_of_types_and_sources()
        self.log(txt,1)


    def print_list_of_sources_from_regdb(self) :
        txt = self.sep + 'Sources from DB:' \
            + cp.blsp.txt_of_sources_in_run()
        self.log(txt,1)


    def print_dark_ave_batch_log(self) :
        path = fnm.path_peds_aver_batch_log()
        if not os.path.exists(path) :
            msg = 'File: %s does not exist' % path
            self.log(msg,2)            
            return
        
        txt = self.sep + 'psana log file %s:\n\n' % path \
            + gu.load_textfile(path) \
            + 'End of psana log file %s' % path
        self.log(txt,1)


    def get_print_lsf_status(self) :
        queue = cp.bat_queue.value()
        farm = cp.dict_of_queue_farm[queue]
        msg, status = gu.msg_and_status_of_lsf(farm, print_bits=0)
        msgi = self.sep + 'LSF status for queue %s on farm %s: \n%s\nLSF status for %s is %s'\
               % (queue, farm, msg, queue, {False:'bad',True:'good'}[status])
        self.log(msgi,1)

        msg, status = gu.msg_and_status_of_queue(queue)
        self.log('\nBatch queue status, %s'%msg, 1)

        return status


    def print_list_of_xtc_files(self) :
        pattern = '-r%s' % self.str_run_number
        lst = fnm.get_list_of_xtc_files()
        lst_for_run = [path for path in lst if pattern in os.path.basename(path)]
        txt = self.sep + 'List of xtc files for exp=%s:run=%s :\n' % (self.exp_name, self.str_run_number)
        txt += '\n'.join(lst_for_run)
        self.log(txt,1)

#------------------------------

    def log(self, msg, level=1) :
        """Internal logger - re-direct all messages to the project logger, critical messages"""
        #logger.levels = ['debug','info','warning','error','critical']
        self.count_msg += 1
        #print 'Received msg: %d' % self.count_msg
        #if self.print_bits & 1 or level==4 : print msg

        if   level==1 : logger.info    (msg, __name__)
        elif level==4 : logger.critical(msg, __name__)
        elif level==0 : logger.debug   (msg, __name__)
        elif level==2 : logger.warning (msg, __name__)
        elif level==3 : logger.error   (msg, __name__)
        else          : logger.info    (msg, __name__)

#------------------------------
#------------------------------
#------------------------------
