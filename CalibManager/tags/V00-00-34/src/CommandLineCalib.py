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

@version $Id:$

@author Mikhail S. Dubrovin
"""
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

#------------------------------
class CommandLineCalib () :
    """Command line calibration of dark runs

    @see FileNameManager, ConfigFileGenerator, ConfigParametersForApp, BatchJobPedestals, BatchLogScanParser, FileDeployer
    """
    def __init__ (self, args, opts) :

        #print '__name__', __name__ # CalibManager.CommandLineCalib

        self.args = args
        self.opts = opts

        #self.print_command_line_pars(args, opts)

        #print cp.getTextParameters()

        if not self.set_pars() : return

        self.print_local_pars()

        self.print_list_of_xtc_files()
        self.print_list_of_sources_from_regdb()

        if self.queue is None :
            self.proc_dark_run_interactively()
        else :
            if not self.get_print_lsf_status() : return
            self.proc_dark_run_in_batch()
            self.print_list_of_types_and_sources_from_xtc()

        self.print_list_of_files_dark_in_work_dir()

        self.deploy_calib_files()
        
        logger.saveLogInFile(fnm.log_file())

#------------------------------

    def set_pars(self) :

	if self.opts['runnum'] is None :
            appname = os.path.basename(sys.argv[0])
	    msg = 'This command line calibration interface should be launched with parameters.'\
                  +'\nTo see the list of parameters use command: %s -h' % appname\
                  +'\nIf the "%s" is launched after "calibman" most of parameters may be already set.' % appname\
	          +'\nBut, at least run number must be specified as an optional parameter, try command:\n    %s -r <number>'\
                  % appname
            print msg
	    return False
        self.runnum = self.opts['runnum']
        self.str_run_number = '%04d' % self.runnum


	if self.opts['runrange'] is None :
            self.str_run_range = '%s-end' % self.runnum
        else :
            self.str_run_range = self.opts['runrange'] 


	self.exp_name = cp.exp_name.value() if self.opts['exp'] is None else self.opts['exp']
        if self.exp_name == cp.exp_name.value_def() :
	    print 'Experiment name is not defined, should be specified as optional parameter -e <exp-name>'
	    return False


        if self.opts['detector'] is None :
            self.det_name = cp.det_name.value()
        else :
            self.det_name = self.opts['detector'].replace(","," ")

        list_of_dets_sel = self.det_name.split()
        list_of_dets_sel_lower = [det.lower() for det in list_of_dets_sel]

        for det, par in zip(cp.list_of_dets_lower, cp.det_cbx_states_list) :
            par.setValue(det in list_of_dets_sel_lower)
            print '%s %s' % (det.ljust(10), par.value())



        if self.det_name == cp.det_name.value_def() :
	    print 'Detector name(s) is not defined, should be specified as optional parameter -d <det-names>, ex.: -d CSPAD,CSPAD2x2 etc'
	    return False


        self.skip_events = cp.bat_dark_start.value() if self.opts['skip_events'] is None else self.opts['skip_events']
        self.num_events  = cp.bat_dark_end.value() - cp.bat_dark_start.value() if self.opts['num_events'] is None else self.opts['num_events']
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

        self.calibdir     = cp.calib_dir.value() if self.opts['calibdir'] is None else self.opts['calibdir']
        if self.calibdir == cp.calib_dir.value_def() :
            self.calibdir = fnm.path_to_calib_dir_default()

        cp.calib_dir     .setValue(self.calibdir)
        cp.dir_work      .setValue(self.workdir)
        cp.bat_queue     .setValue(self.queue)
        cp.bat_dark_start.setValue(self.skip_events)
        cp.bat_dark_end  .setValue(self.num_events+self.skip_events)
        cp.mask_rms_thr  .setValue(self.thr_rms)
	cp.det_name      .setValue(self.det_name)

        #cp.log_file      .setValue(self.logfile)          

        gu.create_directory(cp.dir_work.value())

        return True

#------------------------------

    def print_local_pars(self) :
        print '\n' + 50*'-' \
        + '\nprint_local_pars(): Combination of command line parameters and' \
        + '\nconfiguration parameters from file %s (if available after "calibman")' % cp.getParsFileName() \
        + '\n     str_run_number: %s' % self.str_run_number \
        + '\n     runrange      : %s' % self.str_run_range \
        + '\n     exp_name      : %s' % self.exp_name \
        + '\n     instr_name    : %s' % self.instr_name \
        + '\n     workdir       : %s' % self.workdir \
        + '\n     calibdir      : %s' % self.calibdir \
        + '\n     det_name      : %s' % self.det_name \
        + '\n     queue         : %s' % self.queue \
        + '\n     num_events    : %d' % self.num_events \
        + '\n     skip_events   : %d' % self.skip_events \
        + '\n     thr_rms       : %f' % self.thr_rms \
        + '\n     process       : %s' % self.process \
        + '\n     deploy        : %s' % self.deploy
        #+ '\nself.logfile       : ' % self.logfile     

#------------------------------

    def print_command_line_pars(self, args, opts) :

        print '\nprint_command_line_pars(...):'

        print 'args:', args
        print 'opts:', opts
        #print 'args:\n', ', '.join(args)

        #for k,v in opts.iteritems():
       	#    print '%s : %s' % (k.ljust(16),v)

#------------------------------

    def proc_dark_run_interactively(self) :

        if self.process :
            print '\n' + 50*'-' + '\nBegin dark run data processing interactively'
        else :
            print '\n' + 50*'-' + '\nWARNING: File processing option IS TURNED OFF...'\
                  + '\nAdd "-P" option in the command line to process files' 
            return

        self.bjpeds = BatchJobPedestals(self.runnum)
        self.bjpeds.command_for_peds_scan()

        self.print_list_of_types_and_sources_from_xtc()

        if not self.bjpeds.command_for_peds_aver() :
            msg = '\n' + 50*'-' + '\nSTATUS OF PROCESSING IS NOT GOOD !!!'\
                  +'\nSee details in the logfile(s)'
            print msg
            #return

        self.print_dark_ave_batch_log()
        return

#------------------------------

    def proc_dark_run_in_batch(self) :

        if self.process :
            print '\n' + 50*'-' + '\nBegin dark run data processing in batch queue %s' % self.queue
        else :
            print '\n' + 50*'-' + '\nWARNING: File processing option IS TURNED OFF...'\
                  + '\nAdd "-P" option in the command line to process files' 
            return

        self.bjpeds = BatchJobPedestals(self.runnum)
        self.bjpeds.start_auto_processing()

        sum_dt=0
        dt = 10 # sec
        for i in range(50) :
            sleep(dt)
            sum_dt += dt
            status = self.bjpeds.status_for_peds_files_essential()
            print '%3d sec: Files %s available' % (sum_dt, {False:'ARE NOT', True:'ARE'}[status])
            if status :
                self.print_dark_ave_batch_log()
                return

        print 'WARNING: Too many check cycles. Probably LSF is dead...'
        
        #if self.bjpeds.autoRunStage :            
        #self.bjpeds.stop_auto_processing()


#------------------------------

    def deploy_calib_files(self) :
        list_of_deploy_commands, list_of_sources = fdmets.get_list_of_deploy_commands_and_sources_dark(self.str_run_number, self.str_run_range)
        msg = '\n' + 50*'-' + '\nTentative deployment commands:\n' + '\n'.join(list_of_deploy_commands)
        print msg

        if self.deploy :
            print '\n' + 50*'-' + '\nBegin deployment of calibration files' 
            fdmets.deploy_calib_files(self.str_run_number, self.str_run_range, mode='calibrun-dark', ask_confirm=False)
            print '\nDeployment of calibration files is completed'
        else :
            print '\n' + 50*'-' + '\nWARNING: File deployment option IS TURNED OFF... \nAdd "-D" option in the command line to deploy files' 

#------------------------------

#    def submit_job_in_batch(self) :
#
#        if not cfg.make_psana_cfg_file_for_peds_aver() :
#            print 'Can not make_psana_cfg_file_for_peds_aver(...)' 
#            return
#
#        command      = 'psana -c ' + fnm.path_peds_aver_psana_cfg() + ' ' + fnm.path_to_xtc_files_for_run()
#        queue        = cp.bat_queue.value()
#        bat_log_file = fnm.path_peds_aver_batch_log()
#
#        print 'command', command
#        print 'queue', queue
#        print 'bat_log_file', bat_log_file

#------------------------------

    def print_list_of_files_dark_in_work_dir(self) :
        lst = self.get_list_of_files_dark_in_work_dir()
        msg = '\n' + 50*'-' + '\nList of files in work directory for command "ls %s*"' % fnm.path_prefix_dark()
        if lst == [] : msg += ' is empty'
        else         : msg += ':\n' + '\n'.join(lst)
        print msg


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
        txt = '\n' + 50*'-' + '\nData Types and Sources from xtc scan of the\n' \
            + cp.blsp.txt_list_of_types_and_sources()
        print txt


    def print_list_of_sources_from_regdb(self) :
        txt = '\n' + 50*'-' + '\nSources from DB:\n' \
            + cp.blsp.txt_of_sources_in_run()
        print txt


    def print_dark_ave_batch_log(self) :
        path = fnm.path_peds_aver_batch_log()
        txt = '\n' + 50*'-' + '\npsana log file %s:\n\n' % path \
            + gu.load_textfile(fnm.path_peds_aver_batch_log()) \
            + 'End of psana log file %s' % path
        print txt


    def get_print_lsf_status(self) :
        queue = cp.bat_queue.value()
        msg, status = gu.msg_and_status_of_lsf(queue)
        msgi = '\n' + 50*'-' + '\nLSF status for queue %s: \n%s\nLSF status for %s is %s'\
               % (queue, msg, queue, {False:'bad',True:'good'}[status])
        print msgi
        return status


    def print_list_of_xtc_files(self) :
        pattern = '-r%s' % self.str_run_number
        lst = fnm.get_list_of_xtc_files()
        lst_for_run = [path for path in lst if pattern in os.path.basename(path)]
        txt = '\n' + 50*'-' + '\nList of xtc files for exp=%s:run=%s :\n' % (self.exp_name, self.str_run_number)
        txt += '\n'.join(lst_for_run)
        print txt

#------------------------------
#------------------------------
#------------------------------
