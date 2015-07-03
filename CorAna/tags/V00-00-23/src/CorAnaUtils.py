####!/usr/bin/env python
#--------------------
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CorAnaUtils...
#
#------------------------------------------------------------------------

""" CorAnaUtils.py contains global methods for file management etc.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule: CorAnaUtils.py CorAnaPars.py CorAna*.cpp

@version $Id: 2012-11-02 16:00:00Z dubrovin$

@author Mikhail Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

import numpy as np
import sys
import os

import commands
import subprocess # for subprocess.Popen
import time

#--------------------
#--------------------
#--------------------
#--------------------
#--------------------

def get_array_from_file(fname) :
    print 'get_array_from_file:', fname
    return np.loadtxt(fname, dtype=np.float32)

#--------------------

# This method parse the psana configuration file and returns the one of the xtc file names,
# from the line like:
# files = /reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc \

def get_xtc_fname_from_cfg_file(cfgname) :
    #print 'get_xtc_fname_from_cfg_file("' + cfgname + '")'
    
    if not os.path.exists(cfgname) :
        sys.exit('Configuration file' + cfgname + 'does not exist')

    file = open(cfgname, 'r')

    for line in file :
        if not 'files' in line : continue
        pos = line.find('files')
        if line[pos:pos+5] != 'files' : continue
        fname_xtc = line[line.find('/reg/d/psdm/'):line.find('.xtc')+4]
        print 'The 1st xtc file name: ' + fname_xtc
        return fname_xtc

    return None

#--------------------
# This method parse the psana configuration file and
# return the parameter value or None if not found or line is commented.

def get_parameter_from_cfg_file(fname, parname) :
    #print 'get_parameter_from_cfg_file("' + fname + '", "' + parname + '")'
    if not os.path.exists(fname) :
        sys.exit('Configuration file' + fname + 'does not exist')

    file = open(fname, 'r')

    for line in file :
        if not parname in line : continue
        par_str = get_parameter_from_cfg_file_line(line, parname)
        if par_str is not None : return par_str

    return None

#--------------------
# This method parse the line from psana configuration file
# and return the parameter value or None if not found or line is commented.

def get_parameter_from_cfg_file_line(line, parname) :
    #print 'get_parameter_from_cfg_file_line("' + line + '", "' + parname + '")'

    first_field = line.split(' ')[0]
    if first_field[0:len(parname)] != parname : return None

    pos_eq = line.find('=')
    if pos_eq == -1 : return None        # sign "=" is missing

    par_str = line[pos_eq+1:].strip(' ').rstrip('\n')
    if len(par_str) == 0 : return None   # empty value

    #print parname + ' is found with value: ' +  par_str
    return par_str

#--------------------

#def check_the_file(trailer) :
#    print 'check_the_file(trailer): for trailer: ' + trailer
#    print 'in the directory: ' + cp.dname_work
#    list_of_files = get_list_of_files_in_dir(cp.dname_work)
#    #print 'list_of_files =', list_of_files
#    
#    path = cp.fname_com + trailer
#    dname, fname = os.path.split(path)     # i.e. ('work_corana', 'img-xcs-r0015-b0000.bin')
#    print path,        
#
#    if fname in list_of_files :
#        print '- is found'
#    else :
#        print '- is NOT FOUND !!!'
#        sys.exit('Files with splitted image are not produced successfully... Job is terminated.')

#--------------------

#def check_list_of_files(trailer) :
#    print 'check_list_of_files(trailer): for trailer: ' + trailer
#    print 'in the directory: ' + cp.dname_work
#    list_of_files = get_list_of_files_in_dir(cp.dname_work)
#    #print 'list_of_files =', list_of_files
#    
#    for f in range (cp.nfiles_out) :
#        path = cp.fname_com + '-b%04d'%(f) + trailer
#        dname, fname = os.path.split(path)     # i.e. ('work_corana', 'img-xcs-r0015-b0000.bin')
#        print path,
#
#        if fname in list_of_files :
#            print '- is found'
#        else :
#            print '- is NOT FOUND !!!'
#            sys.exit('Files with splitted image are not produced successfully... Job is terminated.')

#--------------------

#def remove_file(path) :
#    print 'remove file: ' + path
#    p = subprocess.Popen(['rm', path], stdout=subprocess.PIPE)
#    p.wait() # short time waiting untill submission is done, 

#--------------------

#def remove_file_for_tail(list_of_tails) :
#    for tail in list_of_tails :
#        fname = cp.fname_com + tail
#        remove_file(fname)

#--------------------

#def remove_split_files() :
#    for f in range (cp.nfiles_out) :
#        path_com = cp.fname_com + '-b%04d'%(f)
#        remove_file(path_com + '.bin')
#        remove_file(path_com + '-result.bin')
#        remove_file(path_com + '-result-log.txt')
        
#--------------------

 
def print_subproc_attributes(proc):
    """ Use it after command like: proc = subprocess.Popen(bcmd, stdout=subprocess.PIPE)"""
    pid_str = str(proc.pid)
    print 'pid           :', proc.pid
    print 'stdin         :', proc.stdin     # shouuld be treated as open file
    print 'stderr        :', proc.stderr    # shouuld be treated as open file
    print 'stdout        :', proc.stdout    # shouuld be treated as open file
    print 'returncode    :', proc.returncode
    
#--------------------

#--------------------

#def one_batch_job_submit_and_wait (command_seq) :
#    print 'Sequence of parameters for the batch command:\n', command_seq
#    #=====
#    job_id, cp.status = batch_job_submit(command_seq), None
#    #=====
#    print 'Wait untill batch job is compleated...\n',
#    sleep_time = 5 # sleep time in sec
#    counter=0
#    while cp.status != 'DONE':
#        counter+=1
#        time.sleep(sleep_time) # sleep time in sec 
#        cp.status, cp.nodename = batch_job_status_and_nodename(job_id)
#        print 'Check batch status in', counter*sleep_time,'sec after submission:', job_id, cp.status, cp.nodename#
#
#        if cp.status == 'EXIT':
#            print 'Something is going wrong. Check the log file for this command sequence:\n', command_seq
#            sys.exit('EXIT: Job IS NOT completed !!! See the log-file for details.')
            
#--------------------

#def submit_jobs_for_cor_proc_interactive() :
#    cmd_base = cp.cmd_proc # 'corana'
#    print '-cmd_base:\n', cmd_base + ' -f <fname_data> [-t <fname_tau>] [-l <logfile>]'
#
#    for f in range (cp.nfiles_out) :
#        fname = cp.fname_com + '-b%04d'%(f) + '.bin'
#        cmd = cmd_base + ' -f ' + fname
#        if cp.fname_tau is not None : cmd += ' -t ' + cp.fname_tau
#        print cmd
#        print '  Wait untill processing of this file is compleated...\n',
#        status, log =0, 'DEFAULT LOGFILE FOR CORRELATION PROCESSING - THIS IS A TEST MODE !!!\nTHE getstatusoutput(cmd) IS COMMENTED OUT !!!'
#        #=====
#        status, log = commands.getstatusoutput(cmd)
#        #=====
#
#        if status != 0 : 
#            print 'Correlation processing job status:', status
#            sys.exit('Correlation processing job is completed with non-zero status... Job is terminated.')

#--------------------

#def submit_jobs_for_cor_proc() :
#    cmd_base = cp.cmd_proc # 'corana'
#
#    #bcmd = "bsub -q psfehq -o ~/LCLS/PSANA-V01/log.txt 'corana -f img-xcs-r0015-b0001.bin'"
#    #print 'command should be like that:\n', bcmd
#
#    print 'Command stub:', cmd_base + ' -f <fname_data> [-t <fname_tau>] [-l <logfile>]'
#
#    d_jobs = {} # Dict. structure {<int-index-of-the-file>:[<job-id>,<status>]}
#
#    for f in range (cp.nfiles_out) :
#        logfn = cp.fname_com + '-b%04d'%(f) + '-result-log.txt'
#        fname = cp.fname_com + '-b%04d'%(f) + '.bin'
#        cmd = cmd_base + ' -f ' + fname
#        if cp.fname_tau is not None : cmd += ' -t ' + cp.fname_tau
#
#        cmd = 'cd ' + cp.pwdir + '; ' + cmd
#        bcmd = ['bsub', '-q', cp.batch_queue, '-o', logfn, cmd]
#        
#        #print cmd
#        print 'Sequence for batch:\n',bcmd
#
#        job_id, status, node = batch_job_submit(bcmd), None, None
#        d_jobs[f] = [job_id, status, node]
#
#    print 'Wait untill all splitted files processing is compleated...\n',
#
#    sleep_time = 10 # sleep time in sec
#    cp.all_done = False
#    counter=0
#    while not cp.all_done :
#        counter+=1
#        time.sleep(sleep_time)
#        print 'Check batch status in', counter*sleep_time, 'sec after submission:'
#        cp.all_done = True
#        
#        for ind,job_pars in d_jobs.items():
#            job_id = job_pars[0]
#            if job_pars[1] != 'DONE' :
#                job_pars[1],job_pars[2] = batch_job_status_and_nodename(job_id)
#
#            print ind, job_pars
#            if job_pars[1] != 'DONE' :
#                cp.all_done = False
#
#            if job_pars[1] == 'EXIT' :
#                logfn = cp.fname_com + '-b%04d'%(ind) + '-result-log.txt'
#                print '\nSomething is going wrong. Check the log file: ' + logfn
#                sys.exit('EXIT: Job IS NOT completed !!!')
#
#    #sys.exit('TEST EXIT')

#--------------------

#def submit_job_for_splitter_interactive() :
#    cmd_base = cp.cmd_split # 'psana'
#    command = cmd_base + ' -c ' + cp.fname_cfg + ' ' + cp.fname_xtc
#    print 'run command:\n', command
#    print '  Wait untill splitting is compleated...\n',
#
#    status, log =0, 'DEFAULT LOGFILE FOR SPLITTER - THIS IS A TEST MODE !!!\nTHE getstatusoutput(command) IS COMMENTED OUT !!!'
#    #=====
#    status, log = commands.getstatusoutput(command)
#    #=====
#    print 'Log:\n', log
#
#    if status != 0 : 
#       print 'Splitter job status:', status
#       sys.exit('Job for splitter is completed with non-zero status... Job is terminated.')
#    
#--------------------

#def submit_job_for_splitter() :
#    cmd_base = cp.cmd_split # 'psana'
#    logfn = cp.fname_com + '-split-log.txt'
#    cmd  = 'cd ' + cp.pwdir + '; '
#    #cmd += 'sit_setup; '
#    cmd += '. /reg/g/psdm/etc/ana_env.sh; '
#    cmd += cmd_base + ' -c ' + cp.fname_cfg + ' ' + cp.fname_xtc
#
#    one_batch_job_submit_and_wait(['bsub', '-q', cp.batch_queue, '-o', logfn, cmd])
#    print 'Splitter job is completed.'

#--------------------

#def submit_job_for_merging_interactive() :
#    cmd_base = cp.cmd_merge # 'corana_merge'
#    print 'cmd_base:\n', cmd_base + ' -f <fname_data> [-t <fname_tau>]'
#    fname = cp.fname_com + '-b0000-result.bin'
#    cmd = cmd_base + ' -f ' + fname
#    if cp.fname_tau is not None : cmd += ' -t ' + cp.fname_tau
#    print cmd
#    print '  Wait untill merging is compleated...\n',
#    status, log =0, 'DEFAULT LOGFILE FOR MERGING - THIS IS A TEST MODE !!!\nTHE getstatusoutput(cmd) IS COMMENTED OUT !!!'
#    #=====
#    status, log = commands.getstatusoutput(cmd)
#    #=====
#    if status != 0 : 
#       print 'Merging job status: ', status
#       sys.exit('Job for merging is completed with non-zero status... Job is terminated.')
#
##--------------------

#def submit_job_for_merging() :
#    cmd_base = cp.cmd_merge # 'corana_merge'
#    #print 'cmd_base:\n', cmd_base + ' -f <fname_data> [-t <fname_tau>]'
#    logfn = cp.fname_com + '-image-result-log.txt'
#    fname = cp.fname_com + '-b0000-result.bin'
#    cmd  = 'cd ' + cp.pwdir + '; '
#    cmd += cmd_base + ' -f ' + fname
#    if cp.fname_tau is not None : cmd += ' -t ' + cp.fname_tau
#
#    one_batch_job_submit_and_wait(['bsub', '-q', cp.batch_queue, '-o', logfn, cmd])
#    print 'Merging is completed.'
#
##--------------------
#
#def submit_job_for_proc_results_interactive() :
#    cmd_base = cp.cmd_procres # 'corana_procres'
#    print 'cmd_base:\n', cmd_base + ' -f <fname_data> [-t <fname_tau>]'
#    fname = cp.fname_com + '-b0000-result.bin'
#    cmd = cmd_base + ' -f ' + fname
#    if cp.fname_tau is not None : cmd += ' -t ' + cp.fname_tau
#    print cmd
#    print '  Wait untill test processing of results is compleated...\n',
#    status, log =0, 'DEFAULT LOGFILE FOR TEST PROCESSING OF RESULTS - THIS IS A TEST MODE !!!\nTHE getstatusoutput(cmd) IS COMMENTED OUT !!!'
#    #=====
#    status, log = commands.getstatusoutput(cmd)
#    #=====
#    if status != 0 : 
#       print 'Test processing of results job status: ', status
#       sys.exit('Job test processing of results  is completed with non-zero status... Job is terminated.')
#
##--------------------
#
#def submit_job_for_proc_results() :
#    cmd_base = cp.cmd_procres # 'corana_procres'
#    #print 'cmd_base:\n', cmd_base + ' -f <fname_data> [-t <fname_tau>]'
#    logfn = cp.fname_com + '-proc-results-log.txt'
#    fname = cp.fname_com + '-b0000-result.bin'
#    cmd  = 'cd ' + cp.pwdir + '; '
#    cmd += cmd_base + ' -f ' + fname
#    if cp.fname_tau is not None : cmd += ' -t ' + cp.fname_tau
#
#    one_batch_job_submit_and_wait(['bsub', '-q', cp.batch_queue, '-o', logfn, cmd])
#    print 'Test processing of results is completed.'
#
#--------------------

def do_test() :

    pwd = get_pwd()

    print 'pwd =', pwd
    #print 'list_of_files =', get_list_of_files_in_dir(pwd)
    print_all_files_in_dir(pwd)

    command = 'ls -l'
    job_id  = batch_job_submit(command, 'psnehq', 'log-ls.txt') 
    print 'job_id =', job_id

    out = os_command(['df','-k','.'])
    print out

#--------------------

if __name__ == '__main__' :
    do_test()
    sys.exit('The End of Test')

#--------------------
