#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GlobalUtils...
#
#------------------------------------------------------------------------

"""Contains Global Utilities

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

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
#import time
from time import localtime, gmtime, strftime, clock, time, sleep

import numpy as np

#import commands # use 'subprocess' instead of 'commands'
import subprocess # for subprocess.Popen

from Logger import logger

#-----------------------------
# Imports for other modules --
#-----------------------------
#from PkgPackage.PkgModule import PkgClass

#import ConfigParameters as cp

#------------------------
# Exported definitions --
#------------------------

#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------

def get_list_of_files_in_dir(dirname) :
    return os.listdir(dirname)

def print_all_files_in_dir(dirname) :
    print 'List of files in the dir.', dirname
    for fname in get_list_of_files_in_dir(dirname) :
        print fname
    print '\n'

def print_list_of_files_in_dir(dirname, path_or_fname) :
    dname, fname = os.path.split(path_or_fname)     # i.e. ('work_corana', 'img-xcs-r0015-b0000.bin')
    print 'print_list_of_files_in_dir():  directory:' + dirname + '  fname:' + fname

    for fname_in_dir in get_list_of_files_in_dir(dirname) :
        if fname in fname_in_dir :
            print fname_in_dir    
    print '\n'

#----------------------------------

def subproc(command_seq) : # for example, command_seq=['bsub', '-q', cp.batch_queue, '-o', 'log-ls.txt', 'ls -l']
    p = subprocess.Popen(command_seq, stdout=subprocess.PIPE, stderr=subprocess.PIPE) #, stdin=subprocess.STDIN
    p.wait()
    out = p.stdout.read() # reads entire file
    err = p.stderr.read() # reads entire file
    return out, err


#def batch_job_submit(command, queue='psnehq', log_file='batch-log.txt') : # for example, command ='ls -l'
#    p = subprocess.Popen(['bsub', '-q', queue, '-o', log_file, command], stdout=subprocess.PIPE) #, stdin=subprocess.STDIN, stderr=subprocess.PIPE
#    p.wait()
#    #print_subproc_attributes(p)
#    line = p.stdout.readline() # read() - reads entire file
#    # here we parse the line assuming that it looks like: Job <126090> is submitted to queue <psfehq>.
#    #print line
#    line_fields = line.split(' ')
#    if line_fields[0] != 'Job' :
#        sys.exit('EXIT: Unexpected response at batch submission: ' + line)
#    job_id_str = line_fields[1].strip('<').rstrip('>')
#    return job_id_str

def batch_job_submit(command, queue='psnehq', log_file='batch-log.txt') :
    out, err = subproc(['bsub', '-q', queue, '-o', log_file, command])
    line_fields = out.split(' ')
    if line_fields[0] != 'Job' :
        sys.exit('EXIT: Unexpected response at batch submission: ' + line)
        job_id_str = 'JOB_ID_IS_UNKNOWN'
    else :
        job_id_str = line_fields[1].strip('<').rstrip('>')

    return job_id_str, out, err


def batch_job_check(job_id_str, queue='psnehq') :
    out, err = subproc(['bjobs', '-q', queue, job_id_str])
    if err != '' : return err + '\n' + out
    else         : return out


def batch_job_status(job_id_str, queue='psnehq') :
    p = subprocess.Popen(['bjobs', '-q', queue, job_id_str], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait() # short time waiting untill submission is done, 
    err = p.stderr.read() # reads entire file
    if err != '' : logger.warning('batch_job_status:\n' + err, __name__) 
    status = None
    lines  = p.stdout.readlines() # returns the list of lines in file
    if len(lines)<2 : return None
    line   = lines[1].strip('\n')
    status = line.split()[2]
    return status # it might None, 'RUN', 'PEND', 'EXIT', 'DONE', etc 


def batch_job_status_and_nodename(job_id_str, queue='psnehq') :
    p = subprocess.Popen(['bjobs', '-q', queue, job_id_str], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait() # short time waiting untill submission is done, 
    err = p.stderr.read() # reads entire file
    if err != '' : logger.warning('batch_job_status_and_nodename:\n' + err, __name__) 
    status = None
    lines  = p.stdout.readlines() # returns the list of lines in file
    if len(lines)<2 : return None, None
    line   = lines[1].strip('\n')
    fields = line.split()
    #for field in fields :
    #    print field,
    #print ' '    
    status, nodename = fields[2], fields[5]
    return status, nodename # status might None, 'RUN', 'PEND', 'EXIT', 'DONE', etc 


def remove_file(path) :
    #print 'remove file: ' + path
    logger.debug('remove: ' + path, __name__)
    p = subprocess.Popen(['rm', path], stdout=subprocess.PIPE)
    p.wait() # short time waiting untill submission is done, 


#----------------------------------

def get_text_file_content(path) :
    f=open(path,'r')
    text = f.read()
    f.close() 
    return text

#----------------------------------
# assumes: path = /reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc
def parse_xtc_path(path='.') :

    instrument = 'INS'
    experiment = 'expt'
    run_str    = 'r0000'
    run_num    = 0

    pos = path.find('/reg/d/psdm/')
    if pos != -1 :
        fields = path.split('/') 
        instrument = fields[4].upper()              # i.e. XCS 
        experiment = fields[5]                      # i.e. xcsi0112
        bname   = os.path.basename(path)            # i.e. e167-r0015-s00-c00.xtc
    try:
        run_str = bname.split('-')[1]               # i.e. r0015
        run_num = int(run_str[1:])                  # i.e. 15
    except : pass

    return instrument, experiment, run_str, run_num
    
#----------------------------------

def print_parsed_path(path) :                       # Output for path:
    print 'print_parsed_path(path): path:',         # path/reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc
    print 'exists(path)  =', os.path.exists(path)   # True 
    print 'splitext(path)=', os.path.splitext(path) # ('/reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00', '.xtc')
    print 'basename(path)=', os.path.basename(path) # e167-r0015-s00-c00.xtc
    print 'dirname(path) =', os.path.dirname(path)  # /reg/d/psdm/XCS/xcsi0112/xtc
    print 'lexists(path) =', os.path.lexists(path)  # True  
    print 'isfile(path)  =', os.path.isfile(path)   # True  
    print 'isdir(path)   =', os.path.isdir(path)    # False 
    print 'split(path)   =', os.path.split(path)    # ('/reg/d/psdm/XCS/xcsi0112/xtc', 'e167-r0015-s00-c00.xtc') 

#----------------------------------

def parse_path(path) :
    #print 'parse_path("' + path + '")'
    dname, fname = os.path.split(path)     # i.e. ('/reg/d/psdm/XCS/xcsi0112/xtc', 'e167-r0015-s00-c00.xtc')
    name, ext    = os.path.splitext(fname) # i.e. ('e167-r0015-s00-c00', '.xtc')
    return dname, name, ext

def get_dirname_from_path(path) :
    dirname, tail = os.path.split(path)
    if len(dirname) == 0 : dirname = './'
    #print 'get_dirname_from_path():  path: ' + path + '  dirname: ' + dirname
    return dirname

def get_pwd() :
    #pwdir = commands.getoutput('echo $PWD')
    out, err = subproc(['pwd'])
    pwdir = out.strip('\n')
    return pwdir

#----------------------------------

def split_string(str,separator='-s') :
    #print 'split_string("' + str + '") by the separator: "' + separator + '"'
    str_pref, str_suff = str.split(separator, 1) 
    return str_pref, str_suff

#----------------------------------

def get_item_last_name(dsname):
    """Returns the last part of the full item name (after last slash)"""
    path,name = os.path.split(str(dsname))
    return name

def get_item_path_to_last_name(dsname):
    """Returns the path to the last part of the item name"""
    path,name = os.path.split(str(dsname))
    return path

def get_item_path_and_last_name(dsname):
    """Returns the path and last part of the full item name"""
    path,name = os.path.split(str(dsname))
    return path, name

#----------------------------------

def get_item_second_to_last_name(dsname):
    """Returns the 2nd to last part of the full item name"""
    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    return name2 

#----------------------------------

def get_item_third_to_last_name(dsname):
    """Returns the 3nd to last part of the full item name"""

    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    path3,name3 = os.path.split(str(path2))

    str(name3)

    return name3 

#----------------------------------

def get_item_name_for_title(dsname):
    """Returns the last 3 parts of the full item name (after last slashes)"""

    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    path3,name3 = os.path.split(str(path2))

    return name3 + '/' + name2 + '/' + name1

#----------------------------------

def sleep_sec(sec):
    sleep(sec)

#----------------------------------

def get_time_sec():
    return time()

#----------------------------------

#def get_time_sec():
#    return clock()

#----------------------------------

def get_current_local_time_tuple():
    return localtime()

def get_current_gm_time_tuple():
    return gmtime()

#----------------------------------

def get_current_local_time_stamp(fmt='%Y-%m-%d %H:%M:%S %Z'):
    return strftime(fmt, localtime())

def get_current_gm_time_stamp(fmt='%Y-%m-%d %H:%M:%S %Z'):
    return strftime(fmt, gmtime())

#----------------------------------
#----------------------------------
#----------------------------------

def get_array_from_file(fname) :
    if os.path.lexists(fname) :
        logger.info('Get array from file: ' + fname, __name__)         
        return np.loadtxt(fname, dtype=np.float32)
    else :
        logger.warning(fname + ' is not available', __name__)         
        return None

#----------------------------------

def get_text_list_from_file(fname) :
    if not os.path.lexists(fname) :
        logger.warning(fname + ' is not available', __name__)         
        return None
    logger.info('Read and return as a 2-d tuple text array from file: ' + fname, __name__)         

    list_recs = []    
    f=open(fname,'r')
    for line in f :
        if len(line) == 1 : continue # line is empty
        fields = line.rstrip('\n').split() #,1)
        list_recs.append(fields)
    f.close() 
    return list_recs

#-----------------------------

#    def get_pedestals_from_file(self) :
#        fname = fnm.path_pedestals_ave()
#        if os.path.lexists(fname) :
#            return gu.get_array_from_file( fname )
#        else :
#            logger.warning(fname + ' is not available', __name__)         
#            return None

#    
#----------------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    print 'Time (sec) :', int( get_time_sec() )

    print 'Time local :', get_current_local_time_tuple()
    print 'Time (GMT) :', get_current_gm_time_tuple()

    print 'Time local :', get_current_local_time_stamp()
    print 'Time (GMT) :', get_current_gm_time_stamp()

    pwd = get_pwd()
    print 'pwd =', pwd
    print_all_files_in_dir(pwd)

    command = 'ls -l'
    job_id_str, out, err = batch_job_submit(command, 'psnehq', 'log-ls.txt') 
    print 'err =', err
    print 'out =', out
    print 'job_id_str =', job_id_str

    #sleep(5) # sleep time in sec
    #print batch_job_status(job_id_str, 'psnehq')
    #print batch_job_status_and_nodename(job_id_str, 'psnehq')

    #out,err = subproc(['df','-k','.'])
    #print 'out=', out
    #print 'err=', err

    #path = '/reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc'
    #print_parsed_path(path)
    #print 'parse_xtc_path(): ', parse_xtc_path()
    #print 'parse_xtc_path(path): ', parse_xtc_path(path)

    sys.exit ( "End of test" )

#----------------------------------
