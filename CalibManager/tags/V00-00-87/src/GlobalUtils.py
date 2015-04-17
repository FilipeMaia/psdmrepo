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
import pwd
import socket
#import time
from time import localtime, gmtime, strftime, clock, time, sleep
#from datetime import datetime
import tempfile

import numpy as np
from commands import getoutput

#import commands # use 'subprocess' instead of 'commands'
import subprocess # for subprocess.Popen

from Logger import logger
from PyQt4 import QtGui, QtCore
from LogBook import message_poster
from GUIPopupCheckList import *
from GUIPopupRadioList import *

import PyCSPadImage.CSPAD2x2ImageUtils as cspad2x2img
import PyCSPadImage.CSPADImageUtils    as cspadimg

#-----------------------------
# Imports for other modules --
#-----------------------------
#from PkgPackage.PkgModule import PkgClass

from CalibFileFinder import *

#------------------------
# Exported definitions --
#------------------------

#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------        

#-----------------------------

def stringOrNone(value):
    if value is None : return 'None'
    else             : return str(value)

def intOrNone(value):
    if value is None : return None
    else             : return int(value)

#-----------------------------

def list_of_int_from_list_of_str(list_in):
    """Converts  ['0001', '0202', '0203', '0204',...] to [1, 202, 203, 204,...]
    """
    list_out = []
    for item in list_in :
        list_out.append(int(item))
    return list_out

#-----------------------------

def list_of_str_from_list_of_int(list_in, fmt='%04d'):
    """Converts [1, 202, 203, 204,...] to ['0001', '0202', '0203', '0204',...]
    """
    list_out = []
    for item in list_in :
        list_out.append( fmt % item )
    return list_out

#-----------------------------


def create_directory(dir) : 
    if os.path.exists(dir) :
        logger.info('Directory exists: ' + dir, __name__) 
    else :
        os.makedirs(dir)
        #os.fchmod(dir,770)
        logger.info('Directory created: ' + dir, __name__) 


def get_list_of_files_in_dir(dirname) :
    return os.listdir(dirname)


def print_all_files_in_dir(dirname) :
    print 'List of files in the dir.', dirname
    for fname in get_list_of_files_in_dir(dirname) :
        print fname
    print '\n'


def get_list_of_files_in_dir_for_ext(dir, ext='.xtc'):
    """Returns the list of files in the directory for specified extension or None if directory is None."""
    if dir is None : return []
    if not os.path.exists(dir) : return [] 
    
    list_of_files_in_dir = os.listdir(dir)
    list_of_files = []
    for fname in list_of_files_in_dir :
        if os.path.splitext(fname)[1] == ext :
            list_of_files.append(fname)
    return sorted(list_of_files)


def get_list_of_files_in_dir_for_part_fname(dir, pattern='-r0022'):
    """Returns the list of files in the directory for specified file name pattern or [] - empty list."""
    if dir is None : return []
    if not os.path.exists(dir) : return [] 
    
    list_of_files_in_dir = os.listdir(dir)
    list_of_files = []
    for fname in list_of_files_in_dir :
        if pattern in fname :
            fpath = os.path.join(dir,fname)
            list_of_files.append(fpath)
    return sorted(list_of_files)



def print_list_of_files_in_dir(dirname, path_or_fname) :
    dname, fname = os.path.split(path_or_fname)     # i.e. ('work_corana', 'img-xcs-r0015-b0000.bin')
    print 'print_list_of_files_in_dir():  directory:' + dirname + '  fname:' + fname

    for fname_in_dir in get_list_of_files_in_dir(dirname) :
        if fname in fname_in_dir :
            print fname_in_dir    
    print '\n'


def get_path_owner(path) :
    stat = os.stat(path)
    #print ' stat =', stat
    pwuid = pwd.getpwuid(stat.st_uid)
    #print ' pwuid =', pwuid
    user_name  = pwuid.pw_name
    #print ' uid = %s   user_name  = %s' % (uid, user_name)
    return user_name

def get_path_mode(path) :
    return os.stat(path).st_mode


#----------------------------------

def get_tempfile(mode='r+b',suffix='.txt') :
    tf = tempfile.NamedTemporaryFile(mode=mode,suffix=suffix)
    return tf # .name
 
#----------------------------------

def call(command_seq, shell=False) :
    subprocess.call(command_seq, shell=shell) # , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)

#----------------------------------

def subproc_in_log(command_seq, logname, env=None, shell=False) : # for example, command_seq=['bsub', '-q', cp.batch_queue, '-o', 'log-ls.txt', 'ls -l']
    log = open(logname, 'w')
    p = subprocess.Popen(command_seq, stdout=log, stderr=subprocess.PIPE, env=env, shell=shell) #, stdin=subprocess.STDIN
    p.wait()
    err = p.stderr.read() # reads entire file
    return err

#----------------------------------

def subproc(command_seq, env=None, shell=False) : # for example, command_seq=['bsub', '-q', cp.batch_queue, '-o', 'log-ls.txt', 'ls -l']
    p = subprocess.Popen(command_seq, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, shell=shell) #, stdin=subprocess.STDIN
    p.wait()
    out = p.stdout.read() # reads entire file
    err = p.stderr.read() # reads entire file
    return out, err

#----------------------------------

def send_msg_with_att_to_elog(inst='AMO', expt='amodaq09', run='825', tag='TAG1',
                              msg='EMPTY MESSAGE', fname_text=None, fname_att=None, resp=None) :

    poster = message_poster.message_poster_self ( inst, experiment=expt )

    if resp == 'None' : msg_id = poster.post ( msg, attachments=[fname_att], run_num=run, tags=tag )
    else              : msg_id = poster.post ( msg, attachments=[fname_att], parent_message_id = int(resp) )    

    msg_in_log = 'Message with id: %d is submitted to ELog' % (msg_id)

    logger.info(msg_in_log, __name__)
    return msg_id
    

#----------------------------------

def send_msg_with_att_to_elog_v0(inst='AMO', expt='amodaq09', run='825', tag='TAG1',
                              msg='EMPTY MESSAGE', fname_text=None, fname_att=None, resp=None) :
    #pypath = os.getenv('PYTHONPATH')
    my_env = os.environ
    pypath = my_env.get('PYTHONPATH', '')
    my_env['PYTHONPATH'] = pypath + \
                  ':/reg/g/pcds/pds/grabber/lib/python2.7/site-packages/' 
    command_seq = ['/reg/g/pcds/pds/grabber/bin/LogBookPost_self.py',
                   '-i', inst,
                   '-e', expt,
                   '-r', run,
                   '-t', tag
                   ]

    if msg is not None and msg != 'None' :
        command_seq.append('-m')
        command_seq.append(msg)

    if fname_text is not None and fname_text != 'None' :
        command_seq.append('-f')
        command_seq.append(fname_text)

    if fname_att is not None and fname_att != 'None' :
        command_seq.append('-a')
        command_seq.append(fname_att)

    if resp is not None and resp != 'None' :
        command_seq.append('-???')
        command_seq.append(resp)


    #print 'my_env for PYTHONPATH: ', my_env['PYTHONPATH']
    #print 'command_seq: ', command_seq

    str_command_seq = ''
    for v in command_seq : str_command_seq += v + ' '

    logger.info('command_seq: ' + str_command_seq, __name__) 

    #==================
    #out, err = 'submission procedure must be uncommented...', 'Responce should be ok, but...'
    out, err = subproc(command_seq, env=my_env)
    #==================

    #print 'out:\n', out
    #print 'err:\n', err
    if err != '' : return err + '\n' + out
    else         : return out

#----------------------------------


#def batch_job_submit(command, queue='psnehq', log_file='batch-log.txt') : # for example, command ='ls -l'
#    p = subprocess.Popen(['bsub', '-q', queue, '-o', log_file, command], stdout=subprocess.PIPE) #, stdin=subprocess.STDIN, stderr=subprocess.PIPE
#    p.wait()
#    #print_subproc_attributes(p)
#    line = p.stdout.readline() # read() - reads entire file
#    # here we parse the line assuming that it looks like: Job <126090> is submitted to queue <psfehq>.
#    #print line
#    line_fields = line.split()
#    if line_fields[0] != 'Job' :
#        sys.exit('EXIT: Unexpected response at batch submission: ' + line)
#    job_id_str = line_fields[1].strip('<').rstrip('>')
#    return job_id_str

def batch_job_submit(command, queue='psnehq', log_file='batch-log.txt') :

    if os.path.lexists(log_file) : remove_file(log_file)

    out, err = subproc(['bsub', '-q', queue, '-o', log_file, command])
    line_fields = out.split()
    if line_fields[0] != 'Job' :
        msg = 'EXIT: Unexpected response at batch submission:\nout: %s \nerr: %s'%(out, err)
        print msg
        logger.warning(msg, __name__) 
        #sys.exit(msg)
        job_id_str = 'JOB_ID_IS_UNKNOWN'
    else :
        job_id_str = line_fields[1].strip('<').rstrip('>')

    if err != '' :
        msg = ''
        if 'job being submitted without an AFS token' in err :
            msg = err + '      This warning does not matter for jobs on LCLS NFS, continue' 
        else :
            msg = '\n' + 80*'!' + '\n' + err + 80*'!' + '\n'

        logger.warning(msg, __name__) 

    logger.info(out, __name__) 

    return job_id_str, out, err


def batch_job_check(job_id_str, queue='psnehq') :
    out, err = subproc(['bjobs', '-q', queue, job_id_str])    
    if err != '' : return err + '\n' + out
    else         : return out


def bsub_is_available() :
    out, err = subproc(['which', 'bsub'])
    if err != '' :
        msg = 'Check if bsub is available on this node:\n' + err + \
              '\nbsub IS NOT available in current configuration of your node... (try command: which bsub)\n'
        print msg
        logger.warning(msg, __name__)         
        return False
    else :
        return True


def batch_job_kill(job_id_str) :
    command = ['kill', job_id_str]
    #print 'command:', command
    out, err = subproc(command)
    if err != '' : return err + '\n' + out
    else         : return out


def batch_job_status(job_id_str, queue='psnehq') :
    """Returns the batch job status, for example 'RUN', 'PEND', 'EXIT', 'DONE', etc."""
    p = subprocess.Popen(['bjobs', '-q', queue, job_id_str], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait() # short time waiting untill submission is done, 
    err = p.stderr.read() # reads entire file
    if err != '' : logger.warning('batch_job_status:\n' + err, __name__) 
    status = None
    lines  = p.stdout.readlines() # returns the list of lines in file
    #for line in lines : print 'batch_job_status: ' + line
    if len(lines)<2 : return None
    line   = lines[1].strip('\n')
    status = line.split()[2]
    #print 'status: ', status
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
    #print 'remove file: ' + pathOut[11]: ''

    logger.debug('remove: ' + path, __name__)
    p = subprocess.Popen(['rm', path], stdout=subprocess.PIPE)
    p.wait() # short time waiting untill submission is done, 


#----------------------------------

def load_textfile(path) :
    f=open(path,'r')
    text = f.read()
    f.close() 
    return text

#----------------------------------

def save_textfile(text, path, mode='w') :
    """Saves text in file specified by path. mode: 'w'-write, 'a'-append 
    """
    f=open(path,mode)
    f.write(text)
    f.close() 

#----------------------------------

def xtc_fname_parser_helper( part, prefix ) :    
    """In parsing the xtc file name, this function extracts the string after expected prefix, i.e. 'r0123' -> '0123'"""
    if len(part)>1 and part[0] == prefix :
        try :
            return part[1:]
        except :
            pass
    return None


def parse_xtc_file_name(fname):
    """Parse the file name like e170-r0003-s00-c00.xtc and return ('170', '0003', '00', '00', '.xtc')"""
    name, _ext = os.path.splitext(fname) # i.e. ('e167-r0015-s00-c00', '.xtc')
    #print 'name, _ext = ', name, _ext 
    parts = name.split('-') # it gives parts = ('e167', 'r0015', 's00', 'c00')

    _expnum = None
    _runnum = None
    _stream = None
    _chunk  = None

    parts = map( xtc_fname_parser_helper, parts, ['e', 'r', 's', 'c'] )

    if None not in parts :
        _expnum = parts[0]
        _runnum = parts[1]
        _stream = parts[2]
        _chunk  = parts[3]

    #print 'e,r,s,c,ext:', _expnum, _runnum, _stream, _chunk, _ext
    return _expnum, _runnum, _stream, _chunk, _ext

#----------------------------------
# assumes: path = .../<inst>/<experiment>/xtc/<file-name>.xtc
#        i.e         /reg/data/ana12/xcs/xcsi0112/xtc/e167-r0020-s00-c00.xtc 
#        or          /reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc
def parse_xtc_path(path='.') :

    logger.debug( 'parse_xtc_path(...): ' + str(path), __name__)

    instrument = 'INS'
    experiment = 'expt'
    run_str    = 'r0000'
    run_num    = 0

    if path is None : return instrument, experiment, run_str, run_num

    bname   = os.path.basename(path)                # i.e. e167-r0015-s00-c00.xtc
    try:
        run_str = bname.split('-')[1]               # i.e. r0015
        run_num = int(run_str[1:])                  # i.e. 15
    except :
        print 'Unexpected xtc file name:', bname 
        print 'Use default instrument, experiment, run_str, run_num: ', instrument, experiment, run_str, run_num
        return instrument, experiment, run_str, run_num
        pass

    dirname  = os.path.dirname(path)                # i.e /reg/data/ana12/xcs/xcsi0112/xtc
    fields   = dirname.split('/')
    n_fields = len(fields)
    if n_fields < 4 :
        msg1 = 'Unexpected xtc dirname: %s: Number of fields in dirname = %d' % (dirname, n_fields)
        msg2 = 'Use default instrument, experiment, run_str, run_num: %s %s %s %d' % (instrument, experiment, run_str, run_num)
        logger.warning( msg1+msg2, __name__)         
        return instrument, experiment, run_str, run_num
            
    xtc_subdir = fields[-1]                         # i.e. xtc
    experiment = fields[-2]                         # i.e. xcsi0112
    instrument = fields[-3].upper()                 # i.e. XCS 
    msg = 'Set instrument, experiment, run_str, run_num: %s %s %s %d' % (instrument, experiment, run_str, run_num)
    logger.debug( msg, __name__)         

    return instrument, experiment, run_str, run_num
    
#----------------------------------
# assumes: path = /reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc
def parse_xtc_path_v0(path='.') :

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

def xtc_fname_for_all_chunks(path='e167-r0015-s00-c00.xtc') :
    """Converts the xtc file name e167-r0015-s00-c00.xtc or complete path
       path/reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc to the wildname:
       e167-r0015-*.xtc
    """
    bname  = os.path.basename(path)
    fields = bname.split('-')
    fname_all = fields[0] + '-' + fields[1] + '-*.xtc'
    return fname_all


#----------------------------------

def list_of_calib_files_with_run_range(list_of_files) :

    list_out = []
    for file in list_of_files :
        #print file,
        str_range = run_range_from_calib_fname(file) 
        if str_range is None : continue
        list_out.append(str_range.replace('9999','end'))

    for range in sorted(list_out) :
        print range
        


        #beg = begin_run_from_calib_fname(file)
        #if beg is None : continue
        #dic_num_file[beg] = fname



#----------------------------------
# assumes: path = .../<inst>/<experiment>/calib
# for example        /reg/d/psdm/CXI/cxitut13/calib
# or                 /reg/d/psdm/XPP/xpptut13/calib

def get_text_content_of_calib_dir_for_detector(path, det='cspad', subdir='CsPad::CalibV1', level=0, calib_type=None) :

    #logger.debug( 'get_txt_content_of_calib_dir_for_detector(...): ' + path, __name__)
    det_lower = det.lower()
    txt = '    '
    if level == 0 : txt = 85*'-' + '\nContent of: %s for detector: %s' % (path, det)

    if not os.path.exists(path) :
        txt = 'Path %s DOES NOT EXIST' % path
        return txt

    list_of_fnames = os.listdir(path)

    if list_of_fnames == [] :
        txt += '\n' + (level+1)*'    ' + 'Directory IS EMPTY!'        
        return txt

    if level == 3 :

       list_of_cfiles = list_of_sorted_calib_files_from_list_of_files(list_of_fnames)
       dict_fname_range = dict_calib_file_actual_run_range(list_of_cfiles)

       for cfile in list_of_cfiles :
           #print cfile.get_basename()
           fname = cfile.get_basename()
           range = dict_fname_range[fname]
           txt += '\n' + (level+1)*'    '
           if range[0] == -1 :
               txt += '%s  file is not used' % fname.ljust(14)
           else :
               txt += '%s  run range %04d - %04d' % (fname.ljust(14), range[0], range[1])

       return txt

    for i,file in enumerate(list_of_fnames) :
        fname_lower = file.lower()
        #cond0 = level==0 and det.lower()+'::' in fname_lower
        cond0 = level==0 and subdir in file
        cond1 = level==1 and det.lower()+'.'  in fname_lower
        cond2 = level==2 and (file == calib_type or calib_type is None)

        if not ( cond0 or cond1 or cond2) : continue
        
        txt +='\n' + (level+1)*'    ' + file

        path_to_child = os.path.join(path, file)
        if os.path.isdir(path_to_child) : txt += get_text_content_of_calib_dir_for_detector(path_to_child, det, subdir, level=level+1, calib_type=calib_type)
             
    return txt


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

def get_cwd() :
    """get corrent work directory"""
    return os.getcwd()

def get_hostname() :
    return socket.gethostname()
 
def get_enviroment(env='USER') :
    """Returns the value of specified by string name environment variable
    """
    return os.environ[env]

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

def get_local_time_tuple(t_sec_epoch):
    return localtime(t_sec_epoch)

def get_gm_time_tuple(t_sec_epoch):
    return gmtime(t_sec_epoch)

#----------------------------------

def get_current_local_time_stamp(fmt='%Y-%m-%d %H:%M:%S %Z'):
    return strftime(fmt, localtime())

def get_current_gm_time_stamp(fmt='%Y-%m-%d %H:%M:%S %Z'):
    return strftime(fmt, gmtime())

#----------------------------------

def get_local_time_str(time_sec, fmt='%Y-%m-%d %H:%M:%S %Z'):
    #return ctime(time_sec)
    return strftime(fmt, get_local_time_tuple(time_sec))

#----------------------------------

def get_gm_time_str(time_sec, fmt='%Y-%m-%d %H:%M:%S %Z'):
    #return ctime(time_sec)
    return strftime(fmt, get_gm_time_tuple(time_sec))

#----------------------------------

def selectFromListInPopupMenu(list):
    """Shows the list as a pop-up menu and returns the selected item as a string or None"""
    popupMenu = QtGui.QMenu()
    for item in list :
        popupMenu.addAction( item )

    item_selected = popupMenu.exec_(QtGui.QCursor.pos())

    if item_selected is None : return None
    else                     : return str(item_selected.text()) # QString -> str

#----------------------------------

def changeCheckBoxListInPopupMenu(list, win_title='Set check boxes'):
    """Shows the list of check-boxes as a dialog pop-up menu and returns the (un)changed list"""
    popupMenu = GUIPopupCheckList(None, list, win_title)
    #popupMenu.move(QtCore.QPoint(50,50))
    popupMenu.move(QtGui.QCursor.pos())
    response = popupMenu.exec_()

    if   response == QtGui.QDialog.Accepted :
        logger.debug('New checkbox list is accepted', __name__)         
        return 1
    elif response == QtGui.QDialog.Rejected :
        logger.debug('Will use old checkbox list', __name__)
        return 0
    else                                    :
        logger.error('Unknown response...', __name__)
        return 2


#----------------------------------

def selectRadioButtonInPopupMenu(dict_of_pars, win_title='Select option', do_confirm=False):
    """Popup GUI to select radio button from the list:  dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}
    """
    popupMenu = GUIPopupRadioList(None, dict_of_pars, win_title, do_confirm)
    #popupMenu.move(QtCore.QPoint(50,50))
    popupMenu.move(QtGui.QCursor.pos()-QtCore.QPoint(100,100))
    return popupMenu.exec_() # QtGui.QDialog.Accepted or QtGui.QDialog.Rejected

#----------------------------------

def get_array_from_file(fname, dtype=np.float32) :
    if fname is None or fname=='' :
        logger.warning('File name is not specified...', __name__)         
        return None
        
    elif os.path.lexists(fname) :

        fname_ext = os.path.splitext(fname)[1]
        #print 'fname_ext', fname_ext

        logger.info('Get array from file: ' + fname, __name__)         

        if fname_ext == '.npy' :
            return np.load(fname) # load as binary
        else :
            return np.loadtxt(fname, dtype=dtype)
    else :
        logger.warning(fname + ' is not available', __name__)         
        return None

#----------------------------------

def get_image_array_from_file(fname, dtype=np.float32) :

    arr = get_array_from_file(fname, dtype)
    if arr is None :
        return None

    img_arr = None
        
    if   arr.size == 32*185*388 : # CSPAD
        arr.shape = (32*185,388) 
        img_arr = cspadimg.get_cspad_raw_data_array_image(arr)

    elif arr.size == 185*388*2 : # CSPAD2x2
        #arr.shape = (185,388,2) 
        img_arr = cspad2x2img.get_cspad2x2_non_corrected_image_for_raw_data_array(arr)

    elif arr.ndim == 2 : # Use it as any camera 2d image
        img_arr = arr

    else :
        msg = 'Array loaded from file: %s\n has shape: %s, size: %s, ndim: %s' % \
              (fname, str(arr.shape), str(arr.size), str(arr.ndim))
        msg += '\nIs not recognized as image'
        logger.warning(msg, __name__)         
        return None

    return img_arr

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

#----------------------------------

def get_list_of_enumerated_file_names(path1='file.dat', len_of_list=0) :
    """From pattern of the path it makes a list of files with indexes.
    For example, for path1='file.dat', it returns [file-00.dat, file-01.dat, ..., file-<N-1>.dat], where N = len_of_list
    """
    if len_of_list < 2 : return [path1]
    name, ext = os.path.splitext(path1)
    return ['%s-%02i%s' % (name, i, ext) for i in range(len_of_list) ]

#----------------------------------

def get_list_of_files_for_list_of_insets(path1='file.dat', list_of_insets=[]) :
    """Returns the list of file names, where the file name is a combination of path1 and inset from list
    """
    if list_of_insets == [] : return [] # [path1]
    name, ext = os.path.splitext(path1)
    return ['%s-%s%s' % (name, src, ext) for src in list_of_insets]

#----------------------------------
#----------------------------------
#----------------------------------

def printStyleInfo(widg):
    qstyle     = widg.style()
    qpalette   = qstyle.standardPalette()
    qcolor_bkg = qpalette.color(1)
    #r,g,b,alp  = qcolor_bkg.getRgb()
    msg = 'Background color: r,g,b,alpha = %d,%d,%d,%d' % ( qcolor_bkg.getRgb() )
    logger.debug(msg)


#----------------------------------

def get_save_fname_through_dialog_box(parent, path0, dial_title, filter='*.txt'):       

    path = str( QtGui.QFileDialog.getSaveFileName(parent,
                                                  caption   = dial_title,
                                                  directory = path0,
                                                  filter    = filter
                                                  ) )
    if path == '' :
        logger.debug('Saving is cancelled.', 'get_save_fname_through_dialog_box')
        return None
    logger.info('Output file: ' + path, 'get_save_fname_through_dialog_box')
    return path

#----------------------------------

def get_open_fname_through_dialog_box(parent, path0, dial_title, filter='*.txt'):       

    path = str( QtGui.QFileDialog.getOpenFileName(parent, dial_title, path0, filter=filter) )
    dname, fname = os.path.split(path)
    if dname == '' or fname == '' :
        logger.info('Input directiry name or file name is empty... keep file path unchanged...')
        #print 'Input directiry name or file name is empty... keep file path unchanged...'
        return None
    logger.info('Input file: ' + path, 'get_open_fname_through_dialog_box') 
    #print 'Input file: ' + path
    return path

#----------------------------------

def confirm_dialog_box(parent=None, text='Please confirm that you aware!', title='Please acknowledge') :
        """Pop-up MODAL box for confirmation"""

        mesbox = QtGui.QMessageBox(parent, windowTitle=title,
                                           text=text,
                                           standardButtons=QtGui.QMessageBox.Ok)
               #standardButtons=QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
        #mesbox.setDefaultButton(QtGui.QMessageBox.Ok)
        #mesbox.setMinimumSize(400, 200)
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 100);" # Pinkish
        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #mesbox.setStyleSheet (style)

        clicked = mesbox.exec_() # DISPLAYS THE QMessageBox HERE

        #if   clicked == QtGui.QMessageBox.Save :
        #    logger.info('Saving is requested', __name__)
        #elif clicked == QtGui.QMessageBox.Discard :
        #    logger.info('Discard is requested', __name__)
        #else :
        #    logger.info('Cancel is requested', __name__)
        #return clicked

        logger.info('You acknowkeged that saw the message:\n' + text, 'confirm_dialog_box')
        return

#----------------------------------

def confirm_or_cancel_dialog_box(parent=None, text='Please confirm or cancel', title='Confirm or cancel') :
        """Pop-up MODAL box for confirmation"""

        mesbox = QtGui.QMessageBox(parent, windowTitle=title,
                                           text=text,
                                           standardButtons=QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel)
               #standardButtons=QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
        mesbox.setDefaultButton(QtGui.QMessageBox.Ok)
        #mesbox.setMinimumSize(400, 200)
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 100);" # Pinkish
        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #mesbox.setStyleSheet (style)

        clicked = mesbox.exec_() # DISPLAYS THE QMessageBox HERE

        if   clicked == QtGui.QMessageBox.Ok     : return True
        elif clicked == QtGui.QMessageBox.Cancel : return False
        else                                     : return False

#----------------------------------

def help_dialog_box(parent=None, text='Help message goes here', title='Help') :
        """Pop-up NON-MODAL box for help etc."""

        messbox = QtGui.QMessageBox(parent, windowTitle=title,
                                           text=text,
                                           standardButtons=QtGui.QMessageBox.Close)
        messbox.setStyleSheet (cp.styleBkgd)
        messbox.setWindowModality (QtCore.Qt.NonModal)
        messbox.setModal (False)
        #clicked = messbox.exec_() # For MODAL dialog
        clicked = messbox.show()  # For NON-MODAL dialog
        logger.info('Help window is open' + text, 'help_dialog_box')
        return messbox

#----------------------------------

def arr_rot_n90(arr, rot_ang_n90=0) :
    if   rot_ang_n90==  0 : return arr
    elif rot_ang_n90== 90 : return np.flipud(arr.T)
    elif rot_ang_n90==180 : return np.flipud(np.fliplr(arr))
    elif rot_ang_n90==270 : return np.fliplr(arr.T)
    else                  : return arr

#----------------------------------

def has_kerberos_ticket():
    """Checks to see if the user has a valid Kerberos ticket"""
    #stream = os.popen('klist -s')
    #output = getoutput('klist -4')
    #resp = subprocess.call(["klist", "-s"])
    return True if subprocess.call(["klist", "-s"]) == 0 else False


def check_token(do_print=False) :
    token = getoutput('tokens')
    #if do_print : print token
    status = True if 'Expire' in token else False
    timestamp = parse_token(token) if status else ''
    msg = 'Your AFS token %s %s' % ({True:'IS valid until', False:'IS NOT valid'}[status], timestamp)
    if do_print : print msg
    return status, msg


def get_afs_token(do_print=False) :
    output = getoutput('aklog')
    if do_print : print str(output)
    return output


def parse_token(token) :
    """ from string like: User's (AFS ID 5269) tokens for afs@slac.stanford.edu [Expires Feb 28 19:16] 54 75 Expires Feb 28 19:16
        returns date/time: Feb 28 19:16
    """
    timestamp = ''

    for line in token.split('\n') :
        pos_beg = line.find('[Expire')
        if pos_beg == -1 : continue
        pos_end = line.find(']', pos_beg)
        #print line
        timestamp = line[pos_beg+9:pos_end]

        #date_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
        #date_object = datetime.strptime(timestamp, '%b %d %H:%M')
        #print 'date_object', str(date_object)

    return timestamp 

#---------------------------------

def ready_to_start(check_bits=0777, fatal_bits=0777) :
    """Check availability of services and credentuals marked by the check_bits"""

    if check_bits & 1 and not is_good_lustre_version() :
        print 'WARNING: The host "%s" uses old lustre driver version. CHANGE HOST !!!' % get_hostname()
        if fatal_bits & 1 : return False
	else              : print 'Continue with old lustre driver...'

    if check_bits & 2 and not has_kerberos_ticket() :
        print 'WARNING: Kerberos ticket is missing. To get one use command: kinit'
        if fatal_bits & 2 : return False
	else              : print 'Continue without Kerberos ticket...'

    if check_bits & 4 : 
        status, msg = check_token(do_print=False)
        if not status :
            get_afs_token(do_print=True)

            status, msg = check_token(do_print=False)
            if not status :
                print 'WARNING: AFS token is missing. To get one use commands: kinit; aklog'
                if fatal_bits & 4 : return False
	        else              : print 'Continue without AFS token...'

    return True

#---------------------------------


#----------------------------------

def text_status_of_queues(lst_of_queues=['psanaq', 'psnehq', 'psfehq', 'psnehprioq', 'psfehprioq']):
    """Checks status of queues"""
    cmd = 'bqueues %s' % (' '.join(lst_of_queues))
    return cmd, getoutput(cmd)


def msg_and_status_of_queue(queue='psnehq') :
    """Returns status of queue for command: bqueues <queue-name>"""
       #QUEUE_NAME      PRIO STATUS          MAX JL/U JL/P JL/H NJOBS  PEND   RUN  SUSP 
       #psanacsq        115  Open:Active       -    -    -    -     0     0     0     0
    cmd, txt = text_status_of_queues([queue])
    lines = txt.split('\n')
    fields = lines[-1].split()
    msg = 'Status : %s' % fields[2]
    txt = 'Command: %s\n%s\n%s' % (cmd, txt, msg)
    if fields[2] == 'Open:Active' :
        return txt, True
    else :
        return txt, False


def text_sataus_of_lsf_hosts(farm='psnehfarm'):
    """Returns text output of the command: bhosts farm"""
    cmd = 'bhosts %s' % farm
    return cmd, getoutput(cmd)
    

def msg_and_status_of_lsf(farm='psnehfarm', print_bits=0):
    """Checks the LSF status for requested farm"""

    if print_bits & 1 : print 'farm =', farm

    cmd, output = text_sataus_of_lsf_hosts(farm)

    if print_bits & 2 : print 'list_of_hosts:\n', output
    lines = output.split('\n')

       #HOST_NAME          STATUS       JL/U    MAX  NJOBS    RUN  SSUSP  USUSP    RSV 
       #psanacs001         unavail         -     16      0      0      0      0      0
       #psanacs002         closed          -     16     16     16      0      0      0
       #psanacs003         ok              -     16      0      0      0      0      0

    count_nodes = 0
    count_ok = 0
    count_closed = 0
    count_unavail = 0

    for line in lines :
        #print line
        fields = line.split()
        if fields[0][0:5] == 'psana': count_nodes += 1
        if fields[1] == 'ok'        : count_ok += 1
        if fields[1] == 'closed'    : count_closed += 1
        if fields[1] == 'unavail'   : count_unavail += 1

    status = True if count_ok > 0 else False

    msg = 'Number of nodes:%d, ok:%d, closed:%d, unavail:%d, status:%s' %(count_nodes, count_ok, count_closed, count_unavail, status)
    if print_bits & 4 : print msg

    txt = 'Command: %s\n%s\n%s' % (cmd, output, msg)
    return txt, status

#----------------------------------

def get_pkg_version(pkg_name='CalibManager') :
    """Returns the latest revision number of the package"""
    str_revision = __version__.split(':')[1].rstrip('$').strip()
    return 'Rev-%s' % str_revision

#----------------------------------

def get_pkg_tag(pkg_name='CalibManager') :
    """Returns the latest version of the package - VERY SLOW COMMAND"""
    try :
        cmd = 'psvn tags %s' % pkg_name
        output = getoutput(cmd)
        lines = output.split('\n')
        last_line = lines[-1]
        fields = last_line.split()
        version = fields[-1].rstrip('/')
        #print cmd, '\n', output
        #print 'Last line: ', last_line
        #print 'Version: ', version
        return version
    except :
        return 'V-is-N/A'
    
#----------------------------------

def is_good_lustre_version() :
    """Checks the lustre version and returns True/False for new/ols version"""
    try :
        cmd = '/sbin/lsmod | grep lustre'
        output = getoutput(cmd)
        lines = output.split('\n')
        line_one = lines[1]
        for line in lines :
            if line[0:3] == 'lov' : line_one = line

        fields = line_one.split()
        version = fields[1]

        #print cmd, '\n', output
        #print 'Necessary line_one: ', line_one
        #print 'Lustre Version: ', version

        return False if version == '354504' else True

    except :
        return True
    

#----------------------------------

#    def get_pedestals_from_file(self) :
#        fname = fnm.path_peds_ave()
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
    #print_all_files_in_dir(pwd)

    #command = 'ls -l'
    #job_id_str, out, err = batch_job_submit(command, 'psnehq', 'log-ls.txt') 
    #print 'err =', err
    #print 'out =', out
    #print 'job_id_str =', job_id_str

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

    #send_msg_with_att_to_elog(fname_att='../../work/cora-xcsi0112-r0015-data-time-plot.png')

    #print 'xtc_fname_for_all_chunks(...): ', xtc_fname_for_all_chunks('e308-r0178-s02-c00.xtc')
    #print 'xtc_fname_for_all_chunks(...): ', xtc_fname_for_all_chunks('/reg/d/psdm/XPP/xpptut13/xtc/e308-r0178-s02-c00.xtc')

    #print 'Test 1:\n' + get_text_content_of_calib_dir_for_detector(path='/reg/d/psdm/XPP/xpptut13/calibXXX', subdir='CsPad::CalibV1', det='CSPAD')
    #print 'Test 2:\n' + get_text_content_of_calib_dir_for_detector(path='/reg/d/psdm/XPP/xpptut13/calib/', subdir='CsPad2x2::CalibV1', det='CSPAD2x2')
#    print 'Test 3:\n' + get_text_content_of_calib_dir_for_detector(path='/reg/d/psdm/XPP/xpptut13/calib', subdir='CsPad::CalibV1', det='CSPAD', calib_type='tilt')

    #list_of_files = ['220-230.data', '220-end.data', '221-240.data', '528-end.data', '222-end.data', '659-800.data', '373-end.data', '79-end.data', '45-end.data'] 
    #list_of_calib_files_with_run_range(list_of_files)

    #status, msg = check_token(do_print=True)

    #print 'Package version: ', get_pkg_version('CalibManager')

    #print 'has_kerberos_ticket(): ', has_kerberos_ticket()

    #status = is_good_lustre_version()
    #print 'Lustre version status: %s' % status

    #farm='psfehfarm' # psnehfarm, 'psanafarm'
    #output, status = msg_and_status_of_lsf(farm)
    #queue='psfehq' # psnehq, psfehq, psnehprioq, psfehprioq, psanaq
    #print 'LSF status: \n%s \nqueue:%s, status:%s' % (output, queue, status)

    print 'CalibManager package revision: "%s"' % get_pkg_version()
     
    sys.exit ( "End of test" )

#----------------------------------
