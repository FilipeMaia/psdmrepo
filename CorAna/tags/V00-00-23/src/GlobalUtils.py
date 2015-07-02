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
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
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
from PyQt4 import QtGui, QtCore
from LogBook import message_poster

import PyCSPadImage.CSPAD2x2ImageUtils as cspad2x2img
import PyCSPadImage.CSPADImageUtils    as cspadimg

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
#----------------------------------        

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


def print_list_of_files_in_dir(dirname, path_or_fname) :
    dname, fname = os.path.split(path_or_fname)     # i.e. ('work_corana', 'img-xcs-r0015-b0000.bin')
    print 'print_list_of_files_in_dir():  directory:' + dirname + '  fname:' + fname

    for fname_in_dir in get_list_of_files_in_dir(dirname) :
        if fname in fname_in_dir :
            print fname_in_dir    
    print '\n'

#----------------------------------

def subproc(command_seq, env=None) : # for example, command_seq=['bsub', '-q', cp.batch_queue, '-o', 'log-ls.txt', 'ls -l']
    p = subprocess.Popen(command_seq, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) #, stdin=subprocess.STDIN
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
#    line_fields = line.split(' ')
#    if line_fields[0] != 'Job' :
#        sys.exit('EXIT: Unexpected response at batch submission: ' + line)
#    job_id_str = line_fields[1].strip('<').rstrip('>')
#    return job_id_str

def batch_job_submit(command, queue='psnehq', log_file='batch-log.txt') :

    if os.path.lexists(log_file) : remove_file(log_file)

    out, err = subproc(['bsub', '-q', queue, '-o', log_file, command])
    line_fields = out.split(' ')
    if line_fields[0] != 'Job' :
        sys.exit('EXIT: Unexpected response at batch submission: ' + line)
        job_id_str = 'JOB_ID_IS_UNKNOWN'
    else :
        job_id_str = line_fields[1].strip('<').rstrip('>')

    if err != '' : logger.warning( err, __name__) 
    logger.info(out, __name__) 

    return job_id_str, out, err


def batch_job_check(job_id_str, queue='psnehq') :
    out, err = subproc(['bjobs', '-q', queue, job_id_str])
    if err != '' : return err + '\n' + out
    else         : return out


def batch_job_kill(job_id_str) :
    command = ['kill', job_id_str]
    #print 'command:', command
    out, err = subproc(command)
    if err != '' : return err + '\n' + out
    else         : return out


def batch_job_status(job_id_str, queue='psnehq') :
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
# assumes: path = .../<inst>/<experiment>/xtc/<file-name>.xtc
#        i.e         /reg/data/ana12/xcs/xcsi0112/xtc/e167-r0020-s00-c00.xtc 
#        or          /reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc
def parse_xtc_path(path='.') :

    logger.debug( 'parse_xtc_path(...): ' + path, __name__)

    instrument = 'INS'
    experiment = 'expt'
    run_str    = 'r0000'
    run_num    = 0

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
#----------------------------------

def get_array_from_file(fname, dtype=np.float32) :
    if fname is None or fname == '' :
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
        arr.shape = (185,388,2) 
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

def save_textfile(text, path, mode='w') :
    """Saves text in file specified by path. mode: 'w'-write, 'a'-append 
    """
    f=open(path,mode)
    f.write(text)
    f.close() 

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

def get_save_fname_through_dialog_box(parent, path0, dial_title, filter='*.txt'):       

    path = str( QtGui.QFileDialog.getSaveFileName(parent,
                                                  caption   = dial_title,
                                                  directory = path0,
                                                  filter    = filter
                                                  ) )
    if path == '' :
        logger.debug('Saving is cancelled.', __name__)
        return None
    logger.info('Output file: ' + path, __name__)
    return path

#----------------------------------

def get_open_fname_through_dialog_box(parent, path0, dial_title, filter='*.txt'):       

    path = str( QtGui.QFileDialog.getOpenFileName(parent, dial_title, path0, filter=filter) )
    dname, fname = os.path.split(path)
    if dname == '' or fname == '' :
        logger.info('Input directiry name or file name is empty... keep file path unchanged...')
        return None
    logger.info('Input file: ' + path, __name__)
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
        style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        mesbox.setStyleSheet (style)

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

def help_dialog_box(parent=None, text='Help message goes here', title='Help') :
        """Pop-up NON-MODAL box for help etc."""

        mesbox = QtGui.QMessageBox(parent, windowTitle=title,
                                           text=text,
                                           standardButtons=QtGui.QMessageBox.Close)
        style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        mesbox.setStyleSheet (style)
        mesbox.setWindowModality (QtCore.Qt.NonModal)
        mesbox.setModal (False)
        #clicked = mesbox.exec_() # For MODAL dialog
        clicked = mesbox.show()  # For NON-MODAL dialog
        logger.info('Help window is open' + text, 'help_dialog_box')
        return

#----------------------------------

def arr_rot_n90(arr, rot_ang_n90=0) :
    if   rot_ang_n90==  0 : return arr
    elif rot_ang_n90== 90 : return np.flipud(arr.T)
    elif rot_ang_n90==180 : return np.flipud(np.fliplr(arr))
    elif rot_ang_n90==270 : return np.fliplr(arr.T)
    else                  : return arr


#----------------------------------

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

    print 'xtc_fname_for_all_chunks(...): ', xtc_fname_for_all_chunks('e168-r0016-s00-c00.xtc')
    print 'xtc_fname_for_all_chunks(...): ', xtc_fname_for_all_chunks('/reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc')

    sys.exit ( "End of test" )

#----------------------------------
