#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module FileDeployer ...
#
#------------------------------------------------------------------------

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
import stat

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersForApp import cp
from   Logger               import logger
import GlobalUtils          as     gu

#-----------------------------

class FileDeployer :
    """Collection of methods for file deployment in calibration directory tree"""

    def __init__ ( self ) :
        pass

    
    def procDeployCommand(self, cmd):
        
        cmd_seq = cmd.split()
        msg = 'Command: ' + cmd

        path = cmd_seq[2]
        #print 'Destination directory: %s' % dir_des
        #/reg/neh/home/dubrovin/calib/CsPad::CalibV1/CxiDsd.0:Cspad.0/pedestals/132-end.data

        dir_ctype, fname      = path      .rsplit('/',1)
        dir_src,   calib_type = dir_ctype .rsplit('/',1)
        dir_dtype, src        = dir_src   .rsplit('/',1)
        dir_calib, dtype      = dir_dtype .rsplit('/',1)

        #print 'path, fname : ', path, fname
        #print 'path     : ', path
        #print 'fname    : ', fname
        #print 'dir_ctype: ', dir_ctype
        #print 'dir_src  : ', dir_src
        #print 'dir_dtype: ', dir_dtype
        #print 'dir_calib: ', dir_calib

        list_of_dirs = [dir_calib, dir_dtype, dir_src, dir_ctype]

        for dir in list_of_dirs :

            dir_exists = os.path.exists(dir) 
            #print dir, dir_exists

            if not dir_exists :
                gu.create_directory(dir)

        out, err = gu.subproc(cmd_seq)
        if err != '' : msg += '\nERROR: ' + err
        if out != '' : msg += '\nRESPONCE: ' + out
        logger.info(msg, __name__)

        self.changeFilePermissions(path)

        self.addHistoryRecord(cmd)



    def changeFilePermissions(self, path):
        msg = 'Change permissions for file: %s' % path
        logger.info(msg, __name__)
        
        #st = os.stat(path)
        #os.chmod(path, st.st_mode | stat.S_IEXEC)
        os.system('chmod 670 %s' % path)



    def addHistoryRecord(self, cmd):
        #print 'cmd  = ', cmd
        fname_history  = cp.fname_history.value()
        if fname_history == '' : return

        exp_name       = cp.exp_name.value()
        str_run_number = cp.str_run_number.value()

        user   = gu.get_enviroment(env='USER')
        login  = gu.get_enviroment(env='LOGNAME')
        host   = gu.get_enviroment(env='HOST')
        tstamp = gu.get_current_local_time_stamp(fmt='%Y-%m-%dT%H:%M:%S  zone:%Z')

        cmd_cp, path_inp, path_out = cmd.split() 
        dir_inp, fname_inp = path_inp.rsplit('/',1)
        dir_out, fname_out = path_out.rsplit('/',1) 
        path_history = os.path.join(dir_out,fname_history)

        rec = 'file:%s  copy_of:%s  exp:%s  run:%s  user:%s  host:%s  cptime:%s\n' % \
              (fname_out.ljust(14),
               fname_inp,
               exp_name.ljust(8),
               str_run_number.ljust(4),
               user,
               host,
               tstamp.ljust(29))

        #print 'user           = ', user
        #print 'login          = ', login
        #print 'host           = ', host
        #print 'fname_inp      = ', fname_inp
        #print 'fname_out      = ', fname_out
        #print 'dir_out        = ', dir_out
        #print 'tstamp         = ', tstamp
        #print 'exp_name       = ', exp_name
        #print 'str_run_number = ', str_run_number
        #print 'path_history   = ', path_history
        #print 'rec            = ', rec

        msg = 'Save record: \n%sin history file: %s' % (rec,path_history)
        logger.info(msg, __name__)

        gu.save_textfile(rec, path_history, mode='a')

        self.changeFilePermissions(path_history)

#-----------------------------

fd = FileDeployer()

#-----------------------------


        
