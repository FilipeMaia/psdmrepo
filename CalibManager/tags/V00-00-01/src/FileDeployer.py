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

#-----------------------------
# Imports for other modules --
#-----------------------------

#from ConfigParametersForApp import cp
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

#-----------------------------

fd = FileDeployer()

#-----------------------------


        
