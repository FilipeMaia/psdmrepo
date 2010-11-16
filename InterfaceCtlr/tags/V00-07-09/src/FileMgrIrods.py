#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module FileMgrIrods...
#
#------------------------------------------------------------------------

""" Interface to iRODS file manager 

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
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
import popen2

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class FileMgrIrods ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, cmd, config, logger ) :
        """Constructor.

        @param cmd   command to use, one of iput or ireg
        """

        self._cmd = cmd
        self._log = logger
        
        # update our environment
        for env in ['irodsHost','irodsDefResource','irodsPort','irodsZone','irodsUserName','irodsAuthFileName'] :
            val = config.get(env)
            if val:
                os.putenv( env, val )

    #-------------------
    #  Public methods --
    #-------------------

    def storeFile ( self, src_path, dst_path ) :
        """ Store single file """
        self._log.info ( "FileMgrIrods.storeFile: %s -> %s", src_path, dst_path )
        
        # create collection if needed
        coll = dst_path[:dst_path.rfind('/')]
        cmd = [ 'imkdir', '-p', coll ]
        self._log.debug ( "FileMgrIrods.storeFile: creating collection %s", coll )
        returncode = os.spawnvp(os.P_WAIT,cmd[0],cmd)
        if returncode : self._log.warning("imkdir completed with status %d", returncode)
        # this call may fail, but it fails also if directory exists, so just try next thing
        
        if self._cmd == 'iput' :
            cmd = [ 'iput', src_path, dst_path ]
        else :
            cmd = [ 'ireg', src_path, dst_path ]
        
        returncode = os.spawnvp(os.P_WAIT,cmd[0],cmd)
        if returncode : self._log.warning("%s completed with status %d", self._cmd, returncode)
        return returncode
        
    def storeDir ( self, src_dir, dst_coll ) :
        """ Store all files in a directory """

        # create collection if needed
#        self._log.info ( "FileMgrIrods.storeFile: creating collection %s", dst_coll )
#        cmd = [ 'imkdir', '-p', dst_coll ]
#        returncode = os.spawnvp(os.P_WAIT,cmd[0],cmd)
        # this call may fail, but it fails also if directory exists, so just try next thing
        
        self._log.info ( "FileMgrIrods.storeDir: %s -> %s", src_dir, dst_coll )
        if self._cmd == 'iput' :
            cmd = [ 'iput', src_dir, dst_coll ]
        else :
            cmd = [ 'ireg', '-C', src_dir, dst_coll ]
        returncode = os.spawnvp(os.P_WAIT,cmd[0],cmd)
        if returncode : self._log.warning("%s completed with status %d", self._cmd, returncode)
        return returncode
        
    def listdir(self, coll):
        """ List files in the collection """
        
        cmd = ['ils', coll]
        child = popen2.Popen3(cmd)
        child.tochild.close()
        list = child.fromchild.read()
        
        stat = child.wait()
        if stat :
            # error happened
            self._log.error( "FileMgrIrods.listdir: error returned from ils: %s", str(stat) )
            return None

        return list.split()

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
