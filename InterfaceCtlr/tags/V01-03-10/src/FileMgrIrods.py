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
    """
    File management interface implemented on top of the iRODS i-commands.
    """
    
    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, cmd, config, logger ) :
        """
        FileMgrIrods(cmd: str, config: object, logger: object)

        Constructor takes a list of arguments:
            - cmd: command to use, one of "iput" or "ireg"
            - config: configuration object of type Config (from this package) or dictionary 
            - logger: instance of logger class from logging package (or logging module itself)
        """

        self._cmd = cmd
        self._log = logger
        
        # update our environment
        for env in ['irodsHost','irodsDefResource','irodsPort','irodsZone','irodsUserName','irodsAuthFileName'] :
            val = config.get(env)
            if val:
                os.putenv( env, str(val) )

    #-------------------
    #  Public methods --
    #-------------------

    def storeFile ( self, src_path, dst_path, resc=None ) :
        """
        self.storeFile(src_path: str, dst_path: str) -> int
        
        Store single file, takes the name of file on disk and the name of
        file in iRODS. Returns 0 for success, non-zero on failure, code 
        should be interpreted as a result of os.spawnvp() call.
        By defaut the file is stored in the irods resource from the env 
        variable irodsDefResource. It is set in the fm-irods config section. 
        It can be overwritten with the resc parameter.
        """
        self._log.info ( "FileMgrIrods.storeFile: %s -> %s", src_path, dst_path )
        
        # create collection if needed
        coll = dst_path[:dst_path.rfind('/')]
        cmd = [ 'imkdir', '-p', coll ]
        self._log.debug ( "FileMgrIrods.storeFile: creating collection %s", coll )
        returncode = os.spawnvp(os.P_WAIT,cmd[0],cmd)
        if returncode : self._log.warning("imkdir completed with status %d", returncode)
        # this call may fail, but it fails also if directory exists, so just try next thing
        
        resc_opt = '-R ' + resc if resc else ""

        if self._cmd == 'iput' :
            cmd = "iput %s %s %s" % (resc_opt, src_path, dst_path)
        else:
            cmd = "ireg %s %s %s" % (resc_opt, src_path, dst_path)
        cmd = cmd.split()
        
        returncode = os.spawnvp(os.P_WAIT,cmd[0],cmd)
        if returncode : self._log.warning("%s completed with status %d", self._cmd, returncode)
        return returncode
        
    def storeDir ( self, src_dir, dst_coll ) :
        """
        self.storeDir(src_dir: str, dst_coll: str) -> int
        
        Store all files in a directory, takes the name of directory on disk 
        and the name of iRODS collection. Returns 0 for success, non-zero on 
        failure, code should be interpreted as a result of os.spawnvp() call.
        """

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
        """
        self.listdir(coll: str) -> list of strings
        
        Returns the list of files in the collection.
        """
        
        cmd = ['ils', coll]
        child = popen2.Popen3(cmd)
        child.tochild.close()
        list = child.fromchild.read()
        
        stat = child.wait()
        if stat :
            # error happened, could mean either that iRODS server is not accessible 
            # (or other server troubles) in which case we want to stop, or that collection
            # does not exist, then we return empty list.

            # to check that server is running we run 'ips' command to ping it
            cmd = ['ips']
            child = popen2.Popen3(cmd)
            child.tochild.close()
            list = child.fromchild.read()
            if child.wait():
                # connection failed
                raise IOError("Failed to connect to iRODS SERVER")
            
            return []

        return list.split()

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
