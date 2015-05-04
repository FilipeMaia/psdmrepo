#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module FileMgrRegister...
#
#------------------------------------------------------------------------

"""Module which facilitates registration of the files in file manager (iRODS).

This is wrapper for iRODS client commands which provides a transparent
way for registering files in iRODS. It uses Interface Controller 
database to access configuration information and various options
describing storage layout and credentials.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andy Salnikov
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
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from DbTools.DbConnection import DbConnection
from InterfaceCtlr.InterfaceDb import InterfaceDb
from InterfaceCtlr.FileMgrIrods import FileMgrIrods

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# default connection string
_conn_str = "file:/reg/g/psdm/psdatmgr/ic/.icdb-conn"

class _ConfigError(Exception):
    """
    Exception class for errors in this module.
    """
    pass

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class FileMgrRegister ( object ) :
    """
    Wrapper class for iRODS ireg command or corresponding PyRods utility.
    """

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, conn_str=None, config=None ) :
        """
        Constructor takes two optional parameters:
          - connection string for interface controller database (default is 
            "file:/reg/g/psdm/psdatmgr/ic/.icdb-conn")
          - configuration section list (default is ["fm-irods"])
          
        If database access fails or configration parameters are not found
        in the database an exception will be raised.
        """

        if not conn_str: conn_str = _conn_str
        if not config: config = ['fm-irods']

        # setup database
        conn = DbConnection(conn_string=conn_str)
        db = InterfaceDb(conn, self)

        # get the options from database
        self._config = db.read_config(config, False)

        # instantiate file manager
        cmd = self._config.get('filemanager-irods-command')
        if not cmd: raise _ConfigError("filemanager-irods-command option is not defined")
        self._file_mgr = FileMgrIrods(cmd, self._config, logging)

    #-------------------
    #  Public methods --
    #-------------------

    def register(self, path, instrument, experiment, type):
        """
        self.register(path: str, instrument: str, experiment: str, type: str)
        
        Register single file in irods. Takes following parameters:
        - path: full path name of the file to be registered
        - instrument: instrument name
        - experiment: experiment nae
        - type: file type name which is also a directory name in experiment data directory
          For type = 'xtc.idx' the file will be registered in irodsIdxResource (from config db).

        If file cannot be registered an exception is raised.           
        """

        # build object name in irods
        parm = dict(instrument=instrument, instrument_lower=instrument.lower(), experiment=experiment)
        dst_coll = self._config.get('filemanager-experiment-dir', subs=parm)
        if not dst_coll: raise _ConfigError("filemanager-experiment-dir option is not defined")

        # make destination path
        
        if type == 'xtc.idx':
            irods_resource = self._config.get('irodsIdxResource')
            data_subdir = 'xtc/index'
        elif type == 'smd.xtc':
            irods_resource = None
            data_subdir = 'xtc/smalldata'            
        else:
            irods_resource = None
            data_subdir = type

        basename = os.path.basename(path)
        dst_path = "/".join([dst_coll, data_subdir, basename])

        # register
        self._file_mgr.storeFile(path, dst_path, resc=irods_resource)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
