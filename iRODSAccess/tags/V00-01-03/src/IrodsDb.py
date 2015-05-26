#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module IrodsDb...
#
#------------------------------------------------------------------------

"""Interface to iRODS database.

Provides direct access to iRODS database for things like checksum
update.

This software was developed for the SIT project.  If you use all or 
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
import os.path

#---------------------------------
#  Imports of base class module --
#---------------------------------


#-----------------------------
# Imports for other modules --
#-----------------------------
from DbTools.DbConnection import DbConnection


#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class IrodsDb ( object ) :
    """interface to iRODS database."""

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, connStr = "file:/reg/g/psdm/psdatmgr/irods/.irodsdb-conn" ) :
        """Constructor.

        @param connStr   connection string for irods database
        """

        self._conn = DbConnection(conn_string=connStr)

    #-------------------
    #  Public methods --
    #-------------------

    def updateChecksums(self, checksums):
        """Stores checksums for the files in iRODS. Takes a list of tuples,
        first element of a tuple is full path name of the file in iRODS,
        second is the string with MD5 checksum. Returns the list of tuples,
        first element is iRODS path name, second element is a number of 
        the replicas which have been updated.
        
        Example of usage:
        
        checksums = [ 
            ('/psdm-zone/psdm/XPP/xppcom10/hdf5/xppcom10-r0001.h5', '4aff9050964ee62ac1b6f9d44dbebc1f'),
            ('/psdm-zone/psdm/XPP/xppcom10/hdf5/xppcom10-r0002.h5', 'ea812d79027e42cce36d35221f0c4dd2'),
            ]
        idb = IrodsDb()
        res = idb.updateChecksums( checksums )
        for path, count in res :
            print "%d replicas updated for %s" % (count, path)
        
        """

        res = []

        cursor = self._conn.cursor();
        for path, checksum in checksums:
            
            coll, obj = os.path.split(path)
            
            q = """UPDATE r_coll_main c, r_data_main d 
                SET d.data_checksum=%s
                WHERE c.coll_name=%s AND c.coll_id=d.coll_id AND d.data_name=%s"""
            cursor.execute( q, (checksum, coll, obj) )
            count = cursor.rowcount

            res.append( (path, count) )

        cursor.execute("commit")

        return res


    def updateAtime(self, atimes, resc='lustre-resc'):
        """Stores atime for files in iRODS. Takes a list of tuples,
        first element of a tuple is full path name of the file in iRODS,
        second is the atime as an integer. Returns the list of tuples,
        first element is iRODS path name, second element is a number of 
        the replicas which have been updated.
        
        Example of usage:
        
        checksums = [ 
            ('/psdm-zone/psdm/XPP/xppcom10/hdf5/xppcom10-r0001.h5', 1234567899),
            ('/psdm-zone/psdm/XPP/xppcom10/hdf5/xppcom10-r0002.h5', 1323232333),
            ]
        idb = IrodsDb()
        res = idb.updateAtime( checksums )
        for path, count in res :
            print "%d replicas updated for %s" % (count, path)        
        """

        res = []

        cursor = self._conn.cursor();
        for path, atime in atimes:
            
            atimeStr = "%011d" % atime
            coll, obj = os.path.split(path)
            #print coll, obj, atimeStr
            
            q = """UPDATE r_coll_main c, r_data_main d 
                SET d.data_expiry_ts=%s
                WHERE c.coll_name=%s AND c.coll_id=d.coll_id AND d.data_name=%s AND resc_name=%s"""
            cursor.execute( q, (atimeStr, coll, obj, resc) )
            count = cursor.rowcount

            res.append( (path, count) )

        cursor.execute("commit")

        return res

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
