#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module IrodsClient...
#
#------------------------------------------------------------------------

""" iRODS client library.

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

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import irods
from irods_error import *

#----------------------------------
# Local non-exported definitions --
#----------------------------------

def _getSqlResultByInx ( genQueryOut, inx ):
    data = irods.getSqlResultByInx(genQueryOut, inx)
    if data : return data.getValues()
    return None

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class IrodsClient ( object ) :

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, conn ) :
        
        self._conn = conn

    #-------------------
    #  Public methods --
    #-------------------

    def resources(self):
        """ Return the list of resources """
        
        conn = self._conn.connection()
        if not conn : return None
        
        genQueryInp = irods.genQueryInp_t()   
        
        i1 = irods.inxIvalPair_t()
        i1.addInxIval(irods.COL_R_RESC_NAME, 0)
        genQueryInp.setSelectInp(i1)
        
        genQueryInp.setMaxRows(500)
        genQueryInp.setContinueInx(0)
        
        res = []
        while True:

            genQueryOut, status = irods.rcGenQuery(conn, genQueryInp)
            if status == 0 :
                res += self._result2tuples( genQueryOut )
            if genQueryOut.getContinueInx() > 0:
                genQueryInp.setContinueInx(genQueryOut.getContinueInx())
            else :
                break

        return [ x[0] for x in res ]

    def resource(self,name):
        """ Return the description of single resource """
        
        conn = self._conn.connection()
        if not conn : return None
        
        genQueryInp = irods.genQueryInp_t()   
        
        i1 = irods.inxIvalPair_t()
        i1.addInxIval(irods.COL_R_RESC_NAME, 0)
        i1.addInxIval(irods.COL_R_RESC_ID, 0)
        i1.addInxIval(irods.COL_R_ZONE_NAME, 0)
        i1.addInxIval(irods.COL_R_TYPE_NAME, 0)
        i1.addInxIval(irods.COL_R_CLASS_NAME, 0)
        i1.addInxIval(irods.COL_R_LOC, 0)
        i1.addInxIval(irods.COL_R_VAULT_PATH, 0)
        i1.addInxIval(irods.COL_R_FREE_SPACE, 0)
        i1.addInxIval(irods.COL_R_RESC_INFO, 0)
        i1.addInxIval(irods.COL_R_RESC_COMMENT, 0)
        i1.addInxIval(irods.COL_R_CREATE_TIME, 0)
        i1.addInxIval(irods.COL_R_MODIFY_TIME, 0)
        genQueryInp.setSelectInp(i1)
        
        if name:
            i2 = irods.inxValPair_t()
            i2.addInxVal(irods.COL_R_RESC_NAME, "='%s'" % name.encode('utf-8'))
            genQueryInp.setSqlCondInp(i2)
            
        genQueryInp.setMaxRows(100)
        genQueryInp.setContinueInx(0)

        columns = ["name", "resc_id", "zone", "type", "class", "location",
                   "vault", "free_space", "info", "comment", "ctime", "mtime"]
        
        res = []
        while True:

            genQueryOut, status = irods.rcGenQuery(conn, genQueryInp)
            if status == 0 :
                res += self._result2dict( genQueryOut, columns )
            if status == 0 and genQueryOut.getContinueInx() > 0:
                genQueryInp.setContinueInx(genQueryOut.getContinueInx())
            else :
                break

        return res


    def files (self, path, recursive=None ):
        """ Gets the list of files/collections and returns info about every file """

        conn = self._conn.connection()
        if not conn : return None

        rodsPath = irods.rodsPath_t()
        rodsPath.setInPath( path )
        rodsPath.setOutPath( path )
        
        status = rodsPath.getRodsObjType(conn)
        if rodsPath.getObjState() == irods.NOT_EXIST_ST:
            return None
        
        if rodsPath.getObjType() == irods.DATA_OBJ_T:
            res = self._file ( conn, rodsPath.getOutPath() )
        elif rodsPath.getObjType() ==  irods.COLL_OBJ_T:
            res = self._filesInColl ( conn, rodsPath.getOutPath(), recursive )
        else :
            res = None
        
        return res

    #--------------------
    #  Private methods --
    #--------------------

    def _file (self, conn, srcPath):
        """ List all entries for a given file path """

        queryFlags = irods.VERY_LONG_METADATA_FG
        genQueryInp = irods.genQueryInp_t()
        genQueryInp.setQueryInpForData(queryFlags)
        genQueryInp.setMaxRows(irods.MAX_SQL_ROWS)
    
        myColl, myData, status = irods.splitPathByKey(srcPath, '/')
        if status < 0: return None

        condStr = "='%s'" % myColl.encode('utf-8')
        genQueryInp.getSqlCondInp().addInxVal(irods.COL_COLL_NAME, condStr)
        condStr = "='%s'" % myData.encode('utf-8')
        genQueryInp.getSqlCondInp().addInxVal(irods.COL_DATA_NAME, condStr)
        
        genQueryOut, status = irods.rcGenQuery(conn, genQueryInp)        
        if status < 0: return None

        res = [ self._objectEntry(x) for x in self._qo2oe(genQueryOut) ]
        return res

    def _qo2oe(self, genQueryOut):
        """ convert genQueryOut to sequence of collEnt_t """

        chksumStr = _getSqlResultByInx(genQueryOut, irods.COL_D_DATA_CHECKSUM)
        dataPath = _getSqlResultByInx(genQueryOut, irods.COL_D_DATA_PATH)
        dataId = _getSqlResultByInx(genQueryOut, irods.COL_D_DATA_ID)
        dataName = _getSqlResultByInx(genQueryOut, irods.COL_DATA_NAME)
        replNum = _getSqlResultByInx(genQueryOut, irods.COL_DATA_REPL_NUM)
        dataSize = _getSqlResultByInx(genQueryOut, irods.COL_DATA_SIZE)
        dataMode = _getSqlResultByInx(genQueryOut, irods.COL_DATA_MODE)
        rescName = _getSqlResultByInx(genQueryOut, irods.COL_D_RESC_NAME)
        collName = _getSqlResultByInx(genQueryOut, irods.COL_COLL_NAME)
        replStatus = _getSqlResultByInx(genQueryOut, irods.COL_D_REPL_STATUS)
        dataModify = _getSqlResultByInx(genQueryOut, irods.COL_D_MODIFY_TIME)
        dataCreate = _getSqlResultByInx(genQueryOut, irods.COL_D_CREATE_TIME )
        dataOwnerName = _getSqlResultByInx(genQueryOut, irods.COL_D_OWNER_NAME)

        for i in xrange(genQueryOut.getRowCnt()):

            entry = irods.collEnt_t()
            entry.setObjType(irods.DATA_OBJ_T)
            if chksumStr : entry.setChksum( chksumStr[i] )
            if dataCreate : entry.setCreateTime( dataCreate[i] )
            if dataPath : entry.setPhyPath( dataPath[i] )
            if dataId : entry.setDataId( dataId[i] )
            if dataMode : entry.setDataMode( int(dataMode[i]) )
            if dataName : entry.setDataName( dataName[i] )
            if replNum : entry.setReplNum( int(replNum[i]) )
            if dataSize : entry.setDataSize( int(dataSize[i]) )
            if collName : entry.setCollName( collName[i] )
            if rescName : entry.setResource( rescName[i] )
            if replStatus : entry.setReplStatus( int(replStatus[i]) )
            if dataModify : entry.setModifyTime( dataModify[i] )
            if dataOwnerName : entry.setOwnerName( dataOwnerName[i] )
            
            yield entry

        
    def _filesInColl (self, conn, srcColl, recursive=None):
        """ return the list of the entries in collection, optionally recurse
        into the sub-collections """

        queryFlags = irods.DATA_QUERY_FIRST_FG | irods.VERY_LONG_METADATA_FG | irods.NO_TRIM_REPL_FG

        collHandle, status = irods.rclOpenCollection(conn, srcColl, queryFlags)
        if status < 0: return None
        
        res = []
        collEntry = irods.collEnt_t()
        status = irods.rclReadCollection( conn, collHandle, collEntry )
        while status >= 0 :
            
            if collEntry.getObjType() == irods.DATA_OBJ_T :
                res.append( self._objectEntry(collEntry) )
            else :
                res.append( self._collEntry(collEntry) )
                if recursive :
                    subres = self._filesInColl( conn, collEntry.getCollName(), recursive )
                    if subres : res += subres
            
            status = irods.rclReadCollection( conn, collHandle, collEntry )
            
        irods.rclCloseCollection(collHandle)
            
        return res

    def _objectEntry(self, entry):
        """ build the result dictionary from the collEnt_t object for object type """
        
        v = dict( type     = 'object',
                  name     = entry.getDataName(),
                  size     = entry.getDataSize(),
                  ctime    = entry.getCreateTime(),
                  mtime    = entry.getModifyTime(),
                  owner    = entry.getOwnerName(),
                  checksum = entry.getChksum(),
                  collName = entry.getCollName(),
                  id       = entry.getDataId(),
                  datamode = entry.getDataMode(),
                  path     = entry.getPhyPath(),
                  replica  = entry.getReplNum(),
                  replStat = entry.getReplStatus(),
                  resource = entry.getResource() )
        return v
        

    def _collEntry(self, entry):
        """ build the result dictionary from the collEnt_t object for collection type """
        
        v = dict( type     = 'collection',
                  name     = entry.getCollName() )
        specColl = entry.getSpecColl()
        if specColl.getCollClass() == irods.MOUNTED_COLL :
            specCollDict = dict( type      = specColl.getSpecCollTypeStr(),
                                 collClass = specColl.getCollClass(),
                                 objPath   = specColl.getObjPath(),
                                 phyPath   = specColl.getPhyPath(),
                                 resource  = specColl.getResource() )
            v['specColl'] = specCollDict
        elif specColl.getCollClass() != irods.NO_SPEC_COLL :
            specCollDict = dict( type      = specColl.getSpecCollTypeStr(),
                                 collClass = specColl.getCollClass(),
                                 objPath   = specColl.getObjPath(),
                                 cacheDir  = specColl.getCacheDir(),
                                 cacheDirty = specColl.getCacheDirty(),
                                 resource  = specColl.getResource() )
            v['specColl'] = specCollDict
        return v
        

    def _result2tuples (self, genQueryOut ):
        """ Reformat the result in genQueryOut. The output will be the list of tuples, 
        one tuple per result row. """
        
        sqlres = genQueryOut.getSqlResult()
        
        cols = genQueryOut.getAttriCnt()
        rows = genQueryOut.getRowCnt()
        for j in range(rows) :
            t = tuple ( [ sqlres[i].getValues()[j] for i in range(cols) ] ) 
            yield t
            
    def _result2dict (self, genQueryOut, columns=None ):
        """ Reformat the result in genQueryOut. Output will be the list of 
        dictionaries, one dictionary per row, with the keys from the column list
        """
        
        sqlres = genQueryOut.getSqlResult()

        cols = genQueryOut.getAttriCnt()
        rows = genQueryOut.getRowCnt()
        
        for i in range(rows) :
            res = {}
            for col, sql in zip(columns,sqlres) :
                res[col] = sql.getValues()[i]
            yield res

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
