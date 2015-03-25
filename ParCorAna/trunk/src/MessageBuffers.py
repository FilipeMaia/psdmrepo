from mpi4py import MPI
import numpy as np

class SM_MsgBuffer(object):
    '''interface for buffer for server/master messages. 

    Provides interface so client does not need to know how implemented. 
    The implementation is 4 int32 with [msgtag, serverrank, seconds, nanoseconds]
    '''
    SERVER_TO_MASTER_EVT = 1
    SERVER_TO_MASTER_END = 2
    MASTER_TO_SERVER_SEND_TO_WORKERS = 3
    MASTER_TO_SERVER_ABORT = 4
    IDX_MSGTAG = 0
    IDX_RANK = 1
    IDX_SEC = 2
    IDX_NSEC = 3
    MPI_TYPE = MPI.INT32_T
    def __init__(self, msgtag=None, rank=None, sec=None, nsec=None):
        self.msgbuffer = np.zeros(4, np.int32)
        if msgtag != None: self.msgbuffer[SM_MsgBuffer.IDX_MSGTAG]=np.int32(msgtag)
        if rank != None: self.msgbuffer[SM_MsgBuffer.IDX_RANK]=np.int32(rank)
        if sec != None: self.msgbuffer[SM_MsgBuffer.IDX_SEC]=np.int32(sec)
        if nsec != None: self.msgbuffer[SM_MsgBuffer.IDX_NSEC]=np.int32(nsec)

    def getNumpyBuffer(self):
        return self.msgbuffer

    def getMPIType(self):
        return SM_MsgBuffer.MPI_TYPE

    def getRank(self):
        return int(self.msgbuffer[SM_MsgBuffer.IDX_RANK])

    def setRank(self, rank):
        self.msgbuffer[SM_MsgBuffer.IDX_RANK] = np.int32(rank)

    def isEvt(self):
        return self.msgbuffer[SM_MsgBuffer.IDX_MSGTAG] == \
            np.int32(SM_MsgBuffer.SERVER_TO_MASTER_EVT)

    def setEvt(self):
        self.msgbuffer[SM_MsgBuffer.IDX_MSGTAG] = \
            np.int32(SM_MsgBuffer.SERVER_TO_MASTER_EVT)

    def isEnd(self):
        return self.msgbuffer[SM_MsgBuffer.IDX_MSGTAG] == \
            np.int32(SM_MsgBuffer.SERVER_TO_MASTER_END)

    def setEnd(self):
        self.msgbuffer[SM_MsgBuffer.IDX_MSGTAG] = \
            np.int32(SM_MsgBuffer.SERVER_TO_MASTER_END)

    def isSendToWorkers(self):
        return self.msgbuffer[SM_MsgBuffer.IDX_MSGTAG] == \
            np.int32(SM_MsgBuffer.MASTER_TO_SERVER_SEND_TO_WORKERS)

    def setSendToWorkers(self):
        self.msgbuffer[SM_MsgBuffer.IDX_MSGTAG] = \
            np.int32(SM_MsgBuffer.MASTER_TO_SERVER_SEND_TO_WORKERS)

    def isAbort(self):
        return self.msgbuffer[SM_MsgBuffer.IDX_MSGTAG] == \
            np.int32(SM_MsgBuffer.MASTER_TO_SERVER_ABORT)

    def setAbort(self):
        self.msgbuffer[SM_MsgBuffer.IDX_MSGTAG] = \
                        np.int32(SM_MsgBuffer.SERVER_TO_MASTER_EVT)

    def setTime(self, sec, nsec):
        self.msgbuffer[SM_MsgBuffer.IDX_SEC]=np.int32(sec)
        self.msgbuffer[SM_MsgBuffer.IDX_NSEC]=np.int32(nsec)

    def getTime(self):
        return int(self.msgbuffer[SM_MsgBuffer.IDX_SEC]), \
            int(self.msgbuffer[SM_MsgBuffer.IDX_NSEC])

    def getSec(self):
        return int(self.msgbuffer[SM_MsgBuffer.IDX_SEC])

    def getNsec(self):
        return int(self.msgbuffer[SM_MsgBuffer.IDX_NSEC])

class MWV_MPI_Type(object):
    '''returns MPI Type for MVW_MsgBuffer. 

    Upon initialization, creates a new MPI type that is [int32, int32, float] 
    and Commit's the type with the MPI library. Upon deleteion, calls Free 
    on the type.

    Use numpyDtype to get the equivalent numpy type.
    '''
    def __init__(self):
        blocklens = (1,1,1)
        displacements = (0,4,8)
        types = (MPI.INT32_T, MPI.INT32_T, MPI.DOUBLE)
        dtype = MPI.Datatype.Create_struct(blocklens, displacements, types)
        dtype.Commit()
        self.dtype = dtype

    def __del__(self):
        self.dtype.Free()

    def numpyDtype(self):
        return np.dtype([('msgtag',np.int32),
                         ('rank',np.int32),
                         ('relsec',np.float)])

        
class MVW_MsgBuffer(object):
    '''message buffer for MasterViewerWorkers. 
    
    Attributes:
      msgtag: message tag 
      rank:   server rank
      relsec: floating point seconds relative to some start
    '''
    EVT = 10
    END = 20
    UPDATE = 30

    MPI_Type = MWV_MPI_Type()

    def __init__(self, msgtag=None, rank=None, relsec=None):
        
        self.msgbuffer = np.zeros(1, dtype=MVW_MsgBuffer.MPI_Type.numpyDtype())
        if msgtag is not None:
            assert msgtag in [MVW_MsgBuffer.EVT, MVW_MsgBuffer.END,
                              MVW_MsgBuffer.UPDATE], "unknown message tag: %r" % msgtag
            self.msgbuffer[0]['msgtag'] = msgtag
        if rank is not None:
            self.msgbuffer[0]['rank']=rank
        if relsec is not None:
            self.msgbuffer[0]['relsec']=np.float(relsec)

    def getNumpyBuffer(self):
        return self.msgbuffer

    def getMPIType(self):
        return MVW_MsgBuffer.MPI_Type.dtype

    def isEvt(self):
        return self.msgbuffer[0]['msgtag'] == MVW_MsgBuffer.EVT

    def isEnd(self):
        return self.msgbuffer[0]['msgtag'] == MVW_MsgBuffer.END

    def isUpdate(self):
        return self.msgbuffer[0]['msgtag'] == MVW_MsgBuffer.UPDATE

    def setEvt(self):
        self.msgbuffer[0]['msgtag'] = MVW_MsgBuffer.EVT

    def setEnd(self):
        self.msgbuffer[0]['msgtag'] = MVW_MsgBuffer.END

    def setUpdate(self):
        self.msgbuffer[0]['msgtag'] = MVW_MsgBuffer.UPDATE

    def setRelSec(self, relsec):
        self.msgbuffer[0]['relsec'] = relsec
        
    def getRelSec(self):
        return float(self.msgbuffer[0]['relsec'])

    def setRank(self, rank):
        self.msgbuffer[0]['rank'] = np.int32(rank)
        
    def getRank(self):
        return int(self.msgbuffer[0]['rank'])
