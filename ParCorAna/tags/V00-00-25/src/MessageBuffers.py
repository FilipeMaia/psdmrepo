from mpi4py import MPI
import numpy as np

class SM_MsgBuffer(object):
    '''interface for buffer for server/master messages. 

    Provides interface so client does not need to know how implemented. 
    The implementation is 5 int32 with [msgtag, serverrank, seconds, nanoseconds, fiducials]
    '''
    SERVER_TO_MASTER_EVT = 1
    SERVER_TO_MASTER_END = 2
    MASTER_TO_SERVER_SEND_TO_WORKERS = 3
    MASTER_TO_SERVER_ABORT = 4
    IDX_MSGTAG = 0
    IDX_RANK = 1
    IDX_SEC = 2
    IDX_NSEC = 3
    IDX_FIDUCIALS = 4
    MPI_TYPE = MPI.INT32_T
    def __init__(self, msgtag=None, rank=None, sec=None, nsec=None, fiducials=None):
        self.msgbuffer = np.zeros(5, np.int32)
        if msgtag != None: self.msgbuffer[SM_MsgBuffer.IDX_MSGTAG]=np.int32(msgtag)
        if rank != None: self.msgbuffer[SM_MsgBuffer.IDX_RANK]=np.int32(rank)
        if sec != None: self.msgbuffer[SM_MsgBuffer.IDX_SEC]=np.int32(sec)
        if nsec != None: self.msgbuffer[SM_MsgBuffer.IDX_NSEC]=np.int32(nsec)
        if fiducials != None: self.msgbuffer[SM_MsgBuffer.IDX_FIDUCIALS]=np.int32(fiducials)

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

    def setEventId(self, sec, nsec, fiducials):
        self.msgbuffer[SM_MsgBuffer.IDX_SEC]=np.int32(sec)
        self.msgbuffer[SM_MsgBuffer.IDX_NSEC]=np.int32(nsec)
        self.msgbuffer[SM_MsgBuffer.IDX_FIDUCIALS]=np.int32(fiducials)

    def getEventId(self):
        return self.getSec(), self.getNsec(), self.getFiducials()

    def getSec(self):
        return int(self.msgbuffer[SM_MsgBuffer.IDX_SEC])

    def getNsec(self):
        return int(self.msgbuffer[SM_MsgBuffer.IDX_NSEC])

    def getFiducials(self):
        return int(self.msgbuffer[SM_MsgBuffer.IDX_FIDUCIALS])

class MWV_MPI_Type(object):
    '''returns MPI Type for MVW_MsgBuffer in master/viewer/worker communications.

    Upon initialization, creates a new MPI type that is [int32, int32, int32, int32, int32, int64] 
    and Commit's the type with the MPI library. Upon deleteion, calls Free 
    on the type.

    Use numpyDtype to get the equivalent numpy type, which is::
    
      msgtag  np.int32
      rank    np.int32
      sec     np.int32
      nsec    np.int32
      fidcuials np.int32
      counter np.int64 
    '''
 
   # an alternative to the dict below, is to use mpi4py.MPI. __TypeDict__ 
   # which is discussed on the mpi4py forum. This dict takes the np.dtype.char
   # as keys, i.e, it maps values like 'i', 'G', 'Zg' to MPI types.
    NumpyToMPI_TypeDict = {np.dtype(np.int32):MPI.INT32_T,
                           np.dtype(np.int64):MPI.INT64_T}
    def __init__(self):
        # define fields for numpy type to to be message buffer
        fields = [('msgtag',np.int32),
                  ('rank',np.int32),
                  ('sec',np.int32),
                  ('nsec',np.int32),
                  ('fiducials',np.int32),
                  ('counter',np.int64)]
        self.numpyDtype = np.dtype(fields)
        # derive MPI type from numpy description
        MPI_blocklens = (1,) * len(fields)
        MPI_displacements = []
        MPI_types = []
        for fld in fields:
            fldName = fld[0]
            npFldDtype, fldOffset = self.numpyDtype.fields[fldName]
            MPI_displacements.append(fldOffset)
            MPI_types.append(MWV_MPI_Type.NumpyToMPI_TypeDict[npFldDtype])
        MPI_displacements = tuple(MPI_displacements)
        MPI_types = tuple(MPI_types)
        MPI_dtype = MPI.Datatype.Create_struct(MPI_blocklens, MPI_displacements, MPI_types)
        assert MPI_dtype.extent >= self.numpyDtype.itemsize, \
            "extent of MPI type (%d) < numpy dtype (%d)" % (MPI_dtype.extent >= self.numpyDtype.itemsize)
        if MPI_dtype.extent != self.numpyDtype.itemsize:
            MPI_dtype = MPI_dtype.Create_resized(0, self.numpyDtype.itemsize)
        MPI_dtype = MPI_dtype.Commit()
        self.MPI_dtype = MPI_dtype

    def __del__(self):
        self.MPI_dtype.Free()

class MVW_MsgBuffer(object):
    '''message buffer for MasterViewerWorkers. 
    
    Attributes:
      msgtag:    message tag 
      rank:      server rank
      sec:       int32
      nsec:      int32
      fiducials: int32
      counter:   int64 120hz counter for event. Relative to some first event. Possible that it is negative.
    '''
    EVT = 10
    END = 20
    UPDATE = 30

    MPI_Type = MWV_MPI_Type()

    def __init__(self, msgtag=None, rank=None, sec=None, nsec=None, fiducials=None, counter=None):
        
        self.msgbuffer = np.zeros(1, dtype=MVW_MsgBuffer.MPI_Type.numpyDtype)
        if msgtag is not None:
            assert msgtag in [MVW_MsgBuffer.EVT, 
                              MVW_MsgBuffer.END,
                              MVW_MsgBuffer.UPDATE], "unknown message tag: %r" % msgtag
            self.msgbuffer[0]['msgtag'] = msgtag
        if rank is not None:
            self.msgbuffer[0]['rank']=rank
        if sec is not None:
            self.msgbuffer[0]['sec']=np.int32(sec)
        if nsec is not None:
            self.msgbuffer[0]['nsec']=np.int32(nsec)
        if fiducials is not None:
            self.msgbuffer[0]['fiducials']=np.int32(fiducials)
        if counter is not None:
            self.msgbuffer[0]['counter']=np.int64(counter)

    def getNumpyBuffer(self):
        return self.msgbuffer

    def getMPIType(self):
        return MVW_MsgBuffer.MPI_Type.MPI_dtype

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

    def setCounter(self, counter):
        self.msgbuffer[0]['counter'] = np.int64(counter)
        
    def getCounter(self):
        return int(self.msgbuffer[0]['counter'])

    def setSeconds(self, sec):
        self.msgbuffer[0]['sec'] = np.int32(sec)
        
    def getSeconds(self):
        return int(self.msgbuffer[0]['sec'])

    def setNanoSeconds(self, nsec):
        self.msgbuffer[0]['nsec'] = np.int32(nsec)
        
    def getNanoSeconds(self):
        return int(self.msgbuffer[0]['nsec'])

    def setFiducials(self, fiducials):
        self.msgbuffer[0]['fiducials'] = np.int32(fiducials)
        
    def getFiducials(self):
        return int(self.msgbuffer[0]['fiducials'])

    def setRank(self, rank):
        self.msgbuffer[0]['rank'] = np.int32(rank)
        
    def getRank(self):
        return int(self.msgbuffer[0]['rank'])

    def getTime(self):
        timeDict = {'sec':self.getSeconds(),
                    'nsec': self.getNanoSeconds(),
                    'fiducials': self.getFiducials(),
                    'counter': self.getCounter()
        }
        return timeDict

