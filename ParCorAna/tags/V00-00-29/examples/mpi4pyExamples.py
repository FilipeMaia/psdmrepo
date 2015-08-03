import numpy as np
from mpi4py import MPI
import time

def collectiveCommunicationInSubGroup():
    assert MPI.COMM_WORLD.Get_size()>=4, "need at least 4 ranks for this example"
    serverRankA = 0
    serverRankB = 1
    workerRanks = [2,3]
    worldGroup = MPI.COMM_WORLD.Get_group()
    serverRankBandWorkersGroup = worldGroup.Excl([serverRankA])
    serverRankBandWorkersComm = MPI.COMM_WORLD.Create(serverRankBandWorkersGroup)
    serverBrankInNewComm = MPI.Group.Translate_ranks(worldGroup, [serverRankB], serverRankBandWorkersGroup)[0]
    worldRank = MPI.COMM_WORLD.Get_rank()
    counts = (0,6,5)
    offsets = (0,0,6)
    if worldRank == serverRankB:
        dataToScatter = np.array(range(11), dtype=np.float32)
        recvBuffer = np.zeros(0,np.float32)
        serverRankBandWorkersComm.Scatterv([dataToScatter, counts, offsets, MPI.FLOAT], 
                                           recvBuffer, root = serverBrankInNewComm)
    elif worldRank in workerRanks:
        if worldRank == 2:
            recvBuffer = np.zeros(6,np.float32)
        elif worldRank == 3:
            recvBuffer = np.zeros(5,np.float32)
        serverRankBandWorkersComm.Scatterv([None, counts, offsets, MPI.FLOAT],
                                           recvBuffer, root = serverBrankInNewComm)
        print "worker with world rank=%d received: %r" % (worldRank, recvBuffer)
        
        
def MPIType(verbose=True):
    assert MPI.COMM_WORLD.Get_size()>=2, "need at least two ranks for this example"
    fields = [('msgtag',np.int32),
              ('counter',np.int64),
              ('energy',np.float32)]

    numpyDataType = np.dtype(fields)

    MPI_blocklens = (1,1,1)
    MPI_types = (MPI.INT32_T, MPI.INT64_T, MPI.FLOAT)
    fldNames = [fld[0] for fld in fields]
    MPI_displacements = tuple([numpyDataType.fields[nm][1] for nm in fldNames])
    MPIDataType = MPI.Datatype.Create_struct(MPI_blocklens, 
                                             MPI_displacements,
                                             MPI_types)
    assert MPIDataType.extent >= numpyDataType.itemsize, "MPI extent is too small"
    if MPIDataType.extent != numpyDataType.itemsize:
        MPIDataType = MPIDataType.Create_resized(0, numpyDataType.itemsize)
    MPIDataType = MPIDataType.Commit()

    msgbuffer = np.zeros(1, dtype=numpyDataType)

    rank = MPI.COMM_WORLD.Get_rank()
    
    if rank == 0:
        msgbuffer[0]['msgtag']=-34
        msgbuffer[0]['counter']=1<<60
        msgbuffer[0]['energy']=3e20
        MPI.COMM_WORLD.Send([msgbuffer, MPIDataType], dest = 1)
    if rank == 1:
        MPI.COMM_WORLD.Recv([msgbuffer, MPIDataType], source = 0)
        if verbose:
            print "rank 1 received msgbuffer = %d %d %e" % \
                (msgbuffer[0]['msgtag'],
                 msgbuffer[0]['counter'],
                 msgbuffer[0]['energy'])

    MPIDataType.Free()
    
timingDict={}

class timecall(object):
    def __init__(self, timingDict):
        self.timingDict=timingDict
    def __call__(self,f):
        funcName = f.__name__
        if funcName not in self.timingDict:
            self.timingDict[funcName]={'total_time':0.0,'total_calls':0}
        def time_wrap_f(*args, **kwargs):
            t0=time.time()
            res=f(*args, **kwargs)
            self.timingDict[funcName]['total_time'] += time.time()-t0
            self.timingDict[funcName]['total_calls'] += 1
            return res
        return time_wrap_f

@timecall(timingDict)
def timedMPIType(verbose):
    MPIType(verbose)

if __name__ == '__main__':
#    collectiveCommunicationInSubGroup()
    MPIType()
    MPI.COMM_WORLD.Barrier()
    for i in range(10000):
        timedMPIType(verbose=False)
    MPI.COMM_WORLD.Barrier()
    rank = MPI.COMM_WORLD.Get_rank()
    for key,val in timingDict.iteritems():
        print "rank=%d: Timing Dict says %d calls to %s take %.2f sec" % \
        (rank, val['total_calls'], key, val['total_time'])
