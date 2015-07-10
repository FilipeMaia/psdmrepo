.. _progr:


#########################
mpi4py Programming Tips
#########################

On this page we collect techniques for MPI programming with mpi4py in Python. The mpi4py website: http://mpi4py.scipy.org/docs/usrman/tutorial.html is the place to start. The next resource is the demos and tests of the mpi4py package. When logged into the psana machines at LCLS, one can find these package files in the directory /reg/g/psdm/sw/external/build/mpi4py. The mpi4py google group is very helpful as well. Below we go over tips/techniques learned during development of the ParCorAna psana package. Some of the code below can be found in the ParCorAna/examples/mpi4pyExamples.py file.

****************************
Check Health of the Cluster
****************************

The first thing the ParCorAna MPI driver program does is an MPI_Barrier. I started to do this after chasing down an issue with our cluster. Sometimes a node is down and your program will stall during the first MPI Communication with that node. When this happens it can be hard to figure out that it is a cluster problem vs. a problem in your program. The program starts with something like::

   sys.stdout.write("Before first Collective MPI call (MPI_Barrier). If no output follows, there is a problem with the cluster.\n")
   sys.stdout.flush()
   MPI.COMM_WORLD.Barrier()
   sys.stdout.write("After first collective MPI call.")

It is interesting to time how long it takes rank 0 to get through the barrier - a measure of how long it takes to get all the ranks loaded across the cluster.

**************************************
Design Program to Run outside of MPI
**************************************

Debugging intereactively with Python can be very useful. I like to insert::

  import IPython
  IPython.embed()

in the code during development. This doesn't work so well when the program is running under MPI. I designed the program to take a parameter that species a development mode. When this is set, I create mock objects for certain MPI data and try to run as much of the program as I can in a serial fashion, introducing alternative routines when I need to. Development mode can only run on small problems.

************************************************************
Collective Scatterv Communication within a Subset of Ranks
************************************************************

ParCorAna has 4 types of ranks, servers, workers, a viewer and a master. The servers take turns doing collective communication with the workers to scatter a large array to the workers. This is done by setting up a separate communicator for each server. These communicators have one server, and all the workers in them. One can do this by setting up a group of the ranks that you want, and then creating a communicator from that group. One way to do this is::
  
    serverRankA = 0
    serverRankB = 1
    workerRanks = [2,3]
    worldGroup = MPI.COMM_WORLD.Get_group()
    serverRankBandWorkersGroup = worldGroup.Excl([serverRankA])
    serverRankBandWorkersComm = MPI.COMM_WORLD.Create(serverRankBandWorkersGroup)
    serverBrankInNewComm = MPI.Group.Translate_ranks(worldGroup, [serverRankB], serverRankBandWorkersGroup)[0]

When carrying out collective communication from the server to the workers, you need to know the rank of a server in the new communicator::

    serverBrankInNewComm = MPI.Group.Translate_ranks(worldGroup, [serverRankB], serverRankBandWorkersGroup)[0]

Below is an example of doing Scatterv in this new comm. The server scatters an array of 11 floats, 6 to the worker with rank 2, and 5 to the worker with rank 3::

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


One gets the output::

  worker with world rank=3 received: array([  6.,   7.,   8.,   9.,  10.], dtype=float32)
  worker with world rank=2 received: array([ 0.,  1.,  2.,  3.,  4.,  5.], dtype=float32)

Other ways to make intra or extra communicators are with split, and color - a collective call where each rank passes its color which identifies with group it will be in.


***************************
Call MPI_Abort on Exception
***************************

I found it useful to catch any exception, print it, and call MPI_Abort. Helped with getting clean exits.

*****************
Logging with Rank
*****************

I found it useful to use the Python logging module and create loggers that had the MPI rank in the header. I had issues with having loggers log their messages twice. As I understand it, you have to be careful do not inadvertenly have too many handlers for one logger, or create several loggers with the same name. I ended up maintaining a global dictionary of loggers by name to make sure I didn't create the same logger twice, and then explicitly clearing the handlers for the loggers. I had a function that looked something like this::

  loggers = {}
   
  def getLogger(rank, lvl='INFO', propagate=False):
    global loggers
    loggerName = 'rnk:%d' % rank
    numLevel = getattr(logging, lvl)
    logger = loggers.get(loggerName,None)
    if logger is not None:
        return logger

    logger = logging.getLogger(loggerName)
    logger.setLevel(numLevel)
    logger.propagate=propagate
    ch = logging.StreamHandler()
    ch.setLevel(numLevel)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s' )
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(ch)
    loggers[loggerName]=logger
    return logger

*****************************
Messaging with new MPI Types
*****************************

Here is an example of using a numpy dtype with named fields (like a C struct) to create a new MPI Type in mpi4py, and then use the numpy type to send messages. First you create your numpy datatype::

    fields = [('msgtag',np.int32),
              ('counter',np.int64),
              ('energy',np.float32)]

    numpyDataType = np.dtype(fields)

Now you start to make the mactching MPI type. The first two pieces are easy::

    MPI_blocklens = (1,1,1)
    MPI_types = (MPI.INT32_T, MPI.INT64_T, MPI.FLOAT)

for the displacements, we build it from the displacements of the numpy type::

    fldNames = [fld[0] for fld in fields]
    MPI_displacements = tuple([numpyDataType.fields[nm][1] for nm in fldNames])

Then you are ready to create the MPI Type. If it's extent is not at least as long as the numpy type, something went wrong, perhaps one didn't match an MPI type like MPI_INT32_T with the correct numpy type of np.int32. It may however be longer than the numpy type - in which case you need to resize it. After this you are ready to commit::

    assert MPIDataType.extent >= numpyDataType.itemsize, "MPI extent is too small"
    if MPIDataType.extent != numpyDataType.itemsize:
        MPIDataType = MPIDataType.Create_resized(0, numpyDataType.itemsize)

    MPIDataType = MPIDataType.Commit()

Here is an example of sending a message with the numpy type::

    msgbuffer = np.zeros(1, dtype=numpyDataType)

    rank = MPI.COMM_WORLD.Get_rank()
    
    if rank == 0:
        msgbuffer[0]['msgtag']=-34
        msgbuffer[0]['counter']=1<<60
        msgbuffer[0]['energy']=3e20
        MPI.COMM_WORLD.Send([msgbuffer, MPIDataType], dest = 1)
    if rank == 1:
        MPI.COMM_WORLD.Recv([msgbuffer, MPIDataType], source = 0)
        print "rank 1 received msgbuffer = %d %d %e" % \
            (msgbuffer[0]['msgtag'],
             msgbuffer[0]['counter'],
             msgbuffer[0]['energy'])

when done with the committed type, one must free it::

    MPIDataType.Free()

**********
Timing
**********

For tuning an MPI program, it is important to get a sense of how long different stages of the program are taking. My approach was to define a global dictionary, and a Python wrapper that would accumulate time in this dictionary. This would look like::

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


  timingDict={}

  @timecall(timingDict)
  def foo(verbose):
      pass
 
Then one can look at timingDict after calling foo a number of times.

