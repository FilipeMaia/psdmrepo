## system
from mpi4py import MPI
import os
import numpy as np
import time
import StringIO
import traceback
import copy
import collections
import logging

## this package
import CommSystemUtil
import PsanaUtil
from MessageBuffers import SM_MsgBuffer, MVW_MsgBuffer
import Timing
from XCorrBase import XCorrBase
import Counter120hz

########## for Timing.timecall
VIEWEREVT = 'update'
CSPADEVT = 'cspadevt'
timingorder =[]  # list of names in order inserted
timingdict = {}
###########

def roundRobin(n, dictData):
    '''returns list from dict by round robin over keys
 
    Args:
      n (int):         length of list to return
      dictData (dict): values are lists to select return items from

    Returns:
      list: n items from values of dictData

    Examples:
      >>> roundRobin(5, {'keyA':[1,2,3], 'keyB':[10,20,30]})
      [1,10,2,20,3]
    '''
    if n==0:
        return []
    keys = dictData.keys()
    keys.sort()
    assert len(keys)>0, "rounRobin will fail to find n=%d items from empty dict" % n
    nextVal = dict([(ky,0) for ky in keys])
    results = []
    keyIndex = 0
    keysWithAllValuesUsed = set()
    while len(results)<n:
        ky = keys[keyIndex]
        if nextVal[ky] < len(dictData[ky]):
            results.append(dictData[ky][nextVal[ky]])
            nextVal[ky] += 1
        else:
            keysWithAllValuesUsed.add(ky)
        if len(keysWithAllValuesUsed)==len(keys):
            break
        keyIndex += 1
        keyIndex %= len(keys)

    if len(results)!=n:
        raise Exception("roundRobin did not get n=%d values from dictData=%r" % (n, dictData))
    return results
                           
def identifyServerRanks(comm, numServers, serverHosts=None):
    '''returns ranks to be the servers, puts servers on distinct hosts if possible.

    Server ranks will be picked in a round robin fashion among the distinct hosts
    in the MPI world.

    Args:
      comm (MPI.Comm): communicator from whith mpi world hostnames are identified.
                       *IMPORTANT* This function may do blocking collective communication 
                       with comm. All ranks in comm should call this during initialization.
      numServers (int): number of servers to find
      serverHosts (list, optional): None or empty means use default host assignment. Otherwise 
                                    list must be a set of unique hostnames. 
    Returns:
      (tuple): tuple containing:

        * servers (dict)- a list of ints, the ranks in the comm  to use as servers
        * hostmsg (str)- a logging string about  the hosts chosen
    '''

    assert comm.Get_size() >= numServers, "More servers requested than are in the MPI World. numServers=%d > MPI World Size=%d" % \
        (numServers, comm.Get_size())

    if serverHosts is None:
        serverHosts = []

    ## identify host -> rank map through collective MPI communication
    localHostName = MPI.Get_processor_name()
    allHostNames = []
    allHostNames = comm.allgather(localHostName, None)
    assert len(allHostNames) == comm.Get_size(), 'allgather failed - did not get one host per rank'
    serverHost2ranks = collections.defaultdict(list)
    for ii, hostName in enumerate(allHostNames):
        serverHost2ranks[hostName].append(ii)

    for host in serverHosts:
        assert host in serverHost2ranks.keys(), "specified host: %s not in MPI world hosts: %r" % (host, serverHost2ranks.keys())

    if len(serverHosts) == 0:
        ranksForRoundRobin = serverHost2ranks
    else:
        ranksForRoundRobin = dict()
        for host in serverHosts:
            ranksForRoundRobin[host]=serverHost2ranks[host]

    serverRanks = roundRobin(numServers, ranksForRoundRobin)

    hostmsg = 'server host assignment:'
    rank2host=collections.defaultdict(list)
    for host,rankList in serverHost2ranks.iteritems():
        for rank in rankList:
            rank2host[rank].append(host)

    hostmsg += ", ".join(["rnk=%d->host=%s" % (rank, rank2host[rank]) for rank in serverRanks \
                          if rank in serverRanks])
    return serverRanks, hostmsg

class MPI_Communicators:
    '''Keeps track of the different communicators for collective
    communications. Includes many parameters that identify the ranks
    and the communicators they are in.

    Call identifyCommSubsystems to return a initialized instance or
    getTestingMPIObject()

    Then call the method setMask.
    '''
    def __init__(self):
        pass

    def setLogger(self, verbosity):
        self.logger = CommSystemUtil.makeLogger(self.testMode, self.isMaster, \
                        self.isViewer, self.isServer, self.rank, verbosity)

    def notWorker2toN(self):
        if self.isWorker and not self.isFirstWorker:
            return False
        return True

    # these functions make it simpler to exclude logging from all the workers.
    # all workers generally do the same thing. If they all logged it gets noisy.
    def logInfo(self, msg, allWorkers=False):
        '''By default logs a message only from worker one
        '''
        if allWorkers or self.notWorker2toN():
            self.logger.info(msg)

    def logWarning(self, msg, allWorkers=False):
        '''By default logs a message only from worker one
        '''
        if allWorkers or self.notWorker2toN():
            self.logger.warning(msg)

    def logDebug(self, msg, allWorkers=False):
        '''By default logs a message only from worker one
        '''
        if allWorkers or self.notWorker2toN():
            self.logger.debug(msg)

    def logError(self, msg, allWorkers=False):
        '''By default logs a message only from worker one
        '''
        if allWorkers or self.notWorker2toN():
            self.logger.error(msg)


    def setMask(self, maskNdarrayCoords):
        '''sets scatterv parameters and stores mask

        Args:
          maskNdarrayCoords (numpy.ndarray): integer array, 1 for elements that should be processed. 
                                              It must have the same shape as the NDArray for the detector.

        Notes:
          sets the following attributes

          * totalElements:                 number of pixels that are 1 in the mask
          * workerWorldRankToCount[rank]:  number of element that worker processes
          * workerWorldRankToOffset[rank]: offset of where those elements start in 
                                           a flattened version of the ndarray
          * maskNdarrayCoords:             the mask as a logical True/False array
                                           shape has not been changed

        '''
        mask_flat = maskNdarrayCoords.flatten()
        maskValues = set(mask_flat)
        assert maskValues.union(set([0,1])) == set([0,1]), "mask contains values other than 0 and 1." + \
            (" mask contains %d distinct values" % len(maskValues))
        assert 1 in maskValues, "The mask does not have the value 1, it is all 0. Elements marked with 1 are processed"
        self.totalElements = np.sum(mask_flat)
        self.maskNdarrayCoords = maskNdarrayCoords == 1 # is is important to convert mask to array of bool, np.bool

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logDebug("MPIParams.setMask: loaded and stored mask with shape=%s elements included=%d excluded=%s" % \
                          (self.maskNdarrayCoords.shape, np.sum(self.maskNdarrayCoords),
                           np.sum(0==self.maskNdarrayCoords)))

        workerOffsets, workerCounts = CommSystemUtil.divideAmongWorkers(self.totalElements, 
                                                                 self.numWorkers)
        assert self.numWorkers == len(self.workerRanks)
        assert len(workerCounts)==self.numWorkers

        self.workerWorldRankToCount = {}
        self.workerWorldRankToOffset = {}

        for workerRank, workerOffset, workerCount in zip(self.workerRanks,
                                                         workerOffsets,
                                                         workerCounts):
            self.workerWorldRankToCount[workerRank] = workerCount
            self.workerWorldRankToOffset[workerRank] = workerOffset

        for serverRank in self.serverRanks:
            serverCommDict = self.serverWorkers[serverRank]
            serverRankInComm = serverCommDict['serverRankInComm']
            scatterCounts = copy.copy(workerCounts)
            serverCount = 0
            scatterCounts.insert(serverRankInComm, serverCount)
            scatterOffsets = copy.copy(workerOffsets)
            # the value we use for the server offset is not important, but checkCountsOffsets
            # which we call below checks for offset[i+1]=offset[i]+count[i]
            if serverRankInComm == 0:
                serverOffset = 0
            else:
                serverOffset = workerOffsets[serverRankInComm-1]+workerCounts[serverRankInComm-1]
            scatterOffsets.insert(serverRankInComm, serverOffset)
            self.serverWorkers[serverRank]['groupScattervCounts'] = tuple(scatterCounts)
            self.serverWorkers[serverRank]['groupScattervOffsets'] = tuple(scatterOffsets)
            CommSystemUtil.checkCountsOffsets(scatterCounts, scatterOffsets, self.totalElements)

def getTestingMPIObject():
    '''mock up MPI_Communicators object for test_alt mode.
    
    Simulate an MPI_Communicators which looks like a single server/worker/viewer as being the
    same rank.
    '''
    mp = MPI_Communicators()
    mp.isServer = True
    mp.isWorker = True
    mp.isFirstWorker = True
    mp.isViewer = True
    mp.isMaster = False
    mp.testMode = True
    mp.rank = MPI.COMM_WORLD.rank
    assert mp.rank == 0, "test MPI object is for non-MPI environment, but MPI world rank != 0"
    mp.workerRanks = [mp.rank]
    mp.numWorkers = 1
    mp.serverRanks = [0]
    mp.serverWorkers = {}
    mp.serverWorkers[0]={'serverRankInComm':0}
    mp.viewerRankInViewerWorkersComm = 0
    return mp

def identifyCommSubsystems(serverRanks, worldComm=None):
    '''Return a fully initialized instance of a MPI_Communicators object
    The object will contain the following attributes::

      serverRanks      - ranks that are servers in COMM_WORLD
      comm             - duplicate of the COMM_WORLD
      rank             - rank in COMM_WORLD
      worldNumProcs    - size of COMM_WORLD
      masterRank       - master rank in COMM_WORLD
      viewerRank       - viewer rank in COMM_WORLD
      workerRanks      - list of worker ranks in COMM_WORLD
      firstWorkerRank  - first worker rank in COMM_WORLD
      numWorkers       - number of workers

      # these parameters identify which group this rank is
      isMaster
      isViewer
      isServer
      isFirstWorker
      isWorker

      isTestMode = False

      masterWorkersComm - intra communicator for master/workers collective communication
      viewerWorkersComm - intra communicator for viewer/workers collective communication
      viewerRankInViewerWorkersComm      - viewer rank in the above intra-communicator
      firstWorkerRankInViewerWorkersComm - first worker rank in the above intra-communicator

      # the following is a dict with one key for each server rank
      serverWorkers[serverRank]['comm'] - intra communicator, this server and all workers
      serverWorkers[serverRank]['serverRankInComm'] - 
      serverWorkers[serverRank]['workerRanksInCommDict'] -  a key for this dict is a
         worker rank in the world space. The value is the rank in the 'comm' value
      
    '''
    assert len(serverRanks) > 0, "need at least one server"
    assert min(serverRanks) >= 0, "cannot have negative server ranks"
    if worldComm is None:
        worldComm = MPI.COMM_WORLD

    mc = MPI_Communicators()
    mc.testMode = False
    mc.serverRanks = serverRanks
    mc.comm = worldComm.Dup()
    mc.rank = mc.comm.Get_rank()
    mc.worldNumProcs = mc.comm.Get_size()
    assert mc.worldNumProcs >= 4, "need at least 4 ranks for comm system (server/master/viewer/workers)"
    assert mc.worldNumProcs - len(mc.serverRanks) >= 3, "With %d servers but only %d ranks in world, not enough ranks for worker/viewer/master" % \
        (len(mc.serverRanks), mc.worldNumProcs)
    availRanks = [rank for rank in range(mc.worldNumProcs) \
                  if rank not in mc.serverRanks]
    assert len(availRanks)>=3, "To many servers for world size. " + \
        ("Only %d ranks left for master/viewer/workers" % len(availRanks))
    mc.viewerRank = min(availRanks)
    availRanks.remove(mc.viewerRank)
    mc.masterRank = min(availRanks)
    availRanks.remove(mc.masterRank)
    mc.workerRanks = availRanks
    mc.firstWorkerRank = min(mc.workerRanks)

    mc.isMaster = mc.rank == mc.masterRank
    mc.isViewer = mc.rank == mc.viewerRank
    mc.isServer = mc.rank in mc.serverRanks
    mc.isFirstWorker = mc.rank == mc.firstWorkerRank
    mc.isWorker = mc.rank not in ([mc.masterRank, mc.viewerRank] + mc.serverRanks)
    mc.numWorkers = len(mc.workerRanks)
    
    worldGroup = mc.comm.Get_group()
    masterWorkersGroup = worldGroup.Excl([mc.viewerRank] + mc.serverRanks)
    viewerWorkersGroup = worldGroup.Excl([mc.masterRank] + mc.serverRanks)

    mc.masterWorkersComm = mc.comm.Create(masterWorkersGroup)  # will be an invalid group on proc with viewer
    mc.viewerWorkersComm = mc.comm.Create(viewerWorkersGroup)  # will be an invalid group on proc with master

    mc.serverWorkers = dict()
    for serverRank in mc.serverRanks:
        otherServers = [rank for rank in mc.serverRanks if rank != serverRank]
        serverWorkersGroup = worldGroup.Excl([mc.viewerRank, mc.masterRank]+otherServers)
        serverRankInComm = MPI.Group.Translate_ranks(worldGroup, [serverRank],
                                                     serverWorkersGroup)[0]
        workerRanksInComm = MPI.Group.Translate_ranks(worldGroup, mc.workerRanks,
                                                     serverWorkersGroup)
        workerRanksInCommDict = dict(zip(mc.workerRanks,workerRanksInComm))
        serverWorkersComm = mc.comm.Create(serverWorkersGroup)
        
        mc.serverWorkers[serverRank]={'comm':serverWorkersComm,
                                      'serverRankInComm':serverRankInComm,
                                      'workerRanksInCommDict':workerRanksInCommDict,
                                      }
    
    tmp1,tmp2 = MPI.Group.Translate_ranks(worldGroup, [mc.firstWorkerRank, mc.viewerRank], 
                                          viewerWorkersGroup)
    mc.firstWorkerRankInViewerWorkersComm,mc.viewerRankInViewerWorkersComm = tmp1,tmp2
    tmp1,tmp2 = MPI.Group.Translate_ranks(worldGroup, [mc.firstWorkerRank, mc.masterRank], 
                                          masterWorkersGroup)
    mc.firstWorkerRankInMasterWorkersComm, mc.masterRankInMasterWorkersComm = tmp1,tmp2

    return mc

class ScatterDataQueue(object):
    def __init__(self, maskToCopyOutData, dtypeForScatter, logger):
        assert maskToCopyOutData.dtype == np.bool
        self.maskToCopyOutData = maskToCopyOutData
        self.numElements = np.sum(maskToCopyOutData)
        self.dtypeForScatter = dtypeForScatter
        self.iterDataQueue = []
        self.scatterDataQueue = []
        self.logger = logger

    def empty(self):
        assert len(self.iterDataQueue)==len(self.scatterDataQueue), "ScatterDataQueue: internal lists are not the same length"
        return len(self.iterDataQueue) == 0

    def nextEventId(self):
        assert not self.empty(), "ScatterDataQueue: nextEventId called on non-empty data"
        return self.iterDataQueue[0].eventId()

    def popHead(self):
        assert not self.empty(), "ScatterDataQueue: popHead called on non-empty data"
        assert len(self.iterDataQueue)==len(self.scatterDataQueue), "ScatterDataQueue: internal lists are not the same length"
        datum = self.iterDataQueue.pop(0)
        scatter1Darray = self.scatterDataQueue.pop(0)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("ScatterDataQueue: popHead %s" % datum)
        return scatter1Darray

    def addFrom(self, dataGen, num):
        assert len(self.iterDataQueue)==len(self.scatterDataQueue), "ScatterDataQueue: internal lists are not the same length"
        while num > 0:
            try:
                datum = dataGen.next()
            except StopIteration:
                self.logger.debug("ScatterDataQueue: addFrom: dataGen is empty")
                return
            scatterData = np.zeros(self.numElements, dtype=self.dtypeForScatter)
            scatterData[:] = datum.dataArray[self.maskToCopyOutData]
            self.iterDataQueue.append(datum)
            self.scatterDataQueue.append(scatterData)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("ScatterDataQueue: addFrom %s" % datum)
            num -= 1
    
class RunServer(object):
    '''runs server rank

    This function carries out the server side communication in the package.
    It does the following

    ::
    
      * iteratates through data in Python generator argument
      * for each datum from generator:
      ** sends sec/nsec to master
      ** gets OkForWorkers or Abort from master
      ** upon OkForWorkers calls .sendToWorkers(datum) method in dataIterator
         The dataIterator handles details such as scattering detector data to workers

      Args:
        dataIter:  instance of a callback class. Must provide these methods:
                   .dataGenerator()   a Python generator. Each returned object
                                      must have a time() returning sec,nsec
                   .sendToWorkers(datum)  receives a datum returned by dataGenerator.
                                          user can now send data to workers
                                          workers will know the upcoming time already
        comm:       MPI intra-communicator for server ranks and master rank.
        rank:       rank of this server
        masterRank: rank of master
        logger:     Python logging logger
    '''
    def __init__(self, dataIter, xCorrBase, comm, rank, masterRank, logger):
        self.dataIter = dataIter
        self.xCorrBase = xCorrBase
        self.comm = comm
        self.rank = rank
        self.masterRank = masterRank
        self.logger = logger

    def recordTimeToGetData(self, startTime, endTime, numGets=1):
        global timingdict
        global timingorder

        if startTime is None: return
        key = 'ServerTimeToGetData'
        if key not in timingdict:
            timingdict[key]=[0.0, 0, 'event', 1e-3, 'ms']
            timingorder.append(key)
        timingdict[key][0] += endTime-startTime
        timingdict[key][1] += numGets

    def run(self):
        sendEventReadyBuffer = SM_MsgBuffer(rank=self.rank)
        sendEventReadyBuffer.setEvt()
        receiveOkForWorkersBuffer = SM_MsgBuffer(rank=self.rank)
        abortFromMaster = False
        dataGen = self.dataIter.dataGenerator()
        scatterDataQueue=ScatterDataQueue(self.xCorrBase.mp.maskNdarrayCoords, np.float32, self.logger)
        initialQueueSize = 1
        t0 = time.time()
        scatterDataQueue.addFrom(dataGen, initialQueueSize)
        self.recordTimeToGetData(t0, time.time(), initialQueueSize)
        while not scatterDataQueue.empty():
            sec, nsec, fiducials = scatterDataQueue.nextEventId()
            sendEventReadyBuffer.setEventId(sec, nsec, fiducials)
            if self.logger.isEnabledFor(logging.DEBUG):
                debugMsg = "RunServer: data to scatter,"
                debugMsg += " Event Id: sec=0x%8.8X nsec=0x%8.8X fid=0x%5.5X." % (sec, nsec, fiducials)
                debugMsg += " Before Send EVT"
                self.logger.debug(debugMsg)
            self.comm.Send([sendEventReadyBuffer.getNumpyBuffer(),
                            sendEventReadyBuffer.getMPIType()],
                           dest=self.masterRank)
            self.logger.debug("RunServer: After Send, before Recv, first adding to Queue")
            # master is most likely telling one of the other servers to scatter right now.
            # read new data before waiting to be told to scatter
            t0 = time.time()
            scatterDataQueue.addFrom(dataGen, 1) 
            self.recordTimeToGetData(t0, time.time(), 1)
            self.comm.Recv([receiveOkForWorkersBuffer.getNumpyBuffer(), 
                            receiveOkForWorkersBuffer.getMPIType()],
                           source=self.masterRank)
            if receiveOkForWorkersBuffer.isSendToWorkers():
                self.logger.debug("RunServer: After Recv. is Send to workers")
                toScatter1DArray = scatterDataQueue.popHead()
                self.xCorrBase.serverWorkersScatter(detectorData1Darray=toScatter1DArray, 
                                                    serverWorldRank=None)

            elif receiveOkForWorkersBuffer.isAbort():
                self.logger.debug("RunServer: After Recv. Abort")
                abortFromMaster = True
                break
            else:
                raise Exception("unknown msgtag from master. buffer=%r" % receiveOkForWorkersBuffer)

        if abortFromMaster:
            self.dataIter.abortFromMaster()
        else:
            sendEventReadyBuffer.setEnd()
            self.logger.debug("CommSystem.run: Before Send END")
            self.comm.Send([sendEventReadyBuffer.getNumpyBuffer(), 
                            sendEventReadyBuffer.getMPIType()],
                           dest=self.masterRank)
            self.logger.debug("CommSystem.run: After Send END. Finished")

class RunMaster(object):
    '''runs master message passing.
    '''
    def __init__(self, worldComm, masterRank, viewerRank, serverRanks, 
                 masterWorkersComm, masterRankInMasterWorkersComm,
                 updateIntervalEvents, hostmsg, logger):

        self.worldComm = worldComm
        self.masterRank = masterRank
        self.serverRanks = serverRanks
        self.masterWorkersComm = masterWorkersComm
        self.masterRankInMasterWorkersComm = masterRankInMasterWorkersComm
        self.updateIntervalEvents = updateIntervalEvents
        self.worldComm = worldComm
        self.viewerRank = viewerRank
        self.logger = logger
        self.logger.info(hostmsg)

        # initially all servers are not ready
        self.notReadyServers = [r for r in serverRanks]  # MPI Test on request is false
        self.readyServers = []                           # MPI Test on request is True
        self.finishedServers = []                        # rank has returend end

        self.sendOkForWorkersBuffer = SM_MsgBuffer()
        self.bcastWorkersBuffer = MVW_MsgBuffer()
        self.viewerBuffer = MVW_MsgBuffer()
        self.lastUpdate = 0
        self.numEvents = 0
        
        self.eventIdToCounter = None

    def getEarliest(self, serverDataList):
        '''Takes a list of server data buffers. identifies oldest server::
        ARGS:
          serverDataList:  each element is a SeversMasterMessaging buffer
        RET:
         the buffer with the earliest time
        '''
        earliestIdx = 0
        sec, nsec, fiducials = serverDataList[earliestIdx].getEventId()
        for candIdx in range(1,len(serverDataList)):
            curSec, curNsec, curFiducials = serverDataList[candIdx].getEventId()
            earlierByTime = False
            if (curSec < sec) or ((curSec == sec) and (curNsec < nsec)):
                earlierByTime = True
            # TODO: should also check if earlier by fiducial
            if earlierByTime:
                sec = curSec
                nsec = curNsec
                fiducials = curFiducials
                earliestIdx = candIdx
        return serverDataList[earliestIdx]
        
    def initRecvRequestsFromServers(self):
        # create buffers for receiving, and the requests
        serverReceiveData = dict()
        serverRequests = dict()
        self.logger.debug("CommSystem: before first Irecv from servers")
        for serverRank in self.serverRanks:
            serverReceiveBuffer = SM_MsgBuffer(rank=serverRank)
            firstServerRequest = self.worldComm.Irecv([serverReceiveBuffer.getNumpyBuffer(), 
                                                               serverReceiveBuffer.getMPIType()],
                                                              source=serverRank)
            serverReceiveData[serverRank] = serverReceiveBuffer
            serverRequests[serverRank] = firstServerRequest
        self.logger.debug("CommSystem: after first Irecv from servers")
        return serverReceiveData, serverRequests

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def informWorkersOfNewData(self, selectedServerRank, sec, nsec, fiducials, counter):
        self.bcastWorkersBuffer.setEvt()
        self.bcastWorkersBuffer.setRank(selectedServerRank)
        self.bcastWorkersBuffer.setSeconds(sec)
        self.bcastWorkersBuffer.setNanoSeconds(nsec)
        self.bcastWorkersBuffer.setFiducials(fiducials)
        self.bcastWorkersBuffer.setCounter(counter)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("CommSystem: before Bcast -> workers EVT sec=0x%8.8d nsec=0x%8.8X fid=0x%5.5X counter=%d" % \
                              (self.bcastWorkersBuffer.getSeconds(), self.bcastWorkersBuffer.getNanoSeconds(), 
                               self.bcastWorkersBuffer.getFiducials(), self.bcastWorkersBuffer.getCounter()))
        self.masterWorkersComm.Bcast([self.bcastWorkersBuffer.getNumpyBuffer(),
                                      self.bcastWorkersBuffer.getMPIType()],
                                     root=self.masterRankInMasterWorkersComm)
        self.masterWorkersComm.Barrier()
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("CommSystem: after Bcast/Barrier -> workers EVT counter=%d" % counter)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def informViewerOfUpdate(self, sec, nsec, fiducials, counter):
        self.viewerBuffer.setUpdate()
        self.viewerBuffer.setSeconds(sec)
        self.viewerBuffer.setNanoSeconds(nsec)
        self.viewerBuffer.setFiducials(fiducials)
        self.viewerBuffer.setCounter(counter)
        self.worldComm.Send([self.viewerBuffer.getNumpyBuffer(),
                             self.viewerBuffer.getMPIType()],
                            dest=self.viewerRank)

    def sendEndToViewer(self):
        self.viewerBuffer.setEnd()
        self.worldComm.Send([self.viewerBuffer.getNumpyBuffer(),
                             self.viewerBuffer.getMPIType()],
                            dest=self.viewerRank)

    @Timing.timecall(CSPADEVT)
    def informWorkersToUpdateViewer(self):
        self.logger.debug("CommSystem: before Bcast -> workers UPDATE")
        self.bcastWorkersBuffer.setUpdate()
        self.masterWorkersComm.Bcast([self.bcastWorkersBuffer.getNumpyBuffer(),
                                      self.bcastWorkersBuffer.getMPIType()],
                                     root=self.masterRankInMasterWorkersComm)
        self.masterWorkersComm.Barrier()
        self.logger.debug("CommSystem: after Bcast/Barrier -> workers UPDATE")

    def sendEndToWorkers(self):
        self.bcastWorkersBuffer.setEnd()
        self.logger.debug("CommSystem: before Bcast -> workers END")
        self.masterWorkersComm.Bcast([self.bcastWorkersBuffer.getNumpyBuffer(),
                                      self.bcastWorkersBuffer.getMPIType()],
                                     root=self.masterRankInMasterWorkersComm)
        self.masterWorkersComm.Barrier()
        self.logger.debug("CommSystem: after Bcast/Barrier -> workers END")

    def run(self):
        ########## begin helper functions ########
        def waitOnServers(self):
            '''called during communication loop. 
            Called when not all servers are ready, or at end.
            The master wants all servers to have data available, or to be done with their
            data before it picks the next server to scatter.

            identifies a done server through waitany
            tests all done servers
            idenfies finisned servers among done servers

            at the end of this function,

            self.notReadyServers + self.finishedServers + self.readyServers 

            will be the same as it was on entry, and self.notReadyServers will be 0. There will be
            at least one server in the finishedServers or readyServers
            '''
            assert len(self.notReadyServers)>0, "waitOnServers called, but no not-ready servers"

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("CommSystem: before waitany. notReadyServers=%s" % self.notReadyServers)
            requestList = [serverRequests[rnk] for rnk in self.notReadyServers]
            MPI.Request.Waitall(requestList)

            newReadyServers = [s for s in self.notReadyServers]
            self.notReadyServers = []

            newFinishedServers = [server for server in newReadyServers \
                                  if serverReceiveData[server].isEnd()]
            self.finishedServers.extend(newFinishedServers)
            for server in newFinishedServers:
                # take finished servers out of pool to get next event from
                newReadyServers.remove(server)
            
            self.readyServers.extend(newReadyServers)
        ############ end helper functions ######

        serverReceiveData, serverRequests = self.initRecvRequestsFromServers()

        numEventsAtLastDataRateMsg = 0
        timeAtLastDataRateMsg = time.time()
        startTime = time.time()
        waitAllTime = 0.0
        waitAllNum = 0
        noData = True
        while True:
            # a server must be in one of: ready, noReady or finished
            serversAccountedFor = len(self.readyServers) + len(self.finishedServers) + \
                                  len(self.notReadyServers)
            assert serversAccountedFor == len(self.serverRanks), \
                "loop invariant broken? #servers=%d != #accountedfor=%d" % \
                (len(self.serverRanks), len(serversAccountedFor))

            assert len(set(self.readyServers))==len(self.readyServers), "readyServers=%r contains dups" % self.readyServers
            assert len(set(self.finishedServers))==len(self.finishedServers), "readyServers=%r contains dups" % self.readyServers
            assert len(set(self.notReadyServers))==len(self.notReadyServers), "notReadyServers=%r contains dups" % self.notReadyServers

            if len(self.finishedServers)==len(self.serverRanks): 
                break

            if len(self.notReadyServers)!=0:
                t0 = time.time()
                waitOnServers(self)
                waitAllTime += time.time()-t0
                waitAllNum += 1
            if len(self.readyServers)==0:
                # the servers we waited on are finished. 
                continue

            earlyServerData = self.getEarliest([serverReceiveData[server] for server in self.readyServers])
            selectedServerRank = earlyServerData.getRank()
            sec, nsec, fiducials = earlyServerData.getEventId()
            noData = False
            if self.eventIdToCounter is None:
                self.eventIdToCounter = Counter120hz.Counter120hz(sec, nsec, fiducials)
            counter = self.eventIdToCounter.getCounter(sec, fiducials)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("CommSystem: next server rank=%d sec=0x%8.8X nsec=0x%8.8X fiducials=0x%5.5X counter=%5d" % \
                                  (selectedServerRank, sec, nsec, fiducials, counter))
            self.informWorkersOfNewData(selectedServerRank, sec, nsec, fiducials, counter)

            # tell server n to scatter to workers
            self.sendOkForWorkersBuffer.setSendToWorkers()
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("CommSystem: before SendOkForWorkers to server %d" % selectedServerRank)
            self.worldComm.Send([self.sendOkForWorkersBuffer.getNumpyBuffer(), 
                                         self.sendOkForWorkersBuffer.getMPIType()], 
                                        dest=selectedServerRank)

            # do new Irecv from worker n
            self.readyServers.remove(selectedServerRank)
            self.notReadyServers.append(selectedServerRank)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("CommSystem: after sendOk, before replacing request with Irecv from rank %d" % selectedServerRank)
            serverReceiveBuffer = serverReceiveData[selectedServerRank]
            serverRequests[selectedServerRank] = self.worldComm.Irecv([serverReceiveBuffer.getNumpyBuffer(), 
                                                                               serverReceiveBuffer.getMPIType()],  \
                                                                              source = selectedServerRank)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("CommSystem: after Irecv from rank %d" % selectedServerRank)

            # check to see if there should be an update for the viewer
            self.numEvents += 1
            if (self.updateIntervalEvents > 0) and (self.numEvents - self.lastUpdate > self.updateIntervalEvents):
                self.lastUpdate = self.numEvents
                self.logger.debug("CommSystem: Informing viewers and workers to update" )
                self.informViewerOfUpdate(sec, nsec, fiducials, counter)
                self.informWorkersToUpdateViewer()

            # check to display message
            eventsSinceLastDataRateMsg = self.numEvents - numEventsAtLastDataRateMsg
            if eventsSinceLastDataRateMsg > 1200: # about 10 seconds of data at 120hz
                curTime = time.time()
                dataRateHz = eventsSinceLastDataRateMsg/(curTime-timeAtLastDataRateMsg)
                self.logger.info("Current data rate is %.2f Hz. %d events processed" % (dataRateHz, self.numEvents))
                timeAtLastDataRateMsg = curTime
                numEventsAtLastDataRateMsg = self.numEvents
        assert noData == False, "There was no data in master loop"
        self.logger.info('master waited for ready servers %.2f ms per each time. Did %.2f waits per event' % ((waitAllTime*1000.0)/waitAllNum, waitAllNum/max(1,self.numEvents)))

        # one last datarate msg
        dataRateHz = self.numEvents/(time.time()-startTime)
        self.logger.info("Overall data rate is %.2f Hz. Number of events is %d" % (dataRateHz, self.numEvents))

        # send one last update at the end
        self.logger.debug("CommSystem: servers finished. sending one last update")
        self.informViewerOfUpdate(sec, nsec, fiducials, counter)
        self.informWorkersToUpdateViewer()

        self.sendEndToWorkers()
        self.sendEndToViewer()

class RunWorker(object):
    def __init__(self, masterWorkersComm, masterRankInMasterWorkersComm, 
                 wrapEventNumber, xCorrBase, logger, isFirstWorker):
        self.masterWorkersComm = masterWorkersComm
        self.masterRankInMasterWorkersComm = masterRankInMasterWorkersComm
        self.wrapEventNumber = wrapEventNumber
        self.xCorrBase = xCorrBase
        self.logger = logger
        self.isFirstWorker = isFirstWorker
        self.msgBuffer = MVW_MsgBuffer()
        self.evtNumber = 0
        self.wrapped = False

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def workerWaitForMasterBcastWrapped(self):
        self.workerWaitForMasterBcast()

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def workerWaitForMasterBcastNotWrapped(self):
        self.workerWaitForMasterBcast()

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def workerWaitForMasterBcast(self):
        self.masterWorkersComm.Bcast([self.msgBuffer.getNumpyBuffer(),
                                      self.msgBuffer.getMPIType()],
                                     root=self.masterRankInMasterWorkersComm)
        self.masterWorkersComm.Barrier()

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def serverWorkersScatterWrapped(self, serverWorldRank):
        self.xCorrBase.serverWorkersScatter(detectorData1Darray=None,
                                     serverWorldRank = serverWorldRank)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def serverWorkersScatterNotWrapped(self, serverWorldRank):
        self.xCorrBase.serverWorkersScatter(detectorData1Darray=None,
                                     serverWorldRank = serverWorldRank)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def storeNewWorkerDataWrapped(self, counter):
        self.xCorrBase.storeNewWorkerData(counter = counter)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def storeNewWorkerDataNotWrapped(self, counter):
        self.xCorrBase.storeNewWorkerData(counter = counter)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def viewerWorkersUpdateWrapped(self, lastTime):
        self.xCorrBase.viewerWorkersUpdate(lastTime = lastTime)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def viewerWorkersUpdateNotWrapped(self, lastTime):
        self.xCorrBase.viewerWorkersUpdate(lastTime = lastTime)

    def run(self):
        lastTime = {'sec':0, 'nsec':0, 'fiducials':0, 'counter':0}
        numEvents = 0
        while True:
            numEvents += 1
            if numEvents % 1201 == 1200:
                pass
            self.logger.debug("CommSystem.run: before Bcast from master")
            if self.wrapped:
                self.workerWaitForMasterBcastWrapped()
            else:
                self.workerWaitForMasterBcastNotWrapped()

            if self.msgBuffer.isEvt():
                serverWithData = self.msgBuffer.getRank()
                lastTime = self.msgBuffer.getTime()
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("CommSystem.run: after Bcast from master. EVT server=%2d counter=%d" % \
                                      (serverWithData, lastTime['counter']))
                if self.wrapped:
                    self.serverWorkersScatterWrapped(serverWorldRank = serverWithData)
#                    if self.isFirstWorker: self.logger.info("calling storeNewWorkerDataWrapped: from server=%d counter=%d sec=0x%8.8X nsec=0x%8.8X fid=0x%5.5X" % \
#                                    (serverWithData, lastTime['counter'], lastTime['sec'], lastTime['nsec'], lastTime['fiducials']))
                    self.storeNewWorkerDataWrapped(counter = lastTime['counter'])
                else:
                    self.serverWorkersScatterNotWrapped(serverWorldRank = serverWithData)
#                    if self.isFirstWorker: self.logger.info("calling storeNewWorkerDataNotWrapped: from server=%d counter=%d sec=0x%8.8X nsec=0x%8.8X fid=0x%5.5X" % \
#                                    (serverWithData, lastTime['counter'], lastTime['sec'], lastTime['nsec'], lastTime['fiducials']))
                    self.storeNewWorkerDataNotWrapped(counter = lastTime['counter'])

            elif self.msgBuffer.isUpdate():
                self.logger.debug("CommSystem.run: after Bcast from master - UPDATE")
                if self.wrapped:
                    self.viewerWorkersUpdateWrapped(lastTime = lastTime)
                else:
                    self.viewerWorkersUpdateNotWrapped(lastTime = lastTime)
                self.logger.debug("CommSystem.run: returned from viewer workers update")
            elif self.msgBuffer.isEnd():
                self.logger.debug("CommSystem.run: after Bcast from master - END. quiting")
                break
            else:
                raise Exception("unknown msgtag")
            self.evtNumber += 1
            if self.evtNumber >= self.wrapEventNumber:
                self.wrapped = True

class RunViewer(object):
    def __init__(self, worldComm, masterRank, xCorrBase, logger):
        self.worldComm = worldComm
        self.masterRank = masterRank
        self.logger = logger
        self.xCorrBase = xCorrBase
        self.msgbuffer = MVW_MsgBuffer()

    @Timing.timecall(VIEWEREVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def waitForMasterMessage(self):
        self.worldComm.Recv([self.msgbuffer.getNumpyBuffer(),
                   self.msgbuffer.getMPIType()],
                  source=self.masterRank)

    @Timing.timecall(VIEWEREVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def viewerWorkersUpdate(self, lastTime):
        self.xCorrBase.viewerWorkersUpdate(lastTime)

    def run(self):
        while True:
            self.logger.debug('CommSystem.run: before Recv from master')
            self.waitForMasterMessage()

            if self.msgbuffer.isUpdate():
                lastTime = self.msgbuffer.getTime()
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug('CommSystem.run: after Recv from master. get UPDATE: counter=%d' % lastTime['counter'])
                self.viewerWorkersUpdate(lastTime = lastTime)
            elif self.msgbuffer.isEnd():
                self.logger.debug('CommSystem.run: after Recv from master. get END. quiting.')
                break
            else:
                raise Exception("unknown msgtag")
        self.xCorrBase.shutdown_viewer()

def runTestAlt(mp, xCorrBase):
    xCorrBase.serverInit()
    xCorrBase.workerInit()
    xCorrBase.viewerInit()
    eventIter = xCorrBase.makeEventIter()
    eventIds = []
    allData = []
    mp.logInfo("Starting to read through data for test_alt")
    for datum in eventIter.dataGenerator():
        maskedData = datum.dataArray[mp.maskNdarrayCoords]
        maskedData = maskedData.flatten().copy()
        xCorrBase.userObj.workerAdjustData(maskedData)

        eventIds.append((datum.sec, datum.nsec, datum.fiducials))
        allData.append(maskedData)

    mp.logInfo("read through data for test_alt")
    sortedCounters, newDataOrder = PsanaUtil.getSortedCountersBasedOnSecNsecAtHertz(eventIds, 120)
    
    if not np.all(newDataOrder==np.sort(newDataOrder)):
        mp.logWarning("DAQ data did not come in sorted order.")
    
    eventIdNumpyDtype = np.dtype([('sec',np.int32),
                                  ('nsec',np.int32),
                                  ('fiducials',np.int32),
                                  ('counter',np.int64)])
    sortedData = np.zeros((len(allData),len(allData[0])), dtype=allData[0].dtype)
    sortedEventIds = np.zeros(len(allData), dtype=eventIdNumpyDtype)
    for idx,sortedPos in enumerate(newDataOrder):
        sortedData[idx,:]=allData[sortedPos][:]
        sortedEventIds[idx]['counter'] = sortedCounters[idx]
        eventIdInSortOrder = eventIds[sortedPos]
        sortedEventIds[idx]['sec'] = eventIdInSortOrder[0]
        sortedEventIds[idx]['nsec'] = eventIdInSortOrder[1]
        sortedEventIds[idx]['fiducials'] = eventIdInSortOrder[2]
    if xCorrBase.h5file is not None:
        testGroup = xCorrBase.h5file.create_group('test')
        testGroup['detectorEventIds'] = sortedEventIds
        testGroup['detectorData'] = sortedData
        
    xCorrBase.userObj.calcAndPublishForTestAlt(sortedEventIds, sortedData, xCorrBase.h5GroupUser)
    xCorrBase.shutdown_viewer()
    return 0

def runCommSystem(mp, updateInterval, wrapEventNumber, xCorrBase, hostmsg, test_alt):
    '''main driver for the system. 

    ARGS:
      mp - instance of MPI_Communicators
      updateInterval (int): 
      wrapEventNumber (int):
      xCorrBase:
      hostmsg: 
      test_alt (bool): True if this is testing mode
    '''
    if test_alt:
        return runTestAlt(mp, xCorrBase)

    logger = mp.logger
    reportTiming = False
    timingNode = ''
    try:
        if mp.isServer:            
            xCorrBase.serverInit()
            eventIter = xCorrBase.makeEventIter()
            runServer = RunServer(eventIter, xCorrBase,
                                  mp.comm, mp.rank, mp.masterRank, xCorrBase.logger)
            runServer.run()
            reportTiming = True
            timingNode = 'SERVER'

        elif mp.isMaster:
            runMaster = RunMaster(mp.comm, mp.masterRank, mp.viewerRank, mp.serverRanks, 
                                  mp.masterWorkersComm, mp.masterRankInMasterWorkersComm,
                                  updateInterval, hostmsg, logger)
            runMaster.run()
            reportTiming = True
            timingNode = 'MASTER'

        elif mp.isViewer:
            xCorrBase.viewerInit()
            runViewer = RunViewer(mp.comm, mp.masterRank, xCorrBase, logger)
            runViewer.run()
            reportTiming = True
            timingNode = 'VIEWER'

        elif mp.isWorker:
            xCorrBase.workerInit()
            runWorker = RunWorker(mp.masterWorkersComm, mp.masterRankInMasterWorkersComm,
                                  wrapEventNumber, xCorrBase, logger, mp.isFirstWorker)
            runWorker.run()
            if mp.isFirstWorker:
                reportTiming = True
                timingNode = 'FIRST WORKER'
                
        else:
            raise Exception("rank is neither server/master/viewer or worker - internal error")

    except Exception:
        exceptBuffer = StringIO.StringIO()
        traceback.print_exc(file=exceptBuffer)
        logger.error('encountered exception: %s' % exceptBuffer.getvalue())
        MPI.COMM_WORLD.Abort(1)
        return -1

    if reportTiming:
        hdr = '--BEGIN %s TIMING--' % timingNode
        footer = '--END %s TIMING--' % timingNode
        Timing.reportOnTimingDict(logger,hdr, footer,
                                  timingDict=timingdict, keyOrder=timingorder)
    return 0

def isNoneOrListOfStrings(arg):
    def isListOfStrings(arg):
        if not isinstance(arg, list):
            return False
        def isStr(x): return isinstance(x,str)
        return all(map(isStr,arg))
    if arg is None:
        return True
    return isListOfStrings(arg)

class CommSystemFramework(object):
    def __init__(self, system_params, user_params, test_alt=False):
        CommSystemUtil.checkParams(system_params, user_params)
        numServers = int(system_params['numServers'])
        dataset = system_params['dataset']
        serverHosts = system_params['serverHosts']
        assert isNoneOrListOfStrings(serverHosts), "system_params['serverHosts'] is neither None or a list of str"

        self.test_alt = test_alt

        if test_alt:
            assert MPI.COMM_WORLD.size == 1, "In test_alt mode, do not run in MPI mode"
            hostmsg = "test mode - no host assignent"
            mp = getTestingMPIObject()
        else:
            serverRanks, hostmsg = identifyServerRanks(MPI.COMM_WORLD,
                                                       numServers,
                                                       serverHosts)
            # set mpi paramemeters for framework
            mp = identifyCommSubsystems(serverRanks=serverRanks, worldComm=MPI.COMM_WORLD)

        self.hostmsg = hostmsg
        verbosity = system_params['verbosity']
        mp.setLogger(verbosity)
        maskNdarrayCoords_Filename = system_params['maskNdarrayCoords']
        assert os.path.exists(maskNdarrayCoords_Filename), "mask file %s not found" % maskNdarrayCoords_Filename
        maskNdarrayCoords = np.load(maskNdarrayCoords_Filename).astype(np.int8)
        mp.setMask(maskNdarrayCoords)
        srcString = system_params['src']
        numEvents = system_params['numEvents']
        maxTimes = system_params['times']
        assert isinstance(srcString,str), "system parameters src is not a string"
        assert isinstance(numEvents,int), "system parameters numevents is not an int"
        assert isinstance(maxTimes,int), "system parameters maxTimes is not an int"

        xcorrBase = XCorrBase(mp, 
                              dataset, 
                              srcString, 
                              numEvents, 
                              maxTimes, 
                              system_params, 
                              user_params,
                              test_alt)
        self.mp = mp
        self.xcorrBase = xcorrBase
        self.maxTimes = maxTimes
        self.updateInterval = system_params['update']
        
    def run(self):
        return runCommSystem(self.mp, self.updateInterval, self.maxTimes, self.xcorrBase, self.hostmsg, self.test_alt)

