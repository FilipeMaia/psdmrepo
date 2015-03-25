### system
from mpi4py import MPI
import os
import numpy as np
import time
import StringIO
import traceback
import copy
import collections

## this package
import CommSystemUtil
import PsanaUtil
from MessageBuffers import SM_MsgBuffer, MVW_MsgBuffer
import Timing
from XCorrBase import XCorrBase

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
    '''returns ranks to be the servers, defaults to maximimize distinct hosts/nodes.

    Server ranks will be picked in a round robin fashion among the distinct hosts among
    in the MPI world.

    Args:
      comm (MPI.Comm): communicator from whith mpi world hostnames are identified.
                       *IMPORTANT* This function may do blocking collective communication 
                       with comm. All ranks in comm should call this during initialization.
                       best to call this function during the driver program for the system.
      numServers (int): number of servers to find
      serverHosts (list, optional): None or empty means use default host assignment. Otherwise 
                                    list must be a set of unique hostnames. 
    ::

      Returns:
        servers (dict): a list of ints, the ranks in the comm to use as servers
        hostmsg (str): a logging string about the hosts chosen.

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

    Call identifyCommSubsystems to return a initialized instance.
    Then call the method setMask.
    '''
    def __init__(self):
        pass

    def setLogger(self, verbosity):
        self.logger = CommSystemUtil.makeLogger(self.isMaster, \
                        self.isViewer, self.isServer, self.rank, verbosity)

    def notWorker2toN(self):
        if self.isWorker and not self.isFirstWorker:
            return False
        return True

    # these functions make it simpler to log only if you are not a later worker.
    # if all the workers log messages it gets noisy.
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


    def setMask(self, mask_ndarrayCoords):
        '''sets scatterv parameters and stores mask

        Args:
          mask_ndarrayCoords (numpy.ndarray): integer array, 1 for elements that should be processed. 
                                              It must have the same shape as the NDArray for the detector.

        Notes:
          sets the following attributes

          * totalElements:                 number of pixels that are 1 in the mask
          * workerWorldRankToCount[rank]:  number of element that worker processes
          * workerWorldRankToOffset[rank]: offset of where those elements start in 
                                           a flattened version of the ndarray
          * mask_ndarrayCoords:            the mask as a logical True/False array
                                           shape has not been changed

        '''
        mask_flat = mask_ndarrayCoords.flatten()
        maskValues = set(mask_flat)
        assert maskValues.union(set([0,1])) == set([0,1]), "mask contains values other than 0 and 1." + \
            (" mask contains %d distinct values" % len(maskValues))
        assert 1 in maskValues, "The mask does not have the value 1, it is all 0. Elements marked with 1 are processed"
        self.totalElements = np.sum(mask_flat)
        self.mask_ndarrayCoords = mask_ndarrayCoords == 1

        self.logDebug("MPIParams.setMask: loaded and stored mask with shape=%s elements included=%d excluded=%s" % \
                      (self.mask_ndarrayCoords.shape, np.sum(self.mask_ndarrayCoords),
                       np.sum(0==self.mask_ndarrayCoords)))

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
    mc.masterRank = min(availRanks)
    availRanks.remove(mc.masterRank)
    mc.viewerRank = min(availRanks)
    availRanks.remove(mc.viewerRank)
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
    def __init__(self,dataIter, comm, rank, masterRank, logger):
        self.dataIter = dataIter
        self.comm = comm
        self.rank = rank
        self.masterRank = masterRank
        self.logger = logger

    def recordTimeToGetData(self, startTime, endTime):
        global timingdict
        global timingorder

        if startTime is None: return
        key = 'ServerTimeToGetData'
        if key not in timingdict:
            timingdict[key]=[0.0, 0, 'event', 1e-3, 'ms']
            timingorder.append(key)
        timingdict[key][0] += endTime-startTime
        timingdict[key][1] += 1

    def run(self):
        sendEventReadyBuffer = SM_MsgBuffer(rank=self.rank)
        sendEventReadyBuffer.setEvt()
        receiveOkForWorkersBuffer = SM_MsgBuffer(rank=self.rank)
        abortFromMaster = False

        dataGen = self.dataIter.dataGenerator()

        t0 = None
        for datum in dataGen:
            self.recordTimeToGetData(startTime=t0, endTime=time.time())
            sec, nsec = datum.time()
            sendEventReadyBuffer.setTime(sec, nsec)
            self.logger.debug("CommSystem.run: Before Send EVT sec=%d nsec=%d" % (sec, nsec))
            self.comm.Send([sendEventReadyBuffer.getNumpyBuffer(),
                            sendEventReadyBuffer.getMPIType()],
                           dest=self.masterRank)
            self.logger.debug("CommSystem.run: After Send, before Recv")
            self.comm.Recv([receiveOkForWorkersBuffer.getNumpyBuffer(), 
                            receiveOkForWorkersBuffer.getMPIType()],
                           source=self.masterRank)
            if receiveOkForWorkersBuffer.isSendToWorkers():
                self.dataIter.sendToWorkers(datum)
                self.logger.debug("CommSystem.run: After Recv. Send to workers")

            elif receiveOkForWorkersBuffer.isAbort():
                self.logger.debug("CommSystem.run: After Recv. Abort")
                abortFromMaster = True
                break
            else:
                raise Exception("unknown msgtag from master. buffer=%r" % receiveOkForWorkersBuffer)
            t0 = time.time()

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

        self.firstSec = None
        # initially all servers are not ready
        self.notReadyServers = [r for r in serverRanks]  # MPI Test on request is false
        self.readyServers = []                           # MPI Test on request is True
        self.finishedServers = []                        # rank has returend end

        self.sendOkForWorkersBuffer = SM_MsgBuffer()
        self.bcastWorkersBuffer = MVW_MsgBuffer()
        self.viewerBuffer = MVW_MsgBuffer()
        self.lastUpdate = 0
        self.numEvents = 0

    def getEarliest(self, serverDataList):
        '''Takes a list of server data buffers. identifies oldest server::
        ARGS:
          serverDataList:  each element is a SeversMasterMessaging buffer
        RET:
         the buffer with the earliest time
        '''
        idx = 0
        sec, nsec = serverDataList[idx].getTime()
        for curIdx in range(1,len(serverDataList)):
            curSec, curNsec = serverDataList[idx].getTime()
            if (curSec < sec) or ((curSec == sec) and (curNsec < nsec)):
                sec = curSec
                nsec = curNsec
                idx = curIdx
        return serverDataList[idx]
        
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
    def informWorkersOfNewData(self, selectedServerRank, relTime):
        self.bcastWorkersBuffer.setEvt()
        self.bcastWorkersBuffer.setRank(selectedServerRank)
        self.bcastWorkersBuffer.setRelSec(relTime)
        self.logger.debug("CommSystem: before Bcast -> workers EVT at %.5f" % relTime)
        self.masterWorkersComm.Bcast([self.bcastWorkersBuffer.getNumpyBuffer(),
                                      self.bcastWorkersBuffer.getMPIType()],
                                     root=self.masterRankInMasterWorkersComm)
        self.masterWorkersComm.Barrier()
        self.logger.debug("CommSystem: after Bcast/Barrier -> workers EVT at %.5f" % relTime)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def informViewerOfUpdate(self, relSec):
        self.viewerBuffer.setUpdate()
        self.viewerBuffer.setRelSec(relSec)
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
        def waitOnServer(self):
            '''called during communication loop. 
            Called when no ready servers.
            identifies a done server through waitany
            tests all done servers
            idenfies finisned servers among done servers

            at the end of this function,

            self.notReadyServers + self.finishedServers + self.readyServers 

            will be the same as it was on entry, and at least one of the notReadyServers will
            be moved into the finishedServers or readyServers group.
            '''
            assert len(self.notReadyServers)>0, "waitOnServer called, but no not-ready servers"

            self.logger.debug("CommSystem: before waitany. notReadyServers=%s" % self.notReadyServers)
            requestList = [serverRequests[rnk] for rnk in self.notReadyServers]
            idx=MPI.Request.Waitany(requestList)
            self.logger.debug("CommSystem: after waitany. server %d is now ready" % self.notReadyServers[idx])

            newReadyServers = [server for server in self.notReadyServers \
                               if serverRequests[server].Test()]
            newFinishedServers = [server for server in newReadyServers \
                                  if serverReceiveData[server].isEnd()]
            self.finishedServers.extend(newFinishedServers)
            for server in newFinishedServers:
                # take finsished servers out of pool that we wait for a request from
                self.notReadyServers.remove(server)
                # take finished servers out of pool to get next event from
                newReadyServers.remove(server)
            for server in newReadyServers:
                self.notReadyServers.remove(server)
            self.readyServers.extend(newReadyServers)
        ############ end helper functions ######

        serverReceiveData, serverRequests = self.initRecvRequestsFromServers()

        relTime = 0.0
        numEventsAtLastDataRateMsg = 0
        timeAtLastDataRateMsg = time.time()
        startTime = time.time()
        while True:
            # a server must be in one of: ready, noReady or finished
            serversAccountedFor = len(self.readyServers) + len(self.finishedServers) + \
                                  len(self.notReadyServers)
            assert serversAccountedFor == len(self.serverRanks), \
                "loop invariant broken? #servers=%d != #accountedfor=%d" % \
                (len(self.serverRanks), len(serversAccountedFor))

            if len(self.finishedServers)==len(self.serverRanks): 
                break

            if len(self.readyServers)==0:
                waitOnServer(self)

            if len(self.readyServers)==0:
                # the server we waited on was finished. 
                continue

            earlyServerData = self.getEarliest([serverReceiveData[server] for server in self.readyServers])
            selectedServerRank = earlyServerData.getRank()
            sec, nsec = earlyServerData.getTime()
            self.logger.debug("CommSystem: next server rank=%d sec=%d nsec=%10d" % (selectedServerRank, sec, nsec))
            if self.firstSec is None:
                self.firstSec = sec

            relTime = float(sec-self.firstSec)+1e-9*nsec
            self.informWorkersOfNewData(selectedServerRank, relTime)

            # tell server n to scatter to workers
            self.sendOkForWorkersBuffer.setSendToWorkers()
            self.logger.debug("CommSystem: before SendOkForWorkers to server %d" % selectedServerRank)
            self.worldComm.Send([self.sendOkForWorkersBuffer.getNumpyBuffer(), 
                                         self.sendOkForWorkersBuffer.getMPIType()], 
                                        dest=selectedServerRank)

            # do new Irecv from worker n
            self.readyServers.remove(selectedServerRank)
            self.notReadyServers.append(selectedServerRank)
            self.logger.debug("CommSystem: after sendOk, before replacing request with Irecv from rank %d" % selectedServerRank)
            serverReceiveBuffer = serverReceiveData[selectedServerRank]
            serverRequests[selectedServerRank] = self.worldComm.Irecv([serverReceiveBuffer.getNumpyBuffer(), 
                                                                               serverReceiveBuffer.getMPIType()],  \
                                                                              source = selectedServerRank)
            self.logger.debug("CommSystem: after Irecv from rank %d" % selectedServerRank)

            # check to see if there should be an update for the viewer
            self.numEvents += 1
            if (self.updateIntervalEvents > 0) and (self.numEvents - self.lastUpdate > self.updateIntervalEvents):
                self.lastUpdate = self.numEvents
                self.logger.debug("CommSystem: Informing viewers and workers to update" )
                self.informViewerOfUpdate(relTime)
                self.informWorkersToUpdateViewer()

            # check to display message
            eventsSinceLastDataRateMsg = self.numEvents - numEventsAtLastDataRateMsg
            if eventsSinceLastDataRateMsg > 2400: # about 20 seconds of data at 120hz
                curTime = time.time()
                dataRateHz = eventsSinceLastDataRateMsg/(curTime-timeAtLastDataRateMsg)
                self.logger.info("Current data rate is %.2f Hz. %d events processed" % (dataRateHz, self.numEvents))
                timeAtLastDataRateMsg = curTime
                numEventsAtLastDataRateMsg = self.numEvents

        # one last datarate msg
        dataRateHz = self.numEvents/(time.time()-startTime)
        self.logger.info("Overall data rate is %.2f Hz. Number of events is %d" % (dataRateHz, self.numEvents))

        # send one last update at the end
        self.logger.debug("CommSystem: servers finished. sending one last update")
        self.informViewerOfUpdate(relTime)
        self.informWorkersToUpdateViewer()

        self.sendEndToWorkers()
        self.sendEndToViewer()

class RunWorker(object):
    def __init__(self, masterWorkersComm, masterRankInMasterWorkersComm, 
                 wrapEventNumber, g2, logger):
        self.masterWorkersComm = masterWorkersComm
        self.masterRankInMasterWorkersComm = masterRankInMasterWorkersComm
        self.wrapEventNumber = wrapEventNumber
        self.g2 = g2
        self.logger = logger
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
        self.g2.serverWorkersScatter(serverFullDataArray=None,
                                     serverWorldRank = serverWorldRank)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def serverWorkersScatterNotWrapped(self, serverWorldRank):
        self.g2.serverWorkersScatter(serverFullDataArray=None,
                                     serverWorldRank = serverWorldRank)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def storeNewWorkerDataWrapped(self, relSeconds):
        self.g2.storeNewWorkerData(relSeconds = relSeconds)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def storeNewWorkerDataNotWrapped(self, relSeconds):
        self.g2.storeNewWorkerData(relSeconds = relSeconds)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def viewerWorkersUpdateWrapped(self, relSeconds):
        self.g2.viewerWorkersUpdate(relsec = relSeconds)

    @Timing.timecall(CSPADEVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def viewerWorkersUpdateNotWrapped(self, relSeconds):
        self.g2.viewerWorkersUpdate(relsec = relSeconds)

    def run(self):
        relSeconds = 0.0
        while True:
            self.logger.debug("CommSystem.run: before Bcast from master")
            if self.wrapped:
                self.workerWaitForMasterBcastWrapped()
            else:
                self.workerWaitForMasterBcastNotWrapped()

            if self.msgBuffer.isEvt():
                serverWithData = self.msgBuffer.getRank()
                relSeconds = self.msgBuffer.getRelSec()
                self.logger.debug("CommSystem.run: after Bcast from master. EVT server=%2d time=%.5f" % (serverWithData, relSeconds))
                if self.wrapped:
                    self.serverWorkersScatterWrapped(serverWorldRank = serverWithData)
                    self.storeNewWorkerDataWrapped(relSeconds = relSeconds)
                else:
                    self.serverWorkersScatterNotWrapped(serverWorldRank = serverWithData)
                    self.storeNewWorkerDataNotWrapped(relSeconds = relSeconds)

            elif self.msgBuffer.isUpdate():
                self.logger.debug("CommSystem.run: after Bcast from master - UPDATE")
                if self.wrapped:
                    self.viewerWorkersUpdateWrapped(relSeconds = relSeconds)
                else:
                    self.viewerWorkersUpdateNotWrapped(relSeconds = relSeconds)
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
    def __init__(self, worldComm, masterRank, g2, logger):
        self.worldComm = worldComm
        self.masterRank = masterRank
        self.logger = logger
        self.g2 = g2
        self.msgbuffer = MVW_MsgBuffer()

    @Timing.timecall(VIEWEREVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def waitForMasterMessage(self):
        self.worldComm.Recv([self.msgbuffer.getNumpyBuffer(),
                   self.msgbuffer.getMPIType()],
                  source=self.masterRank)

    @Timing.timecall(VIEWEREVT, timingDict=timingdict, timingDictInsertOrder=timingorder)
    def viewerWorkersUpdate(self, relsec):
        self.g2.viewerWorkersUpdate(relsec=relsec)

    def run(self):
        while True:
            self.logger.debug('CommSystem.run: before Recv from master')
            self.waitForMasterMessage()

            if self.msgbuffer.isUpdate():
                relsec = self.msgbuffer.getRelSec()
                self.logger.debug('CommSystem.run: after Recv from master. get UPDATE: relsec=%.3f' % relsec)
                self.viewerWorkersUpdate(relsec=relsec)
            elif self.msgbuffer.isEnd():
                self.logger.debug('CommSystem.run: after Recv from master. get END. quiting.')
                break
            else:
                raise Exception("unknown msgtag")
        self.g2.shutdown_viewer()

def runCommSystem(mp, updateInterval, wrapEventNumber, xCorrBase, hostmsg):
    '''main driver for the system. 
    ARGS:
    mp - a 
    updateInterval - 
    wrapEventNumber - 
    xCorrBase -

    This
    '''
    logger = mp.logger
    reportTiming = False
    timingNode = ''
    try:
        if mp.isServer:            
            xCorrBase.initServer()
            eventIter = xCorrBase.makeEventIter()
            runServer = RunServer(eventIter, 
                                  mp.comm, mp.rank, mp.masterRank, logger)
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
            xCorrBase.initViewer()
            runViewer = RunViewer(mp.comm, mp.masterRank, xCorrBase, logger)
            runViewer.run()
            reportTiming = True
            timingNode = 'VIEWER'

        elif mp.isWorker:
            xCorrBase.initWorker()
            runWorker = RunWorker(mp.masterWorkersComm, mp.masterRankInMasterWorkersComm,
                                  wrapEventNumber, xCorrBase, logger)
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

    if reportTiming:
        hdr = '--BEGIN %s TIMING--' % timingNode
        footer = '--END %s TIMING--' % timingNode
        Timing.reportOnTimingDict(logger,hdr, footer,
                                  timingDict=timingdict, keyOrder=timingorder)


 
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
    def __init__(self, system_params, user_params):
        numservers = int(system_params['numservers'])
        dataset = system_params['dataset']
        serverhosts = system_params['serverhosts']
        assert isNoneOrListOfStrings(serverhosts), "system_params['serverhosts'] is neither None or a list of str"
        serverRanks, hostmsg = identifyServerRanks(MPI.COMM_WORLD,
                                                   numservers,
                                                   serverhosts)
        self.hostmsg = hostmsg
        # set mpi paramemeters for framework
        mp = identifyCommSubsystems(serverRanks=serverRanks, worldComm=MPI.COMM_WORLD)
        verbosity = system_params['verbosity']
        mp.setLogger(verbosity)
        mask_ndarrayCoords_Filename = system_params['mask_ndarrayCoords']
        assert os.path.exists(mask_ndarrayCoords_Filename), "mask file %s not found" % mask_ndarrayCoords_Filename
        mask_ndarrayCoords = np.load(mask_ndarrayCoords_Filename)
        mp.setMask(mask_ndarrayCoords)
        srcString = system_params['src']
        numEvents = system_params['numevents']
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
                              user_params)
        self.mp = mp
        self.xcorrBase = xcorrBase
        self.maxTimes = maxTimes
        self.updateInterval = system_params['update']
        
    def run(self):
        runCommSystem(self.mp, self.updateInterval, self.maxTimes, self.xcorrBase, self.hostmsg)

