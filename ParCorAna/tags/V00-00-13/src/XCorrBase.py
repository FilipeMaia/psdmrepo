from mpi4py import MPI
import numpy as np
import os
import time
import shutil
import StringIO
import pprint
import logging

import h5py

from ParCorAna.WorkerData import WorkerData
import ParCorAna.Timing as Timing
import ParCorAna.CommSystemUtil as CommSystemUtil
import ParCorAna as corAna
from EventIter import EventIter

#################################
# utitlity functions users may use
def makeDelayList(start, stop, num, spacing, logbase=np.e):
    '''Constructs a list of integer delays with either linear or log spacing

    Args:
      start (int):   starting delay
      stop (int):    last delay to include
      num (int):     number of delays to return, including start and stop
      spacing (str): either 'lin' or 'log' for linear or logarthmic
      logbase (float, optional): if 'log', what base to use. Default is e

    Return:
      list of distinct delays, as array of int32. 
      Always returns a list of unique integer values. Makes a best effort to 
      hit the number of user requested delays.
    '''
    ####### helper function #######
    def getUniqueRoundedIntegerDelays(start, stop, numToMakeUserNum, spacing, logbase):
        if spacing == 'log':
            logStart = np.log(start)
            logStop = np.log(stop)
            delays = np.logspace(start=logStart, stop=logStop, num=numToMakeUserNum, endpoint=True, base=logbase)
        else:
            delays = np.linspace(start, stop, numToMakeUserNum)
        delays = list(set(np.round(delays).astype(np.int32)))
        delays.sort()
        return np.array(delays, dtype=np.int32)

    ###### end helper function #####

    assert spacing in ['log','lin'], "spacing must be 'log' or 'lin'"
    assert stop >= start
    assert isinstance(num,int), "num parameter must be an integer"
    assert num > 0
    assert start > 0
    if num >= (stop-start+1):
        return np.array(range(start,stop+1),dtype=np.int32)
    candDelays = getUniqueRoundedIntegerDelays(start, stop, num, spacing, logbase)
    if len(candDelays)==num:
        return candDelays
    assert len(candDelays) < num, "rounding to integers and taking unique values should have produced fewer values"
    delaysLow = candDelays
    del candDelays
    numLow = num
    numHigh = 2*num
    delaysHigh = getUniqueRoundedIntegerDelays(start, stop, numHigh, spacing, logbase)
    while len(delaysHigh) < num:
        print "numHigh=%d len(delaysHigh)=%d" % (numHigh, len(delaysHigh))
        numHigh = 2*numHigh
        delaysHigh = getUniqueRoundedIntegerDelays(start, stop, numHigh, spacing, logbase)
    if len(delaysHigh)==num:
        return delaysHigh
    # try binary search to get user requested number of delays. I'm not positive that the
    # number of delays returned is monotonic with the number of delays requested, 
    # so stop after a certain number of iterations to prevent an infinite loop (and just
    # return however many delays we are generating, should be close to what was requested).
    maxIter = 20
    curIter = 1
    candNum = (numLow + numHigh)/2
    candDelays = getUniqueRoundedIntegerDelays(start, stop, candNum, spacing, logbase)
    while (len(candDelays) != num) and (curIter <= maxIter):
#        print "iter=%3d low=%3d high=%3d cand=%3d candDelays=%3d" % (curIter, numLow, numHigh, candNum, len(candDelays))
        curIter += 1
        if len(candDelays) > num:
            numHigh = candNum
        else:
            numLow = candNum
        candNum = (numLow + numHigh)/2
        candDelays = getUniqueRoundedIntegerDelays(start, stop, candNum, spacing, logbase)
    return candDelays
            


def writeToH5Group(h5Group, name2delay2ndarray):
    '''This writes the name2delay2ndarray 2D dict that the user module viewerPublish receives
    to an hdf5 group. It creates several subgroups:

    h5Group/name/delay/ndarray

    That is for each pair that indexes the name2delay2ndarray 2D dict, we write a ndarray.
    '''
    for nm, delay2ndarrayDict in name2delay2ndarray.iteritems():
        nmGroup = h5Group.create_group(nm)
        for delay, ndarray in delay2ndarrayDict.iteritems():
            dataSetName = 'delay_%6.6d' % delay
            dataSetShape = ndarray.shape
            dataSetDType = ndarray.dtype
            delayDataset = nmGroup.create_dataset(dataSetName, dataSetShape, dataSetDType)
            delayDataset[:] = ndarray[:]

def writeConfig(h5file, system_params, user_params):
    if 'system' in h5file.keys():
        h5Group = h5file['system']
    else:
        h5Group = h5file.create_group('system')
    systemParamsGroup = h5Group.create_group('system_params')
    userParamsGroup = h5Group.create_group('user_params')
    for configDict, h5Group in zip([system_params, user_params],
                                   [systemParamsGroup, userParamsGroup]):
        configKeys = configDict.keys()
        configKeys.sort()
        for key in configKeys:
            if key in ['maskNdarrayCoords', 'colorNdarrayCoords']:
                filename = system_params['maskNdarrayCoords']
                assert os.path.exists(filename), "file %s doesn't exist" % filename
                numpyArray = np.load(file(filename,'r'))
                h5Group[key]=numpyArray
            else:
                value = configDict[key]
                try:
                    h5Group[key]=value
                except:
                    h5Group[key]=str(value)
    
    return h5Group

###############################
class XCorrBase(object):
    def __init__(self, mp, dataSourceString, srcString,
                 numEvents, maxTimes, system_params, user_params, test_alt):
        self.dataSourceString = dataSourceString
        self.srcString = srcString
        self.numEvents = numEvents
        self.maxTimes = maxTimes
        self.system_params = system_params
        self.user_params = user_params
        self.mp = mp
        self.logger = mp.logger
        self.isServerOrFirstWorker = self.mp.isServer or self.mp.isFirstWorker
        self.isViewerOrFirstWorker = self.mp.isViewer or self.mp.isFirstWorker
        self.delays = self.system_params['delays']
        self.numDelays = len(self.delays)
        userClass = system_params['userClass']
        self.userObj = userClass(user_params, system_params, mp, test_alt)
        self.arrayNames = self.userObj.fwArrayNames()
        self.totalMaskedElements = np.sum(self.mp.maskNdarrayCoords)
        self.test_alt = test_alt

    def runTestAlt(self):
        self.userObj.runTestAlt()

    def makeEventIter(self):
        self.logger.debug('XCorrBase.makeEventIter returning EventIter(datasource=%s,rank=%s,servers=%s,numEvents=%s)' % \
                          (self.dataSourceString,self.mp.rank,self.mp.serverRanks,self.numEvents))
        return EventIter(self.dataSourceString,
                         self.mp.rank,
                         self.mp.serverRanks,
                         self.userObj,
                         self.system_params,
                         self.mp.maskNdarrayCoords.shape,
                         self.mp.logger,
                         self.numEvents)

    def serverWorkersScatter(self, detectorData1Darray = None, serverWorldRank = None):
        '''called from both server and worker ranks for the scattering of the data.

        When called from the server, args are
        detectorData1Darray - ndarray of float32, 1D, already masked out
        serverWorldRank     - must be None.

        When called from the worker, args are
        detectorData1Darray  - None
        serverWorldRank      - the server rank, as received from the master in EVT message
        '''
        if serverWorldRank is None:
            assert self.mp.isServer, "XCorrBase.serverWorkersScatter - no serverRank passed but not called as Server"
            serverWorldRank = self.mp.rank
        serverWorkersDict = self.mp.serverWorkers[serverWorldRank]
        counts = serverWorkersDict['groupScattervCounts']
        offsets = serverWorkersDict['groupScattervOffsets']
        comm = serverWorkersDict['comm']
        serverRankInComm = serverWorkersDict['serverRankInComm']
        workerRanksInCommDict = serverWorkersDict['workerRanksInCommDict']
        if self.mp.isServer:
#            if self.logger.isEnabledFor(logging.DEBUG):
            assert detectorData1Darray is not None, "XCorrBase server expected data but got None"
            assert detectorData1Darray.dtype == np.float32, "XCorrBase server data dtype != expected dtype"
            assert len(detectorData1Darray) == sum(counts), "counts for scatter is wrong"
            assert counts[serverRankInComm] == 0, "server count for scatter is not zero"
            self.logger.debug('XCorrBase.serverWorkersScatter: server is sending data with first elem=%r' % detectorData1Darray[0])
                
            sendBuffer = detectorData1Darray
            recvBuffer = self.serverScatterReceiveBuffer
        elif self.mp.isWorker:
            sendBuffer = None
            recvBuffer = self.workerScatterReceiveBuffer
            thisWorkerRankInComm = workerRanksInCommDict[self.mp.rank]
            assert len(recvBuffer) == counts[thisWorkerRankInComm], 'recv buffer len != count'

        if self.isServerOrFirstWorker and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug('XCorrBase.serverWorkersScatter: before Scatterv.')
        comm.Scatterv([sendBuffer,
                       counts,
                       offsets,
                       MPI.FLOAT],
                      recvBuffer,
                      root = serverRankInComm)
        comm.Barrier()
        if self.isServerOrFirstWorker and self.logger.isEnabledFor(logging.DEBUG):
            if self.mp.isServer:
                self.logger.debug('XCorrBase.serverWorkersScatter: after Scatterv and Barrier')
            else:
                self.logger.debug('XCorrBase.serverWorkersScatter: after Scatterv and Barrier. First worker received %r as first element of buffer with dtype=%r' % (recvBuffer[0], recvBuffer.dtype))


    def serverInit(self):
        self.detectorData1Darray = np.empty(self.mp.totalElements, np.float32)
        self.serverScatterReceiveBuffer = np.zeros(0,dtype=np.float32)
        self.userObj.serverInit()

    def copyOurMaskedDataForScatter(self, dataArray):
        '''expects ndarray of float32
        '''
        self.detectorData1Darray[:] = dataArray[self.mp.maskNdarrayCoords]

    def initDelayAndGather(self):
        gatherOneNDArrayCounts = [self.mp.workerWorldRankToCount[rank] for rank in self.mp.workerRanks]
        gatherOneNDArrayCounts.insert(self.mp.viewerRankInViewerWorkersComm,0)
        gatherAllDelayCounts = [c*self.numDelays for c in gatherOneNDArrayCounts]
        
        gatherOneNDArrayOffsets = [0] + list(np.cumsum(gatherOneNDArrayCounts))[0:-1]
        gatherAllDelayOffsets = [0] + list(np.cumsum(gatherAllDelayCounts))[0:-1]

        CommSystemUtil.checkCountsOffsets(gatherOneNDArrayCounts, gatherOneNDArrayOffsets, self.totalMaskedElements)
        CommSystemUtil.checkCountsOffsets(gatherAllDelayCounts, gatherAllDelayOffsets, self.numDelays * self.totalMaskedElements)
        
        self.gatherOneNDArrayCounts = tuple(gatherOneNDArrayCounts)
        self.gatherOneNDArrayOffsets = tuple(gatherOneNDArrayOffsets)

        self.gatherAllDelayCounts = tuple(gatherAllDelayCounts)
        self.gatherAllDelayOffsets = tuple(gatherAllDelayOffsets)

    def workerInit(self):
        worldRank = self.mp.rank
        scatterCount = self.mp.workerWorldRankToCount[worldRank]
        thisWorkerStartElement = self.mp.workerWorldRankToOffset[worldRank]
        self.workerScatterReceiveBuffer = np.zeros(scatterCount,dtype=np.float32)
        self.initDelayAndGather()
        self.userObj.workerInit(scatterCount)
        self.elementsThisWorker = scatterCount
        self.workerData = WorkerData(logger=self.mp.logger, 
                                     isFirstWorker=self.mp.isFirstWorker,
                                     numTimes=self.system_params['times'],
                                     numDataPointsThisWorker=self.elementsThisWorker,
                                     storeDtype=self.system_params['workerStoreDtype'],
                                     addRemoveCallbackObject=self.userObj)

    def viewerInit(self):
        self.initDelayAndGather()

        self.gatheredFlatNDArrays = dict((name,np.zeros(self.numDelays*self.totalMaskedElements, np.float32)) \
                                         for name in self.arrayNames)

        gatheredMsgParts =['gathered_%s.shape=%s' % (nm, self.gatheredFlatNDArrays[nm].shape) \
                           for nm in self.arrayNames]
        gatheredMsg = ' '.join(gatheredMsgParts)
        self.logger.debug('XCorrBase.viewerInit: numDelays=%d totalMaskedElements=%d (includes masked out) %s (masked in only)' % \
                         (self.numDelays, self.totalMaskedElements, gatheredMsg))
        
        for nm in self.arrayNames:
            assert len(self.gatheredFlatNDArrays[nm]) == sum(self.gatherAllDelayCounts), "gathered_%s length != expected" % nm

        self.gatheredInt8array = np.zeros(self.totalMaskedElements, np.int8)
        assert len(self.gatheredInt8array) == sum(self.gatherOneNDArrayCounts), "gathered_int8array len != expected"

        self.h5output = None
        if self.system_params['h5output'] is not None:
            self.h5output = corAna.formatFileName(self.system_params['h5output'])
            self.h5inprogress = self.h5output + '.inprogress'
            if os.path.exists(self.h5output):
                if self.system_params['overwrite']:
                    if os.path.exists(self.h5inprogress):
                        raise Exception("h5output file %s specified with overwrite, but the inprogress file, %s exists. You must remove this file." % (self.h5output, self.h5inprogress))
                    os.unlink(self.h5output)
                    self.mp.logInfo("overwrite=True, removed file=%s" % self.h5output)
                else:
                    raise Exception("h5output file %s specified but that file exists. Set params overwrite to True to overwrite it." % self.h5output)
            if os.path.exists(self.h5inprogress):
                raise Exception("inprogress file for given h5output: %s exists. System will not overwrite even with --overwrite. Delete file before running" % self.h5inprogress)
            self.h5file = h5py.File(self.h5inprogress,'w')
            self.h5GroupFramework = writeConfig(self.h5file,
                                                self.system_params,
                                                self.user_params)
            self.h5GroupUser = self.h5file.create_group('user')
        else:
            self.h5file = None
            self.h5GroupFramework = None
            self.h5GroupUser = None

        self.userObj.viewerInit(self.mp.maskNdarrayCoords, self.h5GroupUser)

    def shutdown_viewer(self):
        if self.h5file is not None:
            del self.h5GroupUser
            del self.h5GroupFramework
            self.h5file.close()
            del self.h5file
            assert not os.path.exists(self.h5output), "ERROR: h5output file %s has been created since program started. Cannot move the inprogress file: %s to replace it. Output is in the inprogress file" % \
                (self.h5output, self.h5inprogress)
            shutil.move(self.h5inprogress, self.h5output)

    def storeNewWorkerData(self, counter):
        assert self.mp.isWorker, "storeNewWorkerData called for non-worker"
        self.workerData.addData(counter, 
                                self.workerScatterReceiveBuffer)

    def checkUserWorkerCalcArgs(self, name2array, counts, int8array):
        assert set(name2array.keys())==set(self.arrayNames), \
            "array names returned by workerCalc != expected named. Returned=%s != expected=%s " % \
            (str(name2array.keys()), str(self.arrayNames))
        for nm, array in name2array.iteritems():
            assert array.shape == (self.numDelays, self.elementsThisWorker,), "user workerCalc array %s has wrong shape. shape is %s != %s" % \
                                 (nm, array.shape, (self.elementsThisWorker,))
            assert array.dtype == np.float32, "workerCalc array=%s does not have type np.float32, it is %r" % array.dtype
        assert counts.dtype == np.int64, "user workerCalc counts array does not have dtype=np.int64"
        assert counts.shape == (self.numDelays,), "user workerCalc counts array shape=%s != (%d,)" % \
            (counts.shape, (self.numDelays,))
        assert int8array.dtype == np.int8, "user workerCalc int8array does not have dtype=np.int8"
        assert int8array.shape == (self.elementsThisWorker,), "user workerCalc int8array counts array shape=%s != (%d,)" % \
            (int8array.shape, (self.elementsThisWorker,))
        
    def viewerWorkersUpdate(self, lastTime):
        assert self.mp.isWorker or self.mp.isViewer, "can only call this function if viewer or worker"
        counter = lastTime['counter']
        # send the delay counts calculated thus far from one worker to the viewer.
        # The delay counts are the same across all the workers.

        # each worker needs to calculate results
        if self.mp.isWorker:
            ## calculate:
            t0 = time.time()
            name2array, counts, int8array = self.userObj.workerCalc(self.workerData)

            calcTime = time.time() - t0
            self.checkUserWorkerCalcArgs(name2array, counts, int8array)
            self.mp.logInfo('g2worker.calc at 120hz counter=%s took %.4f sec' % \
                            (counter, calcTime))
        ### begin point to point
        t0 = time.time()
        ## any data the viewer needs that is the same between the workers, send point to point to 
        # reduce network traffic
        if self.isViewerOrFirstWorker: 
            self.logger.debug('XCorrBase.viewerWorkersUpdate: before point to point Send/Recv for delayCounts ' + \
                              ('from first worker -> viewer. counter=%r' % counter))

        if self.mp.isFirstWorker:            
            assert counts.dtype == np.int64, "expected int64 dtype for delay counts - counts"
            self.mp.viewerWorkersComm.Send([counts,
                                            MPI.INT64_T],
                                           dest = self.mp.viewerRankInViewerWorkersComm)
        elif self.mp.isViewer:
            counts = np.zeros(self.numDelays, np.int64)
            self.mp.viewerWorkersComm.Recv([counts,
                                            MPI.INT64_T],
                                           source = self.mp.firstWorkerRankInViewerWorkersComm)
        self.mp.viewerWorkersComm.Barrier()
        if self.isViewerOrFirstWorker: 
            self.logger.debug('XCorrBase.viewerWorkersUpdate: after point to point Send/Recv for delayCounts from first worker -> viewer and Barrier. counter=%r' % counter)
        #### end point to point

        ### now gather up the results
        if self.isViewerOrFirstWorker: 
            self.logger.debug('XCorrBase.viewerWorkersUpdate: before Gatherv between workers and viewer of results. counter=%r' % counter)
        if self.mp.isViewer:
            sendBuffer = np.zeros(0,np.float32)
            for nm in self.arrayNames:
                gatherArray = self.gatheredFlatNDArrays[nm]
                receiveBuffer = gatherArray
                self.logger.debug('XCorrBase.viewerWorkersUpdate: before Gatherv for %s. sendbuf.shape=%r sendbuf.dtype=%r recvbuf.shape=%r recvbuf.dtype=%r gatherAllDelayCounts=%r gatherAllDelayOffsets=%r' % \
                                  (nm, sendBuffer.shape, sendBuffer.dtype, receiveBuffer.shape, receiveBuffer.dtype, self.gatherAllDelayCounts, self.gatherAllDelayOffsets))
                self.mp.viewerWorkersComm.Gatherv(sendbuf=[sendBuffer, MPI.FLOAT],
                                                  recvbuf=[receiveBuffer,
                                                           (self.gatherAllDelayCounts,
                                                            self.gatherAllDelayOffsets),
                                                           MPI.FLOAT],
                                                  root = self.mp.viewerRankInViewerWorkersComm)
                self.mp.viewerWorkersComm.Barrier()
                self.logger.debug('XCorrBase.viewerWorkersUpdate: after Gatherv and Barrier for %s' % nm)

            # use different counts for the int8array:
            self.logger.debug('XCorrBase.viewerWorkersUpdate: before Gatherv for gathered int8array: sendbuf.shape=%r sendbuf.dtype=%r recvbuf.shape=%r recvbuf.dtype=%r gatherOneCounts=%r gatherOneOffsets=%r' % \
                              (sendBuffer.shape, sendBuffer.dtype, self.gatheredInt8array.shape, self.gatheredInt8array.dtype, self.gatherOneNDArrayCounts, self.gatherOneNDArrayOffsets))
            self.mp.viewerWorkersComm.Gatherv(sendbuf=[sendBuffer, MPI.INT8_T],
                                              recvbuf=[self.gatheredInt8array,
                                                       (self.gatherOneNDArrayCounts,
                                                        self.gatherOneNDArrayOffsets),
                                                       MPI.INT8_T],
                                              root = self.mp.viewerRankInViewerWorkersComm)
            self.mp.viewerWorkersComm.Barrier()
            self.logger.debug('XCorrBase.viewerWorkersUpdate: after Gatherv and Barrier for gatheredInt8array')

        elif self.mp.isWorker:
            receiveBuffer = np.zeros(0,np.float32)
            for nm in self.arrayNames:
                gatherArray = name2array[nm]
                sendBuffer = gatherArray
                self.mp.logDebug('before Gatherv for %s. sendbuf.shape=%r sendbuf.dtype=%r recvbuf.shape=%r recvbuf.dtype=%r' % \
                                 (nm, sendBuffer.shape, sendBuffer.dtype, receiveBuffer.shape, receiveBuffer.dtype))
                self.mp.viewerWorkersComm.Gatherv(sendbuf=[sendBuffer,MPI.FLOAT],
                                                  recvbuf=[receiveBuffer, 
                                                           MPI.FLOAT],
                                                  root = self.mp.viewerRankInViewerWorkersComm)
                self.mp.viewerWorkersComm.Barrier()
                if self.mp.isFirstWorker: self.logger.debug('XCorrBase.viewerWorkersUpdate: after Gatherv for %s and Barrier' % nm)

            sendBuffer = int8array
            if self.mp.isFirstWorker: self.logger.debug('XCorrBase.viewerWorkersUpdate: before Gatherv for saturated. sendbuf.shape=%r sendbuf.dtype=%r recvbuf.shape=%r recvbuf.dtype=%r' % \
                                                        (sendBuffer.shape, sendBuffer.dtype, receiveBuffer.shape, receiveBuffer.dtype))
            self.mp.viewerWorkersComm.Gatherv(sendbuf=[sendBuffer, MPI.INT8_T],
                                              recvbuf=[receiveBuffer, 
                                                       MPI.INT8_T],
                                              root = self.mp.viewerRankInViewerWorkersComm)
            self.mp.viewerWorkersComm.Barrier()
            if self.mp.isFirstWorker: self.logger.debug('XCorrBase.viewerWorkersUpdate: after Gatherv for saturated and Barrier')
        else:
            raise Exception("viewerWorkersUpdate called but neither worker nor viewer")

        viewerWorkerCommTime = time.time()-t0
        if self.isViewerOrFirstWorker:
            self.logger.info("XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: %.3f sec" % viewerWorkerCommTime)

        if self.mp.isViewer:
            # all results are now gathered into flat 1D arrays
            name2delay2ndarray, int8ndarray = self.viewerFormNDarrays(counts, counter)
            
            self.userObj.viewerPublish(counts, lastTime, name2delay2ndarray, 
                                       int8ndarray, self.h5GroupUser)
            
    def viewerFormNDarrays(self, counts, counter):
        t0 = time.time()
        assert self.mp.isViewer, "XCorrBase.viewerFormNDarrays: viewerFormNDarrays called, but not viewer"
        assert len(counts) == self.numDelays, "XCorrBase.viewerFormNDarrays: len(counts)=%d != numDelays=%d" % \
            (len(counts), self.numDelays)
        
        workerStartPositions = [0]

        for workerRank in self.mp.workerRanks[0:-1]:
            workerCount = self.mp.workerWorldRankToCount[workerRank]
            workerStartPositions.append(workerCount * self.numDelays)

        ndarrayShape = self.mp.maskNdarrayCoords.shape  

        name2delay2ndarray = dict([(nm,{}) for nm in self.arrayNames])

        # get all the named arrays into name2delay2ndarray
        for delayIdx, delay in enumerate(self.delays):
            # for each delay, fill out these flattened arrays of the masked elements
            for nm in self.arrayNames:
                name2delay2ndarray[nm][delay] = np.zeros(ndarrayShape, np.float32)
                flatMaskedFromAllWorkers = np.zeros(self.totalMaskedElements, np.float32)

                for workerIdx, workerRank in enumerate(self.mp.workerRanks):
                    workerCount = self.mp.workerWorldRankToCount[workerRank]
                    workerOffset = self.mp.workerWorldRankToOffset[workerRank]
                    startIdx = workerStartPositions[workerIdx] + delayIdx * workerCount 
                    endIdx = startIdx + workerCount
                    flatMaskedThisWorker = self.gatheredFlatNDArrays[nm][startIdx:endIdx]
                    flatMaskedFromAllWorkers[workerOffset:(workerOffset+workerCount)] = flatMaskedThisWorker

                name2delay2ndarray[nm][delay][self.mp.maskNdarrayCoords] = flatMaskedFromAllWorkers

        # form the ndarray shaped int8 array
        int8ndarray = np.zeros(ndarrayShape, np.int8)
        int8ndarray[self.mp.maskNdarrayCoords] = self.gatheredInt8array

        t1 = time.time()
        self.logger.info('viewerFormNDarrays took %.3f sec' % (t1-t0,))

        return name2delay2ndarray, int8ndarray


