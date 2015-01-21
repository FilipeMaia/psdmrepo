from mpi4py import MPI
#import ctypes
import numpy as np
import os
import time
import shutil
import StringIO
import pprint

import h5py

from ParCorAna.XCorrWorkerBase import XCorrWorkerBase
import ParCorAna.Timing as Timing
import ParCorAna.CommSystemUtil as CommSystemUtil
from EventIter import EventIter

#################################
# utitlity functions users may use
def makeDelayList(start, stop, num, spacing, logbase=np.e):
    '''Constructs a list of delays, from [start,top] with endpoints, with either linear or log spacing.
    returns list as array of int32.
    '''
    assert spacing in ['log','lin'], "spacing must be 'log' or 'lin'"
    assert stop >= start
    assert num > 0
    assert start > 0
    if spacing == 'log':
        logStart = np.log(start)
        logStop = np.log(stop)
        delays = np.logspace(start=logStart, stop=logStop, num=num, endpoint=True, base=logbase)
    else:
        delays = np.linspace(start, stop, num)
    delays = list(set(np.round(delays).astype(np.int32)))
    delays.sort()
    return np.array(delays, dtype=np.int32)

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

def writeConfig(h5Group, system_params, user_params):
    dictBuffer = StringIO.StringIO()
    pprint.pprint(system_params, dictBuffer)
    dictString = dictBuffer.getvalue()
    h5Group["system_params"] = dictString

    dictBuffer = StringIO.StringIO()
    pprint.pprint(user_params, dictBuffer)
    dictString = dictBuffer.getvalue()
    h5Group["user_params"] = dictString

    maskFile = system_params['mask_ndarrayCoords']
    assert os.path.exists(maskFile), "mask file doesn't exist"
    maskNumpyArray = np.load(file(maskFile,'r'))
    h5Group['mask_ndarrayCoords']=maskNumpyArray
    
    # really the system shouldn't know about details of user_params, but right now we
    # will look for the color file and write it to the h5 file
    colorFile = user_params['color_ndarrayCoords']
    assert os.path.exists(colorFile), "specified color file doesn't exist: %s" % colorFile
    colorNumpyArray = np.load(file(colorFile,'r'))
    h5Group['color_ndarrayCoords'] = colorNumpyArray        

###############################
class XCorrBase(object):
    def __init__(self, mp, dataSourceString, srcString,
                 numEvents, maxTimes, system_params, user_params):
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
        userClass = system_params['user_class']
        self.userObj = userClass(user_params, system_params, mp)
        self.arrayNames = self.userObj.arrayNames()
        self.totalMaskedElements = np.sum(self.mp.mask_ndarrayCoords)

    def makeEventIter(self):
        self.logger.debug('XCorrBase.makeEventIter returning EventIter(datasource=%s,rank=%s,servers=%s,numEvents=%s)' % \
                          (self.dataSourceString,self.mp.rank,self.mp.serverRanks,self.numEvents))
        return EventIter(self.dataSourceString,
                         self.mp.rank,
                         self.mp.serverRanks,
                         self,
                         self.system_params,
                         self.mp.mask_ndarrayCoords.shape,
                         self.mp.logger,
                         self.numEvents)

    def serverWorkersScatter(self, serverFullDataArray = None, serverWorldRank = None):
        '''called from both server and worker ranks for the scattering of the data.

        When called from the server, args are
        serverFullDataArray - image producer result - 2D nparray float64
        serverWorldRank     - None, one can get rank from self.mp.rank

        When called from the worker, args are
        serverFullDataArray  - None
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
            assert serverFullDataArray is not None, "XCorrBase server expected data but got None"
            assert serverFullDataArray.dtype == self.detectorData1Darray.dtype, "XCorrBase server data dtype != expected dtype"
            self.detectorData1Darray[:] = serverFullDataArray[self.mp.mask_ndarrayCoords]
            assert len(self.detectorData1Darray) == sum(counts), "counts for scatter is wrong"
            assert counts[serverRankInComm] == 0, "sever count for scatter is not zero"
            sendBuffer = self.detectorData1Darray
            recvBuffer = self.serverScatterReceiveBuffer
        elif self.mp.isWorker:
            sendBuffer = None
            recvBuffer = self.workerScatterReceiveBuffer
            thisWorkerRankInComm = workerRanksInCommDict[self.mp.rank]
            assert len(recvBuffer) == counts[thisWorkerRankInComm], 'recv buffer len != count'

        if self.isServerOrFirstWorker: self.logger.debug('XCorrBase.serverWorkersScatter: before Scatterv')
        comm.Scatterv([sendBuffer,
                       counts,
                       offsets,
                       MPI.DOUBLE],
                      recvBuffer,
                      root = serverRankInComm)
        comm.Barrier()
        if self.isServerOrFirstWorker: self.logger.debug('XCorrBase.serverWorkersScatter: after Scatterv and Barrier')

    def initServer(self):
        self.detectorData1Darray = np.empty(self.mp.totalElements, np.float)
        self.serverScatterReceiveBuffer = np.zeros(0,dtype=np.float)
        self.userObj.initServer()

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

    def initWorker(self):
        worldRank = self.mp.rank
        scatterCount = self.mp.workerWorldRankToCount[worldRank]
        thisWorkerStartElement = self.mp.workerWorldRankToOffset[worldRank]
        self.workerScatterReceiveBuffer = np.zeros(scatterCount,dtype=np.float)
        self.initDelayAndGather()
        self.xCorrWorkerBase = XCorrWorkerBase(scatterCount,
                                               self.system_params['times'],
                                               self.mp.isFirstWorker,
                                               self.mp.logger)
                                               
                                               
        self.userObj.initWorker(scatterCount)
        self.elementsThisWorker = scatterCount


    def initViewer(self):
        self.initDelayAndGather()

        self.gatheredFlatNDArrays = dict((name,np.zeros(self.numDelays*self.totalMaskedElements, np.float)) \
                                         for name in self.arrayNames)

        gatheredMsgParts =['gathered_%s.shape=%s' % (nm, self.gatheredFlatNDArrays[nm].shape) \
                           for nm in self.arrayNames]
        gatheredMsg = ' '.join(gatheredMsgParts)
        self.logger.debug('XCorrBase.initViewer: numDelays=%d totalMaskedElements=%d (includes masked out) %s (masked in only)' % \
                         (self.numDelays, self.totalMaskedElements, gatheredMsg))
        
        for nm in self.arrayNames:
            assert len(self.gatheredFlatNDArrays[nm]) == sum(self.gatherAllDelayCounts), "gathered_%s length != expected" % nm

        self.gatheredInt8array = np.zeros(self.totalMaskedElements, np.int8)
        assert len(self.gatheredInt8array) == sum(self.gatherOneNDArrayCounts), "gathered_int8array len != expected"

        if self.system_params['h5output'] is not None:
            h5output = self.system_params['h5output']
            self.h5inprogress = self.system_params['h5output'] + '.inprogress'
            if os.path.exists(h5output):
                if self.system_params['overwrite']:
                    if os.path.exists(self.h5inprogress):
                        raise Exception("h5output file %s specified with overwrite, but the inprogress file, %s exists. You must remove this file." % (h5output, self.h5inprogress))
                    os.unlink(h5output)
                    self.mp.logInfo("overwrite=True, removed file=%s" % h5output)
                else:
                    raise Exception("h5output file %s specified but that file exists. Set params overwrite to True to overwrite it." % self.system_params['h5output'])
            if os.path.exists(self.h5inprogress):
                raise Exception("inprogress file for given h5output: %s exists. System will not overwrite even with --overwrite. Delete file before running" % self.h5inprogress)
            self.h5file = h5py.File(self.h5inprogress,'w')
            self.h5GroupFramework = self.h5file.create_group('system')
            writeConfig(self.h5GroupFramework,
                        self.system_params,
                        self.user_params)
            self.h5GroupUser = self.h5file.create_group('user')
        else:
            self.h5file = None
            self.h5GroupFramework = None
            self.h5GroupUser = None

        self.userObj.initViewer(self.mp.mask_ndarrayCoords, self.h5GroupUser)

    def shutdown_viewer(self):
        if self.h5file is not None:
            del self.h5GroupUser
            del self.h5GroupFramework
            self.h5file.close()
            del self.h5file
            assert not os.path.exists(self.system_params['h5output']), "ERROR: h5output file %s has been created since program started. Cannot move the inprogress file: %s to replace it. Output is in the inprogress file" % \
                (self.system_params['h5output'], self.h5inprogress)
            shutil.move(self.h5inprogress, self.system_params['h5output'])

    def storeNewWorkerData(self, relSeconds):
        assert self.mp.isWorker, "storeNewWorkerData called for non-worker"
        self.xCorrWorkerBase.updateData(relSeconds, self.workerScatterReceiveBuffer,
                                        self.userObj, )

    def checkUserWorkerCalcArgs(self, name2array, counts, int8array):
        assert set(name2array.keys())==set(self.arrayNames), \
            "array names returned by workerCalc != expected named. Returned=%s != expected=%s " % \
            (str(name2array.keys()), str(self.arrayNames))
        for nm, array in name2array.iteritems():
            assert array.shape == (self.numDelays, self.elementsThisWorker,), "user workerCalc array %s has wrong shape. shape is %s != %s" % \
                                 (nm, array.shape, (self.elementsThisWorker,))
        assert counts.dtype == np.int64, "user workerCalc counts array does not have dtype=np.int64"
        assert counts.shape == (self.numDelays,), "user workerCalc counts array shape=%s != (%d,)" % \
            (counts.shape, (self.numDelays,))
        assert int8array.dtype == np.int8, "user workerCalc int8array does not have dtype=np.int8"
        assert int8array.shape == (self.elementsThisWorker,), "user workerCalc int8array counts array shape=%s != (%d,)" % \
            (int8array.shape, (self.elementsThisWorker,))
        
    def viewerWorkersUpdate(self, relsec = None):
        assert self.mp.isWorker or self.mp.isViewer, "can only call this function if viewer or worker"

        # send the delay counts calculated thus far from one worker to the viewer.
        # The delay counts are the same across all the workers.

        # each worker needs to calculate results
        if self.mp.isWorker:
            ## calculate:
            t0 = time.time()
            name2array, counts, int8array = self.userObj.workerCalc(self.xCorrWorkerBase.T, \
                                                                    self.xCorrWorkerBase.numTimesFilled(), \
                                                                    self.xCorrWorkerBase.X)
            calcTime = time.time() - t0
            self.checkUserWorkerCalcArgs(name2array, counts, int8array)
            self.mp.logInfo('g2worker.calc at 120hz counterrelsec=%d took %.4f sec' % \
                            (int(np.round(120*relsec)), calcTime))
        ### begin point to point
        t0 = time.time()
        ## any data the viewer needs that is the same between the workers, send point to point to 
        # reduce network traffic
        if self.isViewerOrFirstWorker: 
            self.logger.debug('XCorrBase.viewerWorkersUpdate: before point to point Send/Recv for delayCounts ' + \
                              ('from first worker -> viewer. relsec=%r' % relsec))

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
            self.logger.debug('XCorrBase.viewerWorkersUpdate: after point to point Send/Recv for delayCounts from first worker -> viewer and Barrier. relsec=%r' % relsec)
        #### end point to point

        ### now gather up the results
        if self.isViewerOrFirstWorker: 
            self.logger.debug('XCorrBase.viewerWorkersUpdate: before Gatherv between workers and viewer of results. relsec=%r' % relsec)
        if self.mp.isViewer:
            sendBuffer = np.zeros(0,np.float)
            for nm in self.arrayNames:
                gatherArray = self.gatheredFlatNDArrays[nm]
                receiveBuffer = gatherArray
                self.logger.debug('XCorrBase.viewerWorkersUpdate: before Gatherv for %s. sendbuf.shape=%r sendbuf.dtype=%r recvbuf.shape=%r recvbuf.dtype=%r gatherAllDelayCounts=%r gatherAllDelayOffsets=%r' % \
                                  (nm, sendBuffer.shape, sendBuffer.dtype, receiveBuffer.shape, receiveBuffer.dtype, self.gatherAllDelayCounts, self.gatherAllDelayOffsets))
                self.mp.viewerWorkersComm.Gatherv(sendbuf=[sendBuffer, MPI.DOUBLE],
                                                  recvbuf=[receiveBuffer,
                                                           (self.gatherAllDelayCounts,
                                                            self.gatherAllDelayOffsets),
                                                           MPI.DOUBLE],
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
            receiveBuffer = np.zeros(0,np.float)
            for nm in self.arrayNames:
                gatherArray = name2array[nm]
                sendBuffer = gatherArray
                self.mp.logDebug('before Gatherv for %s. sendbuf.shape=%r sendbuf.dtype=%r recvbuf.shape=%r recvbuf.dtype=%r' % \
                                 (nm, sendBuffer.shape, sendBuffer.dtype, receiveBuffer.shape, receiveBuffer.dtype))
                self.mp.viewerWorkersComm.Gatherv(sendbuf=[sendBuffer,MPI.DOUBLE],
                                                  recvbuf=[receiveBuffer, 
                                                           MPI.DOUBLE],
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
            name2delay2ndarray, int8ndarray = self.viewerFormNDarrays(counts, relsec)
            
            self.userObj.viewerPublish(counts, relsec, name2delay2ndarray, 
                                       int8ndarray, self.h5GroupUser)
            


    def viewerFormNDarrays(self, counts, relsec):
        t0 = time.time()
        assert self.mp.isViewer, "XCorrBase.viewerFormNDarrays: viewerFormNDarrays called, but not viewer"
        assert len(counts) == self.numDelays, "XCorrBase.viewerFormNDarrays: len(counts)=%d != numDelays=%d" % \
            (len(counts), self.numDelays)
        
        workerStartPositions = [0]

        for workerRank in self.mp.workerRanks[0:-1]:
            workerCount = self.mp.workerWorldRankToCount[workerRank]
            workerStartPositions.append(workerCount * self.numDelays)

        ndarrayShape = self.mp.mask_ndarrayCoords.shape  

        name2delay2ndarray = dict([(nm,{}) for nm in self.arrayNames])

        # get all the named arrays into name2delay2ndarray
        for delayIdx, delay in enumerate(self.delays):
            # for each delay, fill out these flattened arrays of the masked elements
            for nm in self.arrayNames:
                name2delay2ndarray[nm][delay] = np.zeros(ndarrayShape, np.float)
                flatMaskedFromAllWorkers = np.zeros(self.totalMaskedElements, np.float)

                for workerIdx, workerRank in enumerate(self.mp.workerRanks):
                    workerCount = self.mp.workerWorldRankToCount[workerRank]
                    workerOffset = self.mp.workerWorldRankToOffset[workerRank]
                    startIdx = workerStartPositions[workerIdx] + delayIdx * workerCount 
                    endIdx = startIdx + workerCount
                    flatMaskedThisWorker = self.gatheredFlatNDArrays[nm][startIdx:endIdx]
                    flatMaskedFromAllWorkers[workerOffset:(workerOffset+workerCount)] = flatMaskedThisWorker

                name2delay2ndarray[nm][delay][self.mp.mask_ndarrayCoords] = flatMaskedFromAllWorkers

        # form the ndarray shaped int8 array
        int8ndarray = np.zeros(ndarrayShape, np.int8)
        int8ndarray[self.mp.mask_ndarrayCoords] = self.gatheredInt8array

        t1 = time.time()
        self.logger.info('viewerFormNDarrays took %.3f sec' % (t1-t0,))

        return name2delay2ndarray, int8ndarray


