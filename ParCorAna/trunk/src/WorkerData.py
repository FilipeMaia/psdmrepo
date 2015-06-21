import numpy as np
import ParCorAna as pca

class WorkerData(object):
    '''provide access to worker data.

    Clients carrying out correlation calculation on the worker data
    need to efficiently access data based on time. Keeping the data
    itself sorted could be costly. This class keeps the times sorted
    and maintains indicies into the data.
    '''
    TIME_COLUMN = 0
    X_COLUMN = 1
    INVALID_INDEX = -1

    def __init__(self, logger, isFirstWorker, numTimes, numDataPointsThisWorker, 
                 storeDtype=np.float64, addRemoveCallbackObject=None):
        self.numTimes = numTimes
        self.isFirstWorker = isFirstWorker

        numWorkerEventsToStore = numTimes
        numTimesToInitiallyStore = 2*numTimes

        # public data clients will work with
        self.X = np.empty((numWorkerEventsToStore,numDataPointsThisWorker), 
                          dtype=storeDtype)

        # for going from a time to the data
        # _timesXInds is private, it may be resized which would brake references to it
        self._timesXInds = np.empty((numTimesToInitiallyStore,2), np.int64)
        self._timesXInds[:,WorkerData.X_COLUMN] = WorkerData.INVALID_INDEX

        # for going from data to the time, used internally when re-using a data entry in X
#        self._xInd2timeInd = np.empty(numWorkerEventsToStore, np.int64)
#        self._xInd2timeInd[:] = WorkerData.INVALID_INDEX

        self._timeStartIdx = 0       # will increase as we overwrite data
        self._timeAfterEndIdx = 0    # always one more than where the last stored time is
        self._nextXIdx = 0
        self.wrappedX = False
        self.numOutOfOrder = 0
        self.numDupTimes = 0

        self.logger = logger
        self.addRemoveCallbackObject = addRemoveCallbackObject

    def dumpstr(self):
        res = "tmIdx: start=%d afterEnd=%d\n" % (self._timeStartIdx, self._timeAfterEndIdx)
        res += "times=\n%r" % self._timesXInds
        return res

    def timesDataIndexes(self):
        '''iterator over tm,idx pairs, the times in order, with indicies into X
        of where the data for that time is
        '''
        for idx in xrange(self._timeStartIdx, self._timeAfterEndIdx):
            tm, xIndex = self._timesXInds[idx,:]
            if xIndex != WorkerData.INVALID_INDEX:
                yield tm, xIndex

    def tm2idx(self, tm):
        '''For a given time, the X index into the data

        Args:
          tm (int): the time to find

        Return:
          (int): the row index into the X array for where the data is for this time, or
                 None if this time is not stored.
        '''
        currentFilledTimesView = self._timesXInds[self._timeStartIdx:self._timeAfterEndIdx, WorkerData.TIME_COLUMN]
        tmIndex = np.searchsorted(currentFilledTimesView, tm)
        if tmIndex >= len(currentFilledTimesView):
            return None
        foundTime = currentFilledTimesView[tmIndex] == tm
        if not foundTime:
            return None
        xInd = self._timesXInds[self._timeStartIdx + tmIndex, WorkerData.X_COLUMN]
        if xInd == WorkerData.INVALID_INDEX:
            return None
        return xInd

    ## --------- begin helper functions for addData
    def _growTimesIfNeeded(self):
        if self._timeAfterEndIdx + 2 >= self._timesXInds.shape[0]:
            oldShape = self._timesXInds.shape
            newShape = [oldShape[0]+self.numTimes, oldShape[1] ]
            newShape[0] = max(newShape[0], self._timeAfterEndIdx + 3)
            self._timesXInds.resize(newShape, refcheck=False)

    def empty(self):
        return self._timeAfterEndIdx == self._timeStartIdx

    def latestTime(self):
        assert not self.empty(), "WorkerData.latestTime called before any data stored"
        latestTimeIdx = self._timeAfterEndIdx-1
        latestTime = self._timesXInds[latestTimeIdx, WorkerData.TIME_COLUMN]
        return latestTime

    def _nextTimePosition(self, tm):
        if self.empty() or tm > self.latestTime():
            return self._timeAfterEndIdx

        currentFilledTimesView = self._timesXInds[self._timeStartIdx:self._timeAfterEndIdx, 
                                                  WorkerData.TIME_COLUMN]
        tmIndex = np.searchsorted(currentFilledTimesView, tm)
        indexInsideView = (tmIndex >= self._timeStartIdx) and (tmIndex < self._timeAfterEndIdx)
        if indexInsideView:
            if currentFilledTimesView[tmIndex] == tm:
                raise pca.WorkerDataDuplicateTime(tm)
        return tmIndex

    def _nextXPosition(self, tm):
        '''get next xindex for new data. Assumes slot will be used, may set wrappedX. 
        '''
        if not self.wrappedX:
            if self._nextXIdx >= self.X.shape[0]:
                self.wrappedX = True
        
        if self.wrappedX:
            earliestTime, earliestXidx = self._timesXInds[self._timeStartIdx, :]
            if tm <= earliestTime:
                raise pca.WorkerDataNextTimeIsEarliest(tm)
            return earliestXidx
        else:
            xIndForNewData = self._nextXIdx
            self._nextXIdx += 1
            return xIndForNewData


    ## ---------- end helper functions for addData

    def addData(self, tm, newXrow):
        def filled(self):
            if self.wrappedX or self._nextXIdx == self.X.shape[0]:
                return True
            return False

        if filled(self) and tm <= self.minTimeForStoredData():
            if self.isFirstWorker:
                self.logger.warning("filled X and received time=%d <= first time=%d, skipping" % \
                                    (tm, self.minTimeForStoredData()))
            return

        self._growTimesIfNeeded()

        try:
            timeIndForNewData = self._nextTimePosition(tm)
        except pca.WorkerDataDuplicateTime:
            if self.isFirstWorker:
                self.logger.warning("addData: duplicated time=%d, skipping" % tm)
                self.numDupTimes += 1
                return
        if timeIndForNewData < self._timeAfterEndIdx:
            self.numOutOfOrder += 1
            if timeIndForNewData == self._timeStartIdx and self.wrappedX and self.isFirstWorker:
                self.logger.warning("addData: X has already been filled but new data with tm=%d is earlier then all stored data. Dropping data." % timeIndForNewData)
                return
            # we are committed to storing this data.
            # slide down to make room for it.
            self._timesXInds[timeIndForNewData+1:self._timeAfterEndIdx+1,:] = self._timesXInds[timeIndForNewData:self._timeAfterEndIdx,:] 

        try:
            xIndForNewData = self._nextXPosition(tm)
        except pca.WorkerDataNextTimeIsEarliest:
            self.logger.error("WorkerData.AddData: unexpected, already checking for earliest time in filled case, tm=%d, skipping" % tm)
            return
            
        if self.wrappedX:
            assert xIndForNewData == self._timesXInds[self._timeStartIdx, WorkerData.X_COLUMN], "unxpected, wrappedX but new x ind is not for first time"
            if self.addRemoveCallbackObject is not None:
                self.addRemoveCallbackObject.workerBeforeDataRemove(self._timesXInds[self._timeStartIdx, WorkerData.TIME_COLUMN],
                                                                    self._timesXInds[self._timeStartIdx, WorkerData.X_COLUMN],
                                                                    self)
            # remove old time, clients should no longer be able to see it after this
            self._timesXInds[self._timeStartIdx, WorkerData.X_COLUMN] = WorkerData.INVALID_INDEX
            self._timeStartIdx += 1

        # store new time, but not the data yet.
        self._timesXInds[timeIndForNewData,WorkerData.TIME_COLUMN]=tm
        self._timesXInds[timeIndForNewData,WorkerData.X_COLUMN]=WorkerData.INVALID_INDEX
        self._timeAfterEndIdx += 1


        # let client adjust, then store data
        if self.addRemoveCallbackObject is not None:
            self.addRemoveCallbackObject.workerAdjustData(newXrow)

        self.X[xIndForNewData,:] = newXrow[:]
        self._timesXInds[timeIndForNewData, WorkerData.X_COLUMN] = xIndForNewData
        
        if self.addRemoveCallbackObject is not None:
            self.addRemoveCallbackObject.workerAfterDataInsert(tm, xIndForNewData, self)
            
#        print "addData: finished, timeIndForNewData=%d xIndForNewData=%d wrapped=%d tmStard=%d tmAfterEnd=%d" % \
#            (timeIndForNewData, xIndForNewData, self.wrappedX, self._timeStartIdx, self._timeAfterEndIdx)
#        print "addData: stored times: %s" % ','.join(map(str,self._timesXInds[self._timeStartIdx:self._timeAfterEndIdx,0]))
#        print "addData: stored xidxs: %s" % ','.join(map(str,self._timesXInds[self._timeStartIdx:self._timeAfterEndIdx,1]))


    ####### EVAL? ######
    def minTimeForStoredData(self):
        '''
        '''
        assert not self.empty(), "can't ask for min time on empty data. use empty() to check before calling this function"
        return self._timesXInds[self._timeStartIdx, WorkerData.TIME_COLUMN]
    
    def maxTimeForStoredData(self):
        assert not self.empty(), "can't ask for max time on empty data. use empty() to check before calling this function"
        return self._timesXInds[self._timeAfterEndIdx-1, WorkerData.TIME_COLUMN]

    def timesForStoredData(self):
        '''returns sorted copy of times (120hz counters) received thus far by this worker
        
        There may be gaps or negative values, ie [-3, 0,1, 5, 8]
        '''
        return [tmXidx[0] for tmXidx in self.timesDataIndexes()]
        

