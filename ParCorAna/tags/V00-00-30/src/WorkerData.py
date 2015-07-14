import numpy as np
import Exceptions
import logging
import math

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
                 storeDtype=np.float32, addRemoveCallbackObject=None):
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

        self._timeStartIdx = 0       # will increase as we overwrite data
        self._timeAfterEndIdx = 0    # always one more than where the last stored time is
        self._nextXIdx = 0
        self.numOutOfOrder = 0
        self.numDupTimes = 0

        self.logger = logger
        self.addRemoveCallbackObject = addRemoveCallbackObject
        if self.isFirstWorker: self.logger.debug(self.dumpStr())

    def dumpStr(self, long=False):
        res = "WorkerData tmStart=%d tmAfterEnd=%d nextX=%d filledX=%d numOutOfOrder=%d numDupTimes=%d X.shape=%r _timesXInds.shape=%r" % \
              (self._timeStartIdx, self._timeAfterEndIdx, self._nextXIdx, self.filledX(), self.numOutOfOrder, self.numDupTimes, self.X.shape, self._timesXInds.shape)
        if long:
            maxwidth = 2
            def fmt(x):
                return str(x).rjust(maxwidth+1)
            tms = map(fmt,self._timesXInds[self._timeStartIdx:self._timeAfterEndIdx:,0])
            xInds = map(fmt, self._timesXInds[self._timeStartIdx:self._timeAfterEndIdx,1])
            res += "\n_timesXInds[%d:%d,TIME_COL]=%s" % (self._timeStartIdx, self._timeAfterEndIdx,' '.join(tms))
            res += "\n_timesXInds[%d:%d,XIND_COL]=%s" % (self._timeStartIdx, self._timeAfterEndIdx,' '.join(xInds))
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

    def _nextTimePosition(self, tm):
        if self.empty() or tm > self.maxTimeForStoredData():
            return self._timeAfterEndIdx

        if tm < self.minTimeForStoredData():
            return self._timeStartIdx

        currentFilledTimesView = self._timesXInds[self._timeStartIdx:self._timeAfterEndIdx, 
                                                  WorkerData.TIME_COLUMN]
        tmIndexInView = np.searchsorted(currentFilledTimesView, tm)
        if tmIndexInView < len(currentFilledTimesView) and currentFilledTimesView[tmIndexInView] == tm:
            raise Exceptions.WorkerDataDuplicateTime(tm)
        return tmIndexInView + self._timeStartIdx

    ## ---------- end helper functions for addData

    def addData(self, tm, newXrow):
        '''adds new data to WorkerData. Calls callbacks as need be.

        Returns True/False if data added.
        '''
        if self.filledX() and tm <= self.minTimeForStoredData():
            if self.isFirstWorker:
                self.logger.warning("filled X and received time=%d <= first time=%d, skipping" % \
                                    (tm, self.minTimeForStoredData()))
            return False

        self._growTimesIfNeeded()

        try:
            timeIndForNewData = self._nextTimePosition(tm)
        except Exceptions.WorkerDataDuplicateTime:
            if self.isFirstWorker:
                self.logger.warning("addData: duplicated time=%d, skipping" % tm)
                self.numDupTimes += 1
                return False

        assert timeIndForNewData >= self._timeStartIdx and timeIndForNewData <= self._timeAfterEndIdx, \
            "addData: _nextTimePosition returned invalid time index=%d, not in [%d,%d]" % \
            (timeIndForNewData, self._timeStartIdx, self._timeAfterEndIdx)

        if self.filledX() and timeIndForNewData == self._timeStartIdx and self.isFirstWorker:
            self.logger.warning("addData: X filled but new time=%d is <= min of stored times. Dropping data." % tm)
            return False

        # we are committed to storing this data
        xIndForNewData = None
        if self.filledX():
            tmForRemoval = self._timesXInds[self._timeStartIdx, WorkerData.TIME_COLUMN]
            xIndForRemoval = self._timesXInds[self._timeStartIdx, WorkerData.X_COLUMN]
            assert xIndForRemoval >= 0 and xIndForRemoval < self.X.shape[0], "addData: xInd=%d for removal is bad" % xIndForRemoval
            if self.addRemoveCallbackObject is not None:
                self.addRemoveCallbackObject.workerBeforeDataRemove(tmForRemoval, xIndForRemoval, self)
            self._timesXInds[self._timeStartIdx, WorkerData.X_COLUMN] = WorkerData.INVALID_INDEX
            self._timeStartIdx += 1
            xIndForNewData = xIndForRemoval
        else:
            xIndForNewData = self._nextXIdx
            self._nextXIdx += 1
        assert xIndForNewData >= 0 and xIndForNewData < self.X.shape[0], "addData: xIndForNewData=%d is bad" % xIndForNewData

        # let client adjust new data before we store it.
        if self.addRemoveCallbackObject is not None:
            self.addRemoveCallbackObject.workerAdjustData(newXrow)

        if timeIndForNewData < self._timeAfterEndIdx:
            self.numOutOfOrder += 1
            # slide down in time array to make room for new time
            self._timesXInds[timeIndForNewData+1:self._timeAfterEndIdx+1,:] = self._timesXInds[timeIndForNewData:self._timeAfterEndIdx,:] 
        self._timeAfterEndIdx += 1
            
        # store new time and data
        self._timesXInds[timeIndForNewData,WorkerData.TIME_COLUMN]=tm
        self.X[xIndForNewData,:] = newXrow[:]
        self._timesXInds[timeIndForNewData, WorkerData.X_COLUMN] = xIndForNewData
        
        if self.addRemoveCallbackObject is not None:
            self.addRemoveCallbackObject.workerAfterDataInsert(tm, xIndForNewData, self)

        if self.isFirstWorker and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(("addData tm=%d at xIdx=%d -- " % (tm,xIndForNewData)) + self.dumpStr(long=True))
            
        return True


    ######## utility functions ##########

    def empty(self):
        return self._timeAfterEndIdx == self._timeStartIdx

    def filledX(self):
        return self._nextXIdx == self.X.shape[0]

    def minTimeForStoredData(self):
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
        

