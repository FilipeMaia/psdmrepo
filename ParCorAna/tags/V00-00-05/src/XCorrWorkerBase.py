import numpy as np
import logging

def getMinMaxStoreValues(storeType):
    if issubclass(storeType, np.floating):
        info = np.finfo(storeType)
        return info.min, info.max
    if issubclass(storeType, np.integer):
        info = np.iinfo(storeType)
        return info.min, info.max
    raise Exception("getMinMaxStoreValues called with storeType=%r that is neither integral no floating" % storeType)

class XCorrWorkerBase(object):
    SUBTRACT = 1
    ADD = 2
    ALLOWABLE_STORE_TYPES = [np.uint16, np.int16, np.int32, np.uint32, np.float32, np.float64]

    def __init__(self, numElementsWorker, maxTimes, isFirstWorker, storeType, logger):
        '''init XCorrWorkerBase to store scattered event data.
        ARGS:
        numElementsWorker - how many of the detector elements/pixels does this workers handle
        maxTimes          - how many times will the worker store
        isFirstWorker     - True if is first worker, then this worker will log messages.
        storeType         - the numpy type to use to internally store values. one of the
                            class member ALLOWABLE_STORE_TYPES.
        An instance of XCorrWorkerBase creates a numElementsWorker * maxTimes array of storeType.
        '''
        self.numElementsWorker = numElementsWorker
        self.maxTimes = maxTimes
        self.isFirstWorker = isFirstWorker
        self.logger = logger
        
        # a 1D array of times in 120hz counter. The 120hz counter for each row of X below
        self.T = np.empty(self.maxTimes, np.int64)
        
        assert storeType in XCorrWorkerBase.ALLOWABLE_STORE_TYPES, \
            "storeType=%r, but it must be one or %r" % (storeType, \
                                                        XCorrWorkerBase.ALLOWABLE_STORE_TYPES)
           
        self.checkForOverflow=False
        self.storeTypeMinVal = None
        self.storeTypeMaxVal = None
        self.numOverflowEvents = 0
        self.MaxOverflowReports=50
        if storeType != np.float64:
            self.checkForOverflow=True
            self.storeTypeMinVal, self.storeTypeMaxVal = getMinMaxStoreValues(storeType)
            
        # a 2D array of element values. X[i,:] is all the detector elements that this worker 
        # processes from time T[i]
        self.X = np.empty((self.maxTimes, numElementsWorker), storeType)
        

        # which column of X and element of time120Hz to fill out next
        self.nextTimeIdx = 0

        # true once we have filled up X and are overwritting previous entries
        self.wrapped = False

        # set of times to make sure we don't store a repeat.
        # important for guaranteeing that after sorting T, times that are two apart are no
        # more than 2 apart
        self.currentTimesStored = set()

        self.numNewData = 0
        self.numRepeatData = 0
        self.lastRepeatWarning = 0

    def updateData(self, counter, data, userObj):
        '''updates the data.
        '''
        if self.isFirstWorker and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug('updateData from scatter. counter=%d data.shape=%r data.dtype=%r data[0]=%r' % \
                              (counter, data.shape, data.dtype, data[0]))
        next120hz = counter
        # is this repeat data?
        if next120hz in self.currentTimesStored:
            self.numRepeatData += 1
            if self.isFirstWorker and self.numRepeatData - self.lastRepeatWarning > 10:
                self.logger.warning("XCorrWorkerBase.updateData: %d/%d repeat times at 120hz counter: %s" % (self.numRepeatData, self.numNewData, next120hz))
                self.lastRepeatWarning = self.numRepeatData
            return

        self.numNewData += 1

        if self.wrapped:
            # allow user object to subtract the effects of the column we are going to overwrite
            # in any accumulated state

            # assume ciruclarly sorted, so the smallest time is the one we are going to overwrite
            pivotIndex = self.nextTimeIdx
            userObj.workerBeforeDataRemove(self.nextTimeIdx, pivotIndex, self.numTimesFilled(), self.T, self.X)
            
        self.logger.debug('XCorrWorkerBase.updateData next120hz=%d' % (next120hz,))

        # allow user object to make adjustments to data
        userObj.workerAdjustData(data)

        # check for overflow 
        if self.checkForOverflow:
            overflow=np.where(data>self.storeTypeMaxVal)[0]  # data is 1D, where returns tuple with 1 elem
            underflow=np.where(data<self.storeTypeMinVal)[0] # data is 1D, where returns tuple with 1 elem
            warnMsg = ''
            if len(overflow)>0:
                data[overflow]=self.storeTypeMaxVal
                warnMsg += "%d pixels would have overflowed. Set them to maxvalue=%r" % (len(overflow), 
                                                                                         self.storeTypeMaxVal)
            if len(underflow)>0:
                data[underflow]=self.storeTypeMinVal
                warnMsg += "%d pixels would have underflowed. Set them to miinvalue=%r" % (len(underflow), 
                                                                                           self.storeTypeMinVal)
            if len(overflow) or len(underflow):
                self.numOverflowEvents += 1
                if self.numOverflowEvents < self.MaxOverflowReports:
                    self.logger.warning("counter=%d %s" % (counter, warnMsg))

                if self.numOverflowEvents == self.MaxOverflowReports:
                    self.logger.warning(("EvtTime=%s %s. No more overflow/underflow " + \
                                        "warnings will be printed for this worker") % (counter, warnMsg))

        # now replace 
        self.X[self.nextTimeIdx,:] = data
        old120hz = self.T[self.nextTimeIdx]
        if self.wrapped:
            self.currentTimesStored.remove(old120hz)
        self.T[self.nextTimeIdx] = next120hz
        self.currentTimesStored.add(next120hz)

        firstUpdate = (not self.wrapped) and self.nextTimeIdx==0
        # now add in, now that we have added this in, the smallest element will be one past this if we have wrapped, 
        # or 0 if we haven't wrapped, or are at the end
        if (not self.wrapped) or self.nextTimeIdx == self.maxTimes-1:
            pivotIndex = 0
        else:
            pivotIndex = self.nextTimeIdx+1
        userObj.workerAfterDataInsert(self.nextTimeIdx, pivotIndex, self.numTimesFilled(), self.T, self.X)
            
        if self.nextTimeIdx == self.maxTimes - 1:
            self.wrapped = True
            self.nextTimeIdx = 0
        else:
            self.nextTimeIdx += 1

    def numTimesFilled(self):
        if self.wrapped:
            return self.maxTimes
        else:
            return self.nextTimeIdx
