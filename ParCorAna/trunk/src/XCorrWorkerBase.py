import numpy as np

class XCorrWorkerBase(object):
    SUBTRACT = 1
    ADD = 2
    def __init__(self, numElementsWorker, maxTimes, isFirstWorker, logger):
        self.numElementsWorker = numElementsWorker
        self.maxTimes = maxTimes
        self.isFirstWorker = isFirstWorker
        self.logger = logger
        
        # a 1D array of times in 120hz counter. The 120hz counter for each row of X below
        self.T = np.empty(self.maxTimes, np.int64)

        # a 2D array of element values. X[i,:] is all the detector elements that this worker 
        # processes from time T[i]
        self.X = np.empty((self.maxTimes, numElementsWorker), np.float)

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

    def updateData(self, evtTime, data, userObj):
        '''updates the data.
        '''
        next120hz = np.int64(np.round(evtTime*120.0))
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
            userObj.adjustTerms(XCorrWorkerBase.SUBTRACT, self.nextTimeIdx, self.T, self.X)
            
        self.logger.debug('XCorrWorkerBase.updateData evtTime=%.4f next120hz=%d' % (evtTime, next120hz))

        # allow user object to make adjustments to data
        userObj.adjustData(data)

        # now replace 
        self.X[self.nextTimeIdx,:] = data
        old120hz = self.T[self.nextTimeIdx]
        if self.wrapped:
            self.currentTimesStored.remove(old120hz)
        self.T[self.nextTimeIdx] = next120hz
        self.currentTimesStored.add(next120hz)

        firstUpdate = (not self.wrapped) and self.nextTimeIdx==0
        if not firstUpdate:
            # now add in
            userObj.adjustTerms(XCorrWorkerBase.ADD, self.nextTimeIdx, self.T, self.X)
            
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
