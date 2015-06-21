import os
import numpy as np
import ParCorAna
import psana
import logging
import psmon.config as psmonConfig
import psmon.publish as psmonPublish
import psmon.plots as psmonPlots

####################################
class G2MPITest(object):
    '''
    '''
    def __init__(self, user_params, system_params, mpiParams, testAlternate):
        self.user_params = user_params
        self.system_params = system_params
        self.mp = mpiParams
        self.testAlternate = testAlternate

        self._arrayNames = ['G2','IF','IP']

        self.delays = system_params['delays']
        self.numDelays = len(self.delays)

    ### COMMON CALLBACK ####
    # callbacks used by multiple roles, i.e, workers and viewer
    def fwArrayNames(self):
        '''Return a list of names for the arrays calculated. 

        This is how the framework knows how many output arrays the user module creates.
        These must be the same names that workerCalc returns in its dictionary
        of named arrays.

        These will be the names the framework passes to the viewer with the
        assembed arrays.
        '''
        return self._arrayNames

    #### SERVER CALLBACKS #####
    # These callbacks are used on server ranks
    def serverInit(self):
        '''Called after framework initializes server.
        '''
        pass

    def serverEventOk(self, evt):
        '''tells framework if this event is suitable for including in calculation.

        This is a callback from the EventIter class. Return True if an event is Ok to  proceeed with.
        This is called before the detector data array is retreived from the event.
        Although one could look at the detector data here, best practice is to do any filtering 
        that does not require the data array here.

        Filtering based on the data array should be done in :meth:`ParCorAna.UserG2:serverFinalDataArray`

        To use the testing part of the framework, it is a good idea to make this callback a wrapper
        around a function that is accessible to the testing class as well.

        Args:
           evt (psana.Event): a psana event.

        Return:
          (bool): True if this event should be included in analysis
        '''
        return True

    def serverFinalDataArray(self, dataArray, evt):
        '''Callback from the EventIter class. After serverEventOk (above) 
        
        servers calls this to allow user code to validate the event based on the dataArray.
        Return None to say the event should not be processed.
        Return the oringial dataArray to say the event should be processed.
        Optionally, one can make a copy of the dataArray, modify it, and return the copy.
        '''
        return dataArray

    ######## WORKER CALLBACKS #########
    def workerInit(self, numElementsWorker):
        '''initialize UserG2 on a worker rank

        Args:
          numElementsWorker (int): number of elements this worker processes
        '''
        self.numElementsWorker = numElementsWorker

        ## define the output arrays returned by this worker
        # G2 = sum I_i * I_j 
        self.G2 = np.ones((self.numDelays, self.numElementsWorker), np.float32)
        # IP = sum I_i
        self.IP = np.ones((self.numDelays, self.numElementsWorker), np.float32)
        # IF = sum I_j
        self.IF = np.ones((self.numDelays, self.numElementsWorker), np.float32)
        # set to 1 when a element arrives that is saturated
        self.saturatedElements = np.zeros(self.numElementsWorker, np.int8)
        # counts, how many pairs of times for each delays
        self.counts = np.ones(self.numDelays, np.int64)

        self.saturatedValue = self.user_params['saturatedValue']
        self.notzero = self.user_params['notzero']

    def workerBeforeDataRemove(self, tm, xInd, workerData):
        pass

    def workerAfterDataInsert(self, tm, xInd, workerData):
        pass

    def workerAdjustData(self, data):
        pass
        
    def workerCalc(self, workerData):
        return {'G2':self.G2, 'IP':self.IP, 'IF':self.IF}, self.counts, self.saturatedElements


    ######## TEST FUNCTION ##########
    def calcAndPublishForTestAlt(self,sortedEventIds, sortedData, h5GroupUser):
        pass

    ######## VIEWER CALLBACKS #############
    def viewerInit(self, maskNdarrayCoords, h5GroupUser):
        pass

    def viewerPublish(self, counts, lastEventTime, name2delay2ndarray, 
                      int8ndarray,  h5GroupUser):
        pass

    def calcAndPublishForTestAlt(self, sortedEventIds, sortedData, h5GroupUser):
        self.calcAndPublishForTestAltHelper(sortedEventIds, sortedData, h5GroupUser, startIdx=0)

