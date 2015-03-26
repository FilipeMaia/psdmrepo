'''Example of implementing user code for the G2 calculation.

Notes:
  An import of this module should not do anything complicated. Delay anything
  complicated until the __init__ method or other methods of the class. The reason 
  for this is that an import of the module is done when reading the params.py file.
  An import should not depend on MPI or take a long amount of time.
'''

import os
import numpy as np
# to get the definition of SUBTRACT and ADD:
import ParCorAna
import ParCorAna.XCorrWorkerBase as XCorrWorkerBase

##################################
### helper functions for UserG2 class below
def getMatchingIndex(delay, iiTimeIdx, n, timesSortIdx, T):
    '''This is a helper function to workerCalc.

    Assumes that timesSortIdx is a sorted order of T.

    Args:
      delay (int):     delay to match
      iiTimeIdx (int): index into T for start time
      n (int):         number of times to consider in T
      timesSortIdx (list/array): sorted order for T
      T (list/array):          T values

    Return:
      index into T for time such that `T[index]-T[iiTimeIdx]==delay`
      or None if no such time exists in T.

    '''
    jjTimeIdx = iiTimeIdx + delay
    if jjTimeIdx >= n:
        jjTimeIdx = n-1
    iiTime = T[iiTimeIdx]
    while (T[jjTimeIdx]-iiTime) > delay and jjTimeIdx > iiTimeIdx:
        jjTimeIdx -= 1
    if T[jjTimeIdx]-iiTime == delay:
        return jjTimeIdx
    return None

####################################
class UserG2(object):
    '''called on each rank after mpi parameters are identified.
    
    Keep code here simple. This class is called for both the viewer rank
    and all the worker ranks. Initialize variables specific fo a worker or viewer
    in the initWorker() and initViewer() callbacks.
    '''
    def __init__(self, user_params, system_params, mpiParams):
        self.user_params = user_params
        self.system_params = system_params
        self.mp = mpiParams
        
        self._arrayNames = ['G2','IF','IP']

        self.delays = system_params['delays']
        self.numDelays = len(self.delays)

        # Example of printing output:
        # use logInfo, logDebug, logWarning, and logError to log messages.
        # these messages get preceded with the rank and viewer/worker/server.
        # By default, if the message comes from a worker, then only the first 
        # worker logs the message. This reduces noise in the output. 
        # Pass allWorkers=True to have all workers log the message.
        # it is a good idea to include the class and method at the start of log messages
        self.mp.logInfo("UserG2.init: initialized")

    ########## CALLBACKS ###########
    # The framework will call these functions. Some are called only on 
    # server ranks, some only on worker ranks, and some only on the viewer rank. The
    # first, arrayNames, may be called on all ranks. 
    #
    # In the future, some more technical functions may be called on both 
    # workers/server or workers/viewer to allow the user to implement their own
    # collective communication.
    #
    # All these functions need to be implemented.
    def arrayNames(self):
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
    def initServer(self):
        '''Called after framework initializes server.
        '''
        pass

    def eventOk(self, evt):
        '''Callback from the EventIter class. 
        Return True if an event is Ok to  proceeed with.
        This is called before the detector data array is retreived from the event.
        Do any filtering that does not require the data array here.
        Filtering based on the data array should be done in :meth:`ParCorAna.UserG2:finalDataArray`

        Args:
           evt (psana.Event): a psana event.
        '''
        return True

    def finalDataArray(self, dataArray, evt):
        '''Callback from the EventIter class. After eventOk (above) 
        
        servers calls this to allow user code to validate the event based on the dataArray.
        Return None to say the event should not be processed, or the oringial dataArray
        if it is Ok. Optionally, one can make a copy of the dataArray, modify it, and return
        the copy.
        '''
        return dataArray

    ######## WORKER CALLBACKS #########
    def initWorker(self, numElementsWorker):
        '''initialize UserG2 on a worker rank

        Args:
          numElementsWorker (int): number of elements this worker processes
        '''
        self.numElementsWorker = numElementsWorker

        ## define the output arrays returned by this worker
        # G2 = sum I_i * I_j 
        self.G2 = np.zeros((self.numDelays, self.numElementsWorker), np.float)
        # IP = sum I_i
        self.IP = np.zeros((self.numDelays, self.numElementsWorker), np.float)
        # IF = sum I_j
        self.IF = np.zeros((self.numDelays, self.numElementsWorker), np.float)
        # set to 1 when a element arrives that is saturated
        self.saturatedElements = np.zeros(self.numElementsWorker, np.int8)
        # counts, how many pairs of times for each delays
        self.counts = np.zeros(self.numDelays, np.int64)

        self.saturatedValue = self.user_params['saturatedValue']
        self.notzero = self.user_params['notzero']

    def adjustTerms(self, mode, dataIdx, pivotIndex, lenT, T, X):
        '''called before and after X[dataIdx,:] is filled with new data scattered to this
        worker. Allows the user to see what new data has been written, and what the time is
        for that data. Once the T buffer wraps and the framework is overwriting the oldest data, 
        allows the user to see the old data getting removed (useful for maintaining a 'windowed'
        correlation analysis rather than an accumulation of delays over all time.

        Note, T is a ciruclarly sorted array. If the configuration parameters defined T to be a
        500 long array, and we are at event 600, then T[0:100] are the times for the most
        recent 100 events (events 501 to 600) while T[101:] are the times for events 101 to 500.

        When working with T, the three things one needs to know are how much of T has been filled so
        far (lenT argument to function), where the 'pivotIndex' is, and which index the being used 
        for the new data for the worker (the dataIdx argument to the function).
        
        In the above example, lenT=500, pivotIndex = 99 and dataIdx = 99. Before we wrap around in
        T, i.e, for evenst 1 to 500 in the example above, pivotIdx=0, and T is effectively a sorted array.
        But after we wrap around, T is a true sorted circular array where dataIdx will most always be equal 
        to pivotIndex. When dataIdx is not equal to pivotIdx can occur if the DAQ is sending events out
        of order, or due to how we are getting events, we read them out of order. In this case dataIdx
        could be below or above the pivotPoint, however the framework will take care of moving things around
        in X and T to ensure that T is a circularly sorted array (WARNING: not implemented yet).

        Args:
         T:        1D Times array, values are the 120hz int64 counters for events. 
                   T[dataIdx] is the value about to be overwritten (for SUBTRACT) or the value that
                   has just been replaced, or appended during the first time through T (for ADD)
         X:        2D data array, X[dataIdx,:] is the data about to replaced during SUBTRACT, or that has 
                   just been copied in from a server (for ADD)
         dataIdx:  the index into T and X of the time/data about to be removed (SUBTRACT), or that has just
                   been added (ADD)
         pivotIndex: the index for the smallest value in the Times array right now. Note - this can
                     change from the SUTRACT call to the ADD call.
         lenT:      the length of T/X that is being used so far. Will be less than the lenght of T during the
                   runup before we wrap.
         mode:     SUBTRACT/ADD, as discussed above.
        '''
        if mode == XCorrWorkerBase.SUBTRACT:
            pass
        elif mode == XCorrWorkerBase.ADD:
            pass

    def adjustData(self, data):
        self.saturatedElements[data >= self.saturatedValue]=1
        data[data < self.notzero] = self.notzero
        
    def workerCalc(self, T, numTimesFilled, X):
        '''Must be implemented, returns all output arrays.
        
        Args:
          T: 1D array of int64, the 120hz counter identifying the events. Note - 
              T is not sorted.
          numTimesFilled: T may not be completely filled out. This is the number
              of entries in T that are.
          X: the data
        
        Multiple Return::

          namedArrays
          counts
          int8array

          where these output arguments are as follows. Below let D be the number of delays, 
          that is len(system_params['delays'] and numElementsWorker is what was passed in
          during initWorker.
        
          namedArrays - a dictionary, keys are the names returned in arrayNames, 
                        i.e, 'G2', 'IF', 'IP'. Values are all numpy arrays of shape
                        (D x numElementsWorker) dtype=np.float64
        
          counts - a 1D array of np.int64, length=numElementsWorker
        
          int8array - a 1D array of np.int8 length=numElementsWorker
        '''
        allTimes = T
        n = numTimesFilled
        if n < len(allTimes):
            allTimes = T[0:n]
        timesSortIdx = np.argsort(allTimes)
        self.G2[:]=0.0
        self.IP[:]=0.0
        self.IF[:]=0.0
        self.counts[:]=0

        for delayIdx, delay in enumerate(self.delays):
            for ii in range(n-delay):
                iiTimeIdx = timesSortIdx[ii]
                jjTimeIdx = getMatchingIndex(delay, iiTimeIdx, n, timesSortIdx, T)
                if jjTimeIdx is None:
                    continue
                assert T[jjTimeIdx] - T[iiTimeIdx] == delay, "getMatchingIndex failed"
                I_ii = X[iiTimeIdx,:]
                I_jj = X[jjTimeIdx,:]
                self.counts[delayIdx] += 1
                self.G2[delayIdx,:] += I_ii * I_jj
                self.IP[delayIdx,:] += I_ii
                self.IF[delayIdx,:] += I_jj
                
        return {'G2':self.G2, 'IP':self.IP, 'IF':self.IF}, self.counts, self.saturatedElements

    ######## VIEWER CALLBACKS #############
    def initViewer(self, mask_ndarrayCoords, h5GroupUser):
        '''initialze viewer.
        
        Args:
          mask_ndarrayCoords: this is the array in MPI_Params.
          h5GroupUser: if system was given an h5 file for output, this is a h5py group opened
                       into that file. Otherwise this argument is None
        '''
        colorFile = self.user_params['color_ndarrayCoords']        
        assert os.path.exists(colorFile), "user_params['color_ndarrayCoords']=%s not found" % colorFile
        self.color_ndarrayCoords = np.load(colorFile)
        self.mask_ndarrayCoords = mask_ndarrayCoords
        assert np.issubdtype(self.color_ndarrayCoords.dtype, np.integer), "color array does not have an integer type."
        assert self.mask_ndarrayCoords.shape == self.color_ndarrayCoords.shape, "mask.shape=%s != color.shape=%s" % \
            (self.mask_ndarrayCoords.shape, self.color_ndarrayCoords.shape)
        self.colors = [color for color in set(self.color_ndarrayCoords.flatten()) if color > 0]
        self.colors.sort()
        self.numColors = len(self.colors)
        # set up a dictionary that maps each color to a logical index array (in ndarray coords) of
        # what elements are part of that color. Use the mask to take out masked elements.
        self.color2ndarrayInd = {}
        self.color2numElements = {}
        for color in self.colors:
            logicalThisColor = self.color_ndarrayCoords == color
            # take out masked elements
            logicalThisColor[np.logical_not(self.mask_ndarrayCoords)] = False
            self.color2ndarrayInd[color] = logicalThisColor
            self.color2numElements[color] = np.sum(logicalThisColor)

        self.mp.logInfo("UserG2.initViewer: colorfile contains colors=%s. Number of elements in each color: %s" % \
                        (self.colors, [self.color2numElements[c] for c in self.colors]))
        

    def viewerPublish(self, counts, relsec, name2delay2ndarray, 
                      int8ndarray,  h5GroupUser):
        '''results have been gathered from workers. User can now publish, either into 
        h5 group, or by plotting, etc.

        Args:
         counts: this is the 1D array of int64, received from the first worker. It is assumed to be
                 the same for all workers. Typically it is the counts of the number of pairs of times
                 that were a given delay apart.
         relsec: roughly how many seconds of data have been processed for this viewerPublish call.
                 It is a floating point seconds relative floor(seconds) of the first event.

         name2delay2ndarray: this is a 2D dictionary of the gathered, named arrays. For example
                             name2delay2ndarray['G2'][2] will be the ndarray of G2 calcluations for
                             the G2 term.
         int8ndarray: gathered int8 array from all the workers, the pixels they found to be saturated.
         h5GroupUser:  either None, or a valid h5py Group to write results into the h5file
        '''

        assert len(counts) == len(self.delays), "UserG2.viewerPublish: len(counts)=%d != len(delays)=%d" % \
            (len(counts), len(self.delays))

        delayCurves = {}
        for color in self.colors:
            if self.color2numElements[color] > 0:
                delayCurves[color] = np.zeros(len(counts), np.float)
        
        ndarrayShape = self.color_ndarrayCoords.shape
        saturated_ndarrayCoords = int8ndarray
        assert saturated_ndarrayCoords.shape == ndarrayShape, \
            ("UserG2.viewerPublish: gathered saturated_ndarrayCoords.shape=%s != expected=%s" % \
             (saturated_ndarrayCoords.shape, self.color_ndarrayCoords.shape))

        self.maskOutNewSaturatedElements(saturated_ndarrayCoords)

        for delayIdx, delayCount in enumerate(counts):
            delay = self.delays[delayIdx]
            G2 = name2delay2ndarray['G2'][delay]
            IF = name2delay2ndarray['IF'][delay]
            IP = name2delay2ndarray['IP'][delay]
            assert G2.shape == ndarrayShape, "UserG2.viewerPublish: G2.shape=%s != expected=%s" % \
                (G2.shape, ndarrayShape)
            assert IF.shape == ndarrayShape, "UserG2.viewerPublish: IF.shape=%s != expected=%s" % \
                (IF.shape, ndarrayShape)
            assert IP.shape == ndarrayShape, "UserG2.viewerPublish: IP.shape=%s != expected=%s" % \
                (IP.shape, ndarrayShape)
            
            G2 /= float(delayCount)
            IF /= float(delayCount)
            IP /= float(delayCount)

            final = G2 / (IP * IF)

            for color, colorNdarrayInd in self.color2ndarrayInd.iteritems():
                numElementsThisColor = self.color2numElements[color]
                if numElementsThisColor>0:
                    average = np.sum(final[colorNdarrayInd]) / float(numElementsThisColor)
                    delayCurves[color][delayIdx] = average


        if h5GroupUser is not None:
            assert relsec is not None
            counter120hz = int(np.round(120*relsec))
            groupName = 'G2_results_at_%d' % counter120hz
            try:
                group = h5GroupUser.create_group(groupName)
            except ValueError:
                self.mp.logError("Cannot create group  h5 %s. Is viewer update is to frequent?" % groupName)
                return

            delay_ds = group.create_dataset('delays',(len(self.delays),), dtype='i8')
            delay_ds[:] = self.delays[:]
            delay_counts_ds = group.create_dataset('delay_counts',(len(counts),), dtype='i8')
            delay_counts_ds[:] = counts[:]
            for color in self.colors:
                delay_curve_color = group.create_dataset('delay_curve_color_%d' % color,
                                                         (len(counts),),
                                                         dtype='f8')
                delay_curve_color[:] = delayCurves[color][:]

            # write out the G2, IF, IP matrices using framework helper function
            ParCorAna.writeToH5Group(group, name2delay2ndarray)


    ######## VIEWER HELPERS (NOT CALLBACKS, JUST USER CODE) ##########
    def maskOutNewSaturatedElements(self, saturated_ndarrayCoords):
        '''update masks and counts for each color based on new saturated elements

        Args:
          saturated_ndarrayCoords: an int8 with the detector ndarray shape. 
                                   positive values means this is a saturated pixels.
        '''
        
        saturatedIdx = saturated_ndarrayCoords > 0
        if np.sum(saturated_ndarrayCoords)==0: 
            # no saturated pixels
            return

        numColorsChanged = 0
        numNewSaturatedElements = 0
        for color, colorNdarrayInd in self.color2ndarrayInd.iteritems():
            numElementsThisColor = self.color2numElements[color]
            colorNdarrayInd[saturatedIdx]=False
            self.color2numElements[color] = np.sum(colorNdarrayInd)
            if self.color2numElements[color] < numElementsThisColor:
                numColorsChanged += 1
                numNewSaturatedElements += (numElementsThisColor - self.color2numElements[color])
        if numNewSaturatedElements > 0:
            self.mp.logInfo("UserG2.maskOutNewSaturatedElements: removed %d elements from among %d colors" % \
                            (numNewSaturatedElements, numColorsChanged))
