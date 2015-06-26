'''This module demonstrates how to do the G2 calculation in three ways:

* at the end - workers just store data. Once the maximum number of times 
  has been stored, workers overwrite the oldest stored data. The final
  calculation is done with the most recently stored data during the
  viewer publish callback. This calculation is O(T*D) where T is the number
  of stored times, and D is the number of delays.

* Incrementally, accumulating delays over all time. 

* Incrementally and windowed - the same result as at the end, but done in
  an ongoing fashion.
'''

import os
import numpy as np
import ParCorAna
import psana
import logging
import psmon.config as psmonConfig
import psmon.publish as psmonPublish
import psmon.plots as psmonPlots

def getStats(vec):
    '''returns dict with 'min','5','10',25','med','75','90','95','max' and 'T' (# elem)
    to see what data looks like
    '''
    assert len(vec.shape)==1, "getStats for 1D vector of data"
    svec = np.sort(vec)
    n = len(svec)
    assert n>0, "getStats called on empty vector"
    res = {'T':n}
    res['min']=svec[0]
    res['max']=svec[-1]
    res['med']=svec[min(n-1, max(0, int(n/2)))]
    for percentile in [5,10,25,75,90,95]:
        res[str(percentile)]=svec[min(n-1, max(0, int((percentile/100.0)*n)))]
    return res

def replaceSubsetsWithAverage(A, subsetDict):
    '''Takes a matrix, and a dictionary  of logical indicies for
    the matrix. Replaces each logicalIndex of values with its average.
    '''
    avgA = np.zeros(A.shape, A.dtype)
    for key,subsetInd in subsetDict.iteritems():
        assert subsetInd.shape == A.shape, "replaceSubsetsWithAverage: subset for key=%r has shape=%r != orig shape=%r" % \
            (key, subsetInd.shape, orig.shape)
        assert subsetInd.dtype == np.bool, "replaceSubsetsWithAverage: subset for key=%r does not have type np.bool" % \
            (key,)
        avgVal = np.average(A[subsetInd])
        avgA[subsetInd] = avgVal
    return avgA

####################################
class G2Common(object):
    '''
    '''
    def __init__(self, user_params, system_params, mpiParams, testAlternate):
        self.user_params = user_params
        self.system_params = system_params
        self.mp = mpiParams
        self.logger = self.mp.logger
        self.testAlternate = testAlternate

        self._arrayNames = ['G2','IF','IP']

        self.delays = system_params['delays']
        self.numDelays = len(self.delays)

        self.debugPlot = self.user_params['debug_plot']
        self.maskNdarrayCoords = self.mp.maskNdarrayCoords
        assert self.maskNdarrayCoords.dtype == np.bool
        assert os.path.exists(self.user_params['colorNdarrayCoords']), "color file: %s doesn't exist" % self.user_params['colorNdarrayCoords']
        colorMask = np.load(self.user_params['colorNdarrayCoords']).astype(np.int32) >= 1
        self.debugMask = self.maskNdarrayCoords * colorMask
        self.iX = None
        self.iY = None
        self.fullImageShape = None
        self.debugPlotImageBounds = None
        if self.debugPlot:
            numElemOrig = np.cumprod(self.maskNdarrayCoords.shape)[-1]
            numElemMask = np.sum(self.debugMask)
            self.logInfo("debugPlot=1, debugMask has %d elements (%.2f%% of total)" % \
                         (numElemMask, 100.0*numElemMask/float(numElemOrig)))
            assert 'iX' in user_params, "user_params['debug_plot'] specified, but user_params['iX'] not found. iX file needed (see parCorAnaMaskColorTool)"
            assert 'iY' in user_params, "user_params['debug_plot'] specified, but user_params['iY'] not found. iY file needed (see parCorAnaMaskColorTool)"
            iXfname = user_params['iX']
            iYfname = user_params['iY']
            assert os.path.exists(iXfname), "file user_params['iX']= %s doesn't exist" % iXfname
            assert os.path.exists(iYfname), "file user_params['iY']= %s doesn't exist" % iYfname
            self.iX = np.load(iXfname)
            self.iY = np.load(iYfname)
            assert self.iX.shape == self.debugMask.shape, "loaded iX shape=%s != mask shape=%s" % (self.iX.shape, self.debugMask.shape)
            assert self.iY.shape == self.debugMask.shape, "loaded iY shape=%s != mask shape=%s" % (self.iY.shape, self.debugMask.shape)
            self.fullImageShape, self.debugPlotImageBounds = ParCorAna.imgBoundBox(self.iX, self.iY, self.debugMask)
            

    def logInfo(self, msg, allWorkers=False):
        self.mp.logInfo(msg, allWorkers)

    def logDebug(self, msg, allWorkers=False):
        self.mp.logDebug(msg, allWorkers)

    def logError(self, msg, allWorkers=False):
        self.mp.logError(msg, allWorkers)

    def logWarning(self, msg, allWorkers=False):
        self.mp.logWarning(msg, allWorkers)

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
        assert 'ipimb_threshold_lower' in self.user_params, "must supply ipimb_threshold_lower in user_params for ipimb"
        assert 'ipimb_srcs' in self.user_params, "must supply ipimb_srcs in user_params for list of ipimb srcs (each a str)"
        self.ipimbThresh = self.user_params['ipimb_threshold_lower']
        ipimbSrcStrings = self.user_params['ipimb_srcs']
        self.ipimbSrcs = [psana.Source(srcString) for srcString in ipimbSrcStrings]

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
        for src in self.ipimbSrcs:
            fexData = evt.get(psana.Lusi.IpmFexV1, src)
            if fexData is None:
                self.logWarning("no fex at %r" % src)
                return False
            if fexData.sum() < self.ipimbThresh:
                return False
        return True

    def serverFinalDataArray(self, dataArray, evt):
        '''Callback from the EventIter class. After serverEventOk (above) 
        
        servers calls this to allow user code to validate the event based on the dataArray.
        Return None to say the event should not be processed.
        Return the oringial dataArray to say the event should be processed.
        Optionally, one can make a copy of the dataArray, modify it, and return the copy.
        '''
        if self.logger.isEnabledFor(logging.DEBUG):
            dataForStats = dataArray[self.debugMask].flatten()
            stats = getStats(dataForStats)
            evtId = evt.get(psana.EventId)
            sec, nsec = evtId.time()
            fiducials = evtId.fiducials()
            self.logDebug("dataArray: sec=0x%8.8X fid=0x%5.5X T=%10d min=%5.0f 5=%5.0f 10=%5.0f 25=%5.0f med=%5.0f 75=%5.0f 90=%5.0f 95=%5.0f max=%5.0f" % \
                         (sec, fiducials, stats['T'], stats['min'], stats['5'], stats['10'], stats['25'],
                          stats['med'], stats['75'], stats['90'], stats['95'], stats['max']))
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
        self.G2 = np.zeros((self.numDelays, self.numElementsWorker), np.float32)
        # IP = sum I_i
        self.IP = np.zeros((self.numDelays, self.numElementsWorker), np.float32)
        # IF = sum I_j
        self.IF = np.zeros((self.numDelays, self.numElementsWorker), np.float32)
        # set to 1 when a element arrives that is saturated
        self.saturatedElements = np.zeros(self.numElementsWorker, np.int8)
        # counts, how many pairs of times for each delays
        self.counts = np.zeros(self.numDelays, np.int64)

        self.saturatedValue = self.user_params['saturatedValue']
        self.notzero = self.user_params['notzero']

    def workerBeforeDataRemove(self, tm, xInd, workerData):
        '''called right before data is being overwritten

        Args:
          tm (int):   the counter time of the data being overwritten
          xInd (int): the index into workerData.X of the row about to be overwritten
          workerData: instance of WorkerData with the data stored for this worker

        Return:
          None
        '''
        raise Exception("G2Common.workerBeforeDataRemove not implemented: use sub class")

    def workerAfterDataInsert(self, tm, xInd, workerData):
        '''called after data has been added, and workerAdjustData has been called.

        Args:
          tm (int):   the counter time of the data that has been added
          xInd (int): the index into workerData.X of the row of new data
          workerData: instance of WorkerData with the data stored for this worker

        Return:
          None
        '''
        raise Exception("G2Common.workerAfterDataInsert not implemented: use sub class")

    def workerAdjustData(self, data):
        '''called before data added to WorkerData.X - allows filtering/cleaning of data

        Args:
          data: ndarray of data to be changed. Data must be modified in place.

        Return:
          None
        '''
        indexOfSaturatedElements = data >= self.saturatedValue
        self.saturatedElements[indexOfSaturatedElements]=1
        indexOfNegativeAndSmallNumbers = data < self.notzero   ## FACTOR OUT
        data[indexOfNegativeAndSmallNumbers] = self.notzero
        if self.logger.isEnabledFor(logging.DEBUG):
            numSaturated = np.sum(indexOfSaturatedElements)
            numNeg = np.sum(indexOfNegativeAndSmallNumbers)
            self.logDebug("G2Common.workerAdjustData: set %d negative values to %r, identified %d saturated pixels" % \
                             (numNeg, self.notzero, numSaturated))
        
    def workerCalc(self, workerData):
        '''Must be implemented, returns all output arrays.
        
        Args:
          workerData (WorkerData): object that contains data
        
        Multiple Return::

          namedArrays
          counts
          int8array

          where these output arguments are as follows. Below let D be the number of delays, 
          that is len(system_params['delays'] and numElementsWorker is what was passed in
          during workerInit.
        
          namedArrays - a dictionary, keys are the names returned in fwArrayNames, 
                        i.e, 'G2', 'IF', 'IP'. Values are all numpy arrays of shape
                        (D x numElementsWorker) dtype=np.float32
        
          counts - a 1D array of np.int64, length=numElementsWorker
        
          int8array - a 1D array of np.int8 length=numElementsWorker
        '''
        raise Exception("G2Common.workerCalc not implemented, use sub class")


    ######## TEST FUNCTION ##########
    def calcAndPublishForTestAlt(self,sortedEventIds, sortedData, h5GroupUser):
        '''run the alternative test

        This function receives all the data, including any data that would have been 
        overwritten due to wrap around, with counters, sorted. It should
        implement the G2 calculation in a robust, alternate way, that can then be compared
        to the result of the framework. It should only be run with a small mask file to limit the
        amount of data accumulated.

        The simplest way to compare results is to reuse the viewerPublish function, however one
        may want to write an alternate calculate to test this as well.

        Args:
          sortedEventIds: array of int64, sorted 120hz counters for all the data in the test.
          sortedData: 2D array of all the accummulated, masked data. 
          h5GroupUser (h5group): the output group that one should write results to.

        '''
        raise Exception("G2Common.calcAndPublishForTestAlt Not implemented - use subclass")

    def calcAndPublishForTestAltHelper(self,sortedEventIds, sortedData, h5GroupUser, startIdx):
        '''This implements the alternative test calculation in a way that is suitable for the sub classes
        
        The Incremental will always go through the data from the start, so startingIndex should
        be 0. However the atEnd and Windowed will only work through the last numEvents of 
        data, so startingIndex should be that far back.
        '''
        G2 = {}
        IP = {}
        IF = {}
        counts = {}
        assert len(sortedData.shape)==2
        assert sortedData.shape[0]==len(sortedEventIds)
        assert startIdx < len(sortedEventIds), "startIdx > last index into data"
        numPixels = sortedData.shape[1]
        for delay in self.delays:
            counts[delay]=0
            G2[delay]=np.zeros(numPixels)
            IP[delay]=np.zeros(numPixels)
            IF[delay]=np.zeros(numPixels)

        self.mp.logInfo("UserG2.calcAndPublishForTestAltHelper: starting double loop")

        for idxA, eventIdA in enumerate(sortedEventIds):
            if idxA < startIdx: continue
            counterA = eventIdA['counter']
            for idxB in range(idxA+1,len(sortedEventIds)):
                counterB = sortedEventIds[idxB]['counter']
                delay = counterB - counterA
                if delay == 0:
                    print "warning: unexpected - same counter twice in data - idxA=%d, idxB=%d" % (idxA, idxB)
                    continue
                if delay not in self.delays: 
                    continue
                counts[delay] += 1
                dataA = sortedData[idxA,:]
                dataB = sortedData[idxB,:]
                G2[delay] += dataA*dataB
                IP[delay] += dataA
                IF[delay] += dataB

        self.mp.logInfo("calcAndPublishForTestAlt: finished double loop")

        name2delay2ndarray = {}
        for nm,delay2masked in zip(['G2','IF','IP'],[G2,IF,IP]):
            name2delay2ndarray[nm] = {}
            for delay,masked in delay2masked.iteritems():
                name2delay2ndarray[nm][delay]=np.zeros(self.maskNdarrayCoords.shape, np.float32)
                name2delay2ndarray[nm][delay][self.maskNdarrayCoords] = masked[:]
            
        saturatedElements = np.zeros(self.maskNdarrayCoords.shape, np.int8)
        saturatedElements[self.maskNdarrayCoords] = self.saturatedElements[:]

        lastEventTime = {'sec':sortedEventIds[-1]['sec'],
                         'nsec':sortedEventIds[-1]['nsec'],
                         'fiducials':sortedEventIds[-1]['fiducials'],
                         'counter':sortedEventIds[-1]['counter']}

        countsForViewerPublish = np.zeros(len(counts),np.int)
        for idx, delay in enumerate(self.delays):
            countsForViewerPublish[idx]=counts[delay]
        self.viewerPublish(countsForViewerPublish, lastEventTime, 
                           name2delay2ndarray, saturatedElements, h5GroupUser)


    ######## VIEWER CALLBACKS #############
    def viewerInit(self, maskNdarrayCoords, h5GroupUser):
        '''initialze viewer.
        
        Args:
          maskNdarrayCoords: this is the array in MPI_Params.
          h5GroupUser: if system was given an h5 file for output, this is a h5py group opened
                       into that file. Otherwise this argument is None
        '''
        colorFile = self.user_params['colorNdarrayCoords']        
        assert os.path.exists(colorFile), "user_params['colorNdarrayCoords']=%s not found" % colorFile
        self.color_ndarrayCoords = np.load(colorFile).astype(np.int32)

        finecolorFile = self.user_params['colorFineNdarrayCoords']        
        assert os.path.exists(finecolorFile), "user_params['colorFineNdarrayCoords']=%s not found" % finecolorFile
        self.finecolor_ndarrayCoords = np.load(finecolorFile).astype(np.int32)

        assert self.maskNdarrayCoords.shape == maskNdarrayCoords.shape, "viewerInit maskNdarrayCoords has different shape=%s then what was set in UserG2.init=%s" % (maskNdarrayCoords.shape, self.maskNdarrayCoords.shape)
        assert np.issubdtype(self.color_ndarrayCoords.dtype, np.integer), "color array does not have an integer type."
        assert np.issubdtype(self.finecolor_ndarrayCoords.dtype, np.integer), "finecolor array does not have an integer type."
        assert self.maskNdarrayCoords.shape == self.color_ndarrayCoords.shape, "mask.shape=%s != color.shape=%s" % \
            (self.maskNdarrayCoords.shape, self.color_ndarrayCoords.shape)
        assert self.maskNdarrayCoords.shape == self.finecolor_ndarrayCoords.shape, "mask.shape=%s != finecolor.shape=%s" % \
            (self.maskNdarrayCoords.shape, self.finecolor_ndarrayCoords.shape)

        self.colors = [color for color in set(self.color_ndarrayCoords.flatten()) if color > 0]
        self.colors.sort()
        self.numColors = len(self.colors)

        self.finecolors = [color for color in set(self.finecolor_ndarrayCoords.flatten()) if color > 0]
        self.finecolors.sort()
        self.finenumColors = len(self.colors)

        # set up a dictionary that maps each color to a logical index array (in ndarray coords) of
        # what elements are part of that color. Use the mask to take out masked elements.
        self.color2ndarrayInd = {}
        self.color2numElements = {}
        for color in self.colors:
            logicalThisColor = self.color_ndarrayCoords == color
            # take out masked elements
            logicalThisColor[np.logical_not(self.maskNdarrayCoords)] = False
            self.color2ndarrayInd[color] = logicalThisColor
            self.color2numElements[color] = np.sum(logicalThisColor)

        # set up a dictionary that maps each finecolor to a logical index array (in ndarray coords) of
        # what elements are part of that color. Use the mask to take out masked elements.
        self.finecolor2ndarrayInd = {}
        self.finecolor2numElements = {}
        for finecolor in self.finecolors:
            logicalThisFineColor = self.finecolor_ndarrayCoords == finecolor
            # take out masked elements
            logicalThisFineColor[np.logical_not(self.maskNdarrayCoords)] = False
            self.finecolor2ndarrayInd[finecolor] = logicalThisFineColor
            self.finecolor2numElements[finecolor] = np.sum(logicalThisFineColor)

        self.mp.logInfo("UserG2.viewerInit: colorfile contains colors=%s. Number of elements in each color: %s" % \
                        (self.colors, [self.color2numElements[c] for c in self.colors]))

        self.mp.logInfo("UserG2.viewerInit: finecolorfile contains finecolors=%s. Number of elements in each color: %s" % \
                        (self.finecolors, [self.finecolor2numElements[c] for c in self.finecolors]))

        self.plot = self.user_params['psmon_plot']
        if self.plot or self.debugPlot:
            hostname = os.environ.get('HOSTNAME','*UNKNOWN*')
            port = self.user_params.get('psmon_port',psmonConfig.APP_PORT)
            psmonPublish.init()
            self.mp.logInfo("Initialized psmon. viewer host is: %s" % hostname)
            psplotCmd = 'psplot --logx -s %s -p %s MULTI' % (hostname, port)
            debugPlotCmd = 'psplot -s %s -p %s DEBUG' % (hostname, port)
            self.logInfo("*********** PSPLOT CMD *************")
            if self.plot:
                self.logInfo("Run cmd: %s" % psplotCmd)
            if self.debugPlot:
                self.logInfo("For debug plot, run cmd: %s" % debugPlotCmd)
                tmp = self.debugPlotImageBounds
                rowA,rowB,colA,colB = tmp['rowA'], tmp['rowB'], tmp['colA'], tmp['colB']
                self.logInfo(" will plot image[%d:%d,%d:%d] for IF,IP,G2 for a few delays" % \
                             (rowA, rowB, colA, colB))
            self.logInfo("*********** END PSPLOT CMD *************")
            
        assert 'plot_colors' in self.user_params, "new parameter 'plot_colors' required in config. Set to 'None' to plot all colors, otherwise list like '[1,2,3]'"
        assert 'print_delay_curves' in self.user_params, "new parameter 'print_delay_curves' required in config. Set to 'True' or 'False'"
        self.plotColors = self.user_params['plot_colors']
        assert self.plotColors is None or isinstance(self.plotColors,list), "plot_colors must be None or a list of colors"
        self.printDelayCurves = self.user_params['print_delay_curves']

    def viewerPublish(self, counts, lastEventTime, name2delay2ndarray, 
                      int8ndarray,  h5GroupUser):
        '''results have been gathered from workers. User can now publish, either into 
        h5 group, or by plotting, etc.

        Args:
         counts (array): this is the 1D array of int64, received from the first worker. It is assumed to be
                 the same for all workers. Typically it is the counts of the number of pairs of times
                 that were a given delay apart.
         lastEventTime (dict): has keys 'sec', 'nsec', 'fiducials', and 'counter' for the last event that was
                 scattered to workers before gathering at the viewer.
         counter: 120hz counter for last event before publish

         name2delay2ndarray: this is a 2D dictionary of the gathered, named arrays. For example
                             name2delay2ndarray['G2'][2] will be the ndarray of G2 calcluations for
                             the G2 term.
         int8ndarray: gathered int8 array from all the workers, the pixels they found to be saturated.
         h5GroupUser: either None, or a valid h5py Group to write results into the h5file
        '''

        assert len(counts) == len(self.delays), "UserG2.viewerPublish: len(counts)=%d != len(delays)=%d" % \
            (len(counts), len(self.delays))

        delayCurves = {}
        for color in self.colors:
            if self.color2numElements[color] > 0:
                delayCurves[color] = np.zeros(len(counts), np.float32)
        
        ndarrayShape = self.color_ndarrayCoords.shape
        saturated_ndarrayCoords = int8ndarray
        assert saturated_ndarrayCoords.shape == ndarrayShape, \
            ("UserG2.viewerPublish: gathered saturated_ndarrayCoords.shape=%s != expected=%s" % \
             (saturated_ndarrayCoords.shape, self.color_ndarrayCoords.shape))

        self.maskOutNewSaturatedElements(saturated_ndarrayCoords)

        eps = 1e-6 # protect from division by zero

        counter120hz = lastEventTime['counter']

        if self.debugPlot:
            rowA = self.debugPlotImageBounds['rowA']
            rowB = self.debugPlotImageBounds['rowB']
            colA = self.debugPlotImageBounds['colA']
            colB = self.debugPlotImageBounds['colB']

            # each row is a delay, columns are IF IP G2
            debugPlot = psmonPlots.MultiPlot(counter120hz, 'DEBUG', ncols=3)
            delaysToDo = self.delays[0:3]
            for nm in ['IF','IP','G2']:
                for delay in delaysToDo:
                    ndarr = name2delay2ndarray[nm][delay]
                    fullImage = np.zeros(self.fullImageShape,np.float32)
                    fullImage[self.iX.flatten(), self.iY.flatten()] = ndarr.flatten()[:]
                    image = fullImage[rowA:rowB,colA:colB]
                    title = "cntr=%d %s dly=%d" % (counter120hz, nm, delay)
                    imagePlot = psmonPlots.Image(counter120hz, title, image)
                    debugPlot.add(imagePlot)
                    stats = getStats(image.flatten())
                    self.logInfo("%s: cntr=%d min=%5.3f 5=%5.3f 10=%5.3f 25=%5.3f med=%5.3f 75=%5.3f 90=%5.3f 95=%5.3f max=%5.3f" % \
                                 (title, counter120hz, stats['min'], stats['5'], stats['10'], stats['25'],
                                  stats['med'], stats['75'], stats['90'], stats['95'], stats['max']))
            psmonPublish.send('DEBUG', debugPlot)

        for delayIdx, delayCount in enumerate(counts):
            if delayCount <= 0:
                continue
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
            
            G2 /= np.float32(delayCount)
            IF /= np.float32(delayCount)
            IP /= np.float32(delayCount)

#            fineColorAvg_IF = IF #replaceSubsetsWithAverage(IF, self.finecolor2ndarrayInd)
#            fineColorAvg_IP = IP # replaceSubsetsWithAverage(IP, self.finecolor2ndarrayInd)
            fineColorAvg_IF = replaceSubsetsWithAverage(IF, self.finecolor2ndarrayInd)
            fineColorAvg_IP = replaceSubsetsWithAverage(IP, self.finecolor2ndarrayInd)
            
#            fineColorAvg_IF[fineColorAvg_IF<=eps]=eps
#            fineColorAvg_IP[fineColorAvg_IP<=eps]=eps

            final = G2 / (fineColorAvg_IP * fineColorAvg_IF)

            for color, colorNdarrayInd in self.color2ndarrayInd.iteritems():
                numElementsThisColor = self.color2numElements[color]
                if numElementsThisColor>0:
                    average = np.sum(final[colorNdarrayInd]) / np.float32(numElementsThisColor)
                    delayCurves[color][delayIdx] = average


        if self.printDelayCurves:
            for color in self.color2ndarrayInd.keys():
                if color not in delayCurves: 
                    continue
                self.logInfo("evt=%5d color=%2d delayCurve=%s ... %s" % \
                             (counter120hz, color, ', '.join(map(str,delayCurves[color][0:10])),
                              ', '.join(map(str,delayCurves[color][-10:]))))
                    

        groupName = 'G2_results_at_%6.6d' % counter120hz

        if h5GroupUser is not None:
            createdGroup = False
            try:
                group = h5GroupUser.create_group(groupName)
                createdGroup = True
            except ValueError:
                pass
            if not createdGroup:
                self.mp.logError("Cannot create group  h5 %s. Is viewer update is to frequent?" % groupName)
            else:
                delay_ds = group.create_dataset('delays',(len(self.delays),), dtype='i8')
                delay_ds[:] = self.delays[:]
                delay_counts_ds = group.create_dataset('delay_counts',(len(counts),), dtype='i8')
                delay_counts_ds[:] = counts[:]

                for color in self.colors:
                    if color not in delayCurves: continue
                    delay_curve_color = group.create_dataset('delay_curve_color_%d' % color,
                                                             (len(counts),),
                                                             dtype='f8')
                    delay_curve_color[:] = delayCurves[color][:]

                # write out the G2, IF, IP matrices using framework helper function
                ParCorAna.writeToH5Group(group, name2delay2ndarray)

        if self.plot:
            goodDelays = counts > 0
            multi = psmonPlots.MultiPlot(counter120hz, 'MULTI', ncols=3)
            for color in self.colors:
                if color not in delayCurves: continue
                if (self.plotColors is not None) and (not color in self.plotColors):
                    continue
                thisPlot = psmonPlots.XYPlot(counter120hz, 'color/bin=%d' % color, 
                                             [d/120.0 for d in self.delays[goodDelays]], delayCurves[color][goodDelays], 
                                             xlabel='tau (sec)', ylabel='G2', formats='bs-')
                multi.add(thisPlot)
            psmonPublish.send('MULTI', multi)



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


#################### Incremental Accumulator Calculation #######################        
class G2IncrementalAccumulator(G2Common):
    def __init__(self, user_params, system_params, mpiParams, testAlternate):
        super(G2IncrementalAccumulator,self).__init__(user_params, system_params, mpiParams, testAlternate)
        # Example of printing output:
        # use logInfo, logDebug, logWarning, and logError to log messages.
        # these messages get preceded with the rank and viewer/worker/server.
        # By default, if the message comes from a worker, then only the first 
        # worker logs the message. This reduces noise in the output. 
        # Pass allWorkers=True to have all workers log the message.
        # it is a good idea to include the class and method at the start of log messages
        self.mp.logInfo("G2IncrementalAccumulator: object initialized")

    def workerBeforeDataRemove(self, tm, xInd, workerData):
        pass

    def workerAfterDataInsert(self, tm, xInd, workerData):
        maxStoredTime = workerData.maxTimeForStoredData()
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logInfo("tm=%d xInd=%s new worker data avg: %.2f" % \
                         (tm, xInd, np.average(workerData.X[xInd])), allWorkers=True)
        for delayIdx, delay in enumerate(self.delays):
            if delay > maxStoredTime: break
            tmEarlier = tm - delay
            xIndEarlier = workerData.tm2idx(tmEarlier)
            earlierLaterPairs=[]
            if xIndEarlier is not None:
                earlierLaterPairs.append((xIndEarlier, xInd))
            tmLater = tm + delay
            xIndLater = workerData.tm2idx(tmLater)
            if xIndLater is not None:
                earlierLaterPairs.append((xInd, xIndLater))
            for earlierLaterPair in earlierLaterPairs:
                idxEarlier, idxLater = earlierLaterPair
                intensitiesFirstTime = workerData.X[idxEarlier,:]
                intensitiesLaterTime = workerData.X[idxLater,:]
                self.G2[delayIdx,:] += intensitiesFirstTime * intensitiesLaterTime
                self.IP[delayIdx,:] += intensitiesFirstTime
                self.IF[delayIdx,:] += intensitiesLaterTime
                self.counts[delayIdx] += 1
#            self.mp.logInfo("tm=%s delay=%d G2.min=%.1f G2.avg=%.1f G2.max=%.1f" % (tm, delay, np.min(self.G2[delayIdx,:]), np.average(self.G2[delayIdx,:]), np.max(self.G2[delayIdx,:])), allWorkers=True)
#            self.mp.logInfo("tm=%s delay=%d IF.min=%.1f IF.avg=%.1f IF.max=%.1f" % (tm, delay, np.min(self.IF[delayIdx,:]), np.average(self.IF[delayIdx,:]), np.max(self.IF[delayIdx,:])), allWorkers=True)
#            self.mp.logInfo("tm=%s delay=%d IP.min=%.1f IP.avg=%.1f IP.max=%.1f" % (tm, delay, np.min(self.IP[delayIdx,:]), np.average(self.IP[delayIdx,:]), np.max(self.IP[delayIdx,:])), allWorkers=True)

    def workerCalc(self, workerData):
        return {'G2':self.G2, 'IP':self.IP, 'IF':self.IF}, self.counts, self.saturatedElements

    def calcAndPublishForTestAlt(self, sortedEventIds, sortedData, h5GroupUser):
        self.calcAndPublishForTestAltHelper(sortedEventIds, sortedData, h5GroupUser, startIdx=0)

#################### Incremental Windowed Calculation #######################        
class G2IncrementalWindowed(G2IncrementalAccumulator):
    def __init__(self, user_params, system_params, mpiParams, testAlternate):
        super(G2IncrementalWindowed,self).__init__(user_params, system_params, mpiParams, testAlternate)
        self.mp.logInfo("G2IncrementalWindowed: initialized base (Accumulator) and now Windowed object initialized")

    def workerBeforeDataRemove(self, tm, xInd, workerData):
        maxStoredTime = workerData.maxTimeForStoredData()
        for delayIdx, delay in enumerate(self.delays):
            if delay > maxStoredTime: break
            earlierLaterPairs = []
            tmEarlier = tm - delay
            xIndEarlier = workerData.tm2idx(tmEarlier)
            if xIndEarlier is not None:
                earlierLaterPairs.append((xIndEarlier, xInd))
            tmLater = tm + delay
            xIndLater = workerData.tm2idx(tmLater)
            if xIndLater is not None:
                earlierLaterPairs.append((xInd, xIndLater))
            for earlierLaterPair in earlierLaterPairs:
                idxEarlier, idxLater = earlierLaterPair
                intensitiesEarlier = workerData.X[idxEarlier,:]
                intensitiesLater = workerData.X[idxLater,:]
                assert self.counts[delayIdx] > 0, "G2IncrementalWindowed.workerBeforeDataRemove - about to remove affect at delay=%d but counts=0" % delay
                self.counts[delayIdx] -= 1
                self.G2[delayIdx,:] -= intensitiesEarlier * intensitiesLater
                self.IP[delayIdx,:] -= intensitiesEarlier
                self.IF[delayIdx,:] -= intensitiesLater

        
    def calcAndPublishForTestAlt(self, sortedEventIds, sortedData, h5GroupUser):
        times = self.system_params['times']
        startIdx = max(0, len(sortedData)-times)
        self.calcAndPublishForTestAltHelper(sortedEventIds, sortedData, h5GroupUser, startIdx=startIdx)

############## Straightforward Calculation of all Results at End ############
class G2atEnd(G2Common):
    def __init__(self, user_params, system_params, mpiParams, testAlternate):
        super(G2atEnd,self).__init__(user_params, system_params, mpiParams, testAlternate)
        self.mp.logInfo("G2atEnd: object initialized")

    def workerBeforeDataRemove(self, tm, xInd, workerData):
        pass

    def workerAfterDataInsert(self, tm, xInd, workerData):
        pass

    def workerCalc(self, workerData):
        assert not workerData.empty(), "UserG2.workerCalc called on empty data"
        maxStoredTime = workerData.maxTimeForStoredData()
        
        for delayIdx, delay in enumerate(self.delays):
            if delay > maxStoredTime: break
            for tmA, xIdxA in workerData.timesDataIndexes():
                tmB = tmA + delay
                if tmB > maxStoredTime: break
                xIdxB = workerData.tm2idx(tmB)
                timeNotStored = xIdxB is None
                if timeNotStored: continue
                intensities_A = workerData.X[xIdxA,:]
                intensities_B = workerData.X[xIdxB,:]
                self.counts[delayIdx] += 1
                self.G2[delayIdx,:] += intensities_A * intensities_B
                self.IP[delayIdx,:] += intensities_A
                self.IF[delayIdx,:] += intensities_B

        return {'G2':self.G2, 'IP':self.IP, 'IF':self.IF}, self.counts, self.saturatedElements

    def calcAndPublishForTestAlt(self, sortedEventIds, sortedData, h5GroupUser):
        times = self.system_params['times']
        startIdx = max(0, len(sortedData)-times)
        self.calcAndPublishForTestAltHelper(sortedEventIds, sortedData, h5GroupUser, startIdx=startIdx)
