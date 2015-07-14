'''Utility functions for CommSystem
'''
import math
import logging
import time
import datetime
import glob
from mpi4py import MPI
import sys
import os
import numpy as np

def formatFileName(fname):
    '''Looks for %T in a file and %C. Replaces them with 

     %T  -> yyyymmddhhmmss (year, month, day, hour, minute, second)
     %C  -> look at files on disk - use next one up counter

    ignore .inprogress when adding counters
    '''
    splitT = fname.split('%T')
    assert len(splitT)<=2, "Doesn't make sense to have more than one %%T in fname: %s" % fname
    if len(splitT)==2:
        dt=datetime.datetime.fromtimestamp(time.time())
        timestamp='%4.4d%2.2d%2.2d%2.2d%2.2d%2.2d' % (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        fname = timestamp.join(splitT)
    splitC = fname.split('%C')
    assert len(splitC)<=2, "Doesn't make sense to have more than one %%C in fname: %s" % fname
    if len(splitC)<2: return fname
    beforeC,afterC = splitC
    globmatch = fname.replace('%C','*')
    globfilesA = glob.glob(globmatch)
    globfilesB = glob.glob(globmatch+'.inprogress')
    globfiles = globfilesA + [fname[0:-11] for fname in globfilesB]
    curCounters = []
    for globfname in globfiles:
        counterMatch = globfname[len(beforeC):-len(afterC)]
        try:
            counter = int(counterMatch)
            if counter >=0 and counter <= 999:
                curCounters.append(counter)
            else:
                pass
                # there is other stuff in here, could be a timestamp that looks like a big int
        except ValueError:
            pass
    if len(curCounters)==0:
        nextCounter=0
    else:
        nextCounter = max(curCounters)+1
    fname = beforeC + ("%3.3d" % nextCounter) + afterC
    return fname

            
def checkCountsOffsets(counts, offsets, n):    
    '''Makes sure that the counts and offsets partition n. 
    
    Throws an exception if there is a problem

    Examples:
      >>> checkCountsOffsets(counts=[2,2,2], offsets=[0,2,4], n=6)
      # this is correct. 
      >>> checkCountsOffsets(counts=[2,2,2], offsets=[0,2,4], n=7)
      # incorrect, throws assert
      >>> checkCountsOffsets(counts=[2,2,2], offsets=[2,4,6], n=6)
      # incorrect, ect
    '''
    assert sum(counts)==n, 'counts=%r offsets=%r do not partition n=%d' % (counts, offsets, n)
    assert offsets[0]==0, 'counts=%r offsets=%r do not partition n=%d' % (counts, offsets, n)
    assert len(counts)==len(offsets), 'counts=%r offsets=%r do not partition n=%d' % (counts, offsets, n)
    for i in range(1,len(counts)):
        assert offsets[i]==offsets[i-1]+counts[i-1], 'counts=%r offsets=%r do not partition n=%d' % (counts, offsets, n)
    assert offsets[-1]+counts[-1]==n, 'counts=%r offsets=%r do not partition n=%d' % (counts, offsets, n)

def divideAmongWorkers(dataLength, numWorkers):
    '''partition the amount of data as evenly as possible among the given number of workers.

    Examples: 
      >>> divideAmongWorkers(11,3) 
      returns offsets=[0,4,8]
              counts=[4,4,3]    
    '''
    assert numWorkers > 0, "dividAmongWorkers - numWorkers is <= 0"
    k = int(math.floor(dataLength / numWorkers))
    r = dataLength % numWorkers
    offsets=[]
    counts=[]
    nextOffset = 0
    for w in range(numWorkers):
        offsets.append(nextOffset)
        count = k
        if r > 0: 
            count += 1
            r -= 1
        counts.append(count)
        nextOffset += count
    checkCountsOffsets(counts, offsets, dataLength)
    return offsets, counts            
 
loggers = {}
   
def makeLogger(isTestMode, isMaster, isViewer, isServer, rank, lvl='INFO', propagate=False):
    '''Returns Python logger with prefix identifying master/viewer/server/worker & rnk

    If isTestMode is true, returns logger with 'noMpi' prefix.

    Args:
     isTestMode (bool): True if this is test mode. only additional parameter used
                        is lvl in this case.
     isMaster (bool): True if this is logger for master rank
     isViewer (bool): True if this is logger for viewer rank
     isServer (bool): True is this logger is for server rank
     rank (bool): rank to report
     lvl (str): logging level
     propogate (bool): True is logging message should propogage to parent

    Return:
      (logger): python logging logger is returned with formatting describing role in framework, and time of logged messages.
    '''
    numLevel = None
    if hasattr(logging,lvl.upper()) and isinstance(getattr(logging, lvl.upper()), int):
        numLevel = getattr(logging, lvl.upper())
    else:
        validLevels = ','.join([attr for attr in dir(logging) if attr.isupper() and isinstance(getattr(logging, attr),int)])
        raise ValueError("invalid verbosity/logging level=%s. valid levels are: %s" % (lvl, validLevels))
    global loggers

    loggerName = None
    if isTestMode:
        loggerName = 'testmode'
    else:
        assert int(isMaster) + int(isViewer) + int(isServer) in [0,1], "more than one of isMaster, isViewer, isServer is true"
        if isMaster:
            loggerName = 'master-rnk:%d' % rank
        elif isViewer:
            loggerName = 'viewer-rnk:%d' % rank
        elif isServer:
            loggerName = 'server-rnk:%d' % rank
        else:
            loggerName = 'worker-rnk:%d' % rank

    logger = loggers.get(loggerName,None)
    if logger is not None:
        return logger

    logger = logging.getLogger(loggerName)
    logger.setLevel(numLevel)
    logger.propagate=propagate
    ch = logging.StreamHandler()
    ch.setLevel(numLevel)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s' )
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(ch)
    loggers[loggerName]=logger
    return logger

def checkParams(system_params, user_params, checkUserParams=False):
    '''Checks for correct keys in system_params. Optionally checks userParams.

    When checking userParams, checks against UserG2 module in this package. 
    Loads mask file from system_params and checks its shape against that of
    the color and finecolor files in the user_params. Checks for consistency in
    the color/finecolor file. Warns of 'pixel waste', pixels that are 0 in the color
    files but 1 in the mask file (These pixels could be masked out without changing the
    resulting delay curves).
    '''
    expectedSystemKeys = set(['dataset',
                              'src',
                              'psanaType', 
                              'ndarrayProducerOutKey',
                              'ndarrayCalibOutKey', 
                              'psanaOptions',
                              'outputArrayType', 
                              'workerStoreDtype',
                              'maskNdarrayCoords',
                              'testMaskNdarrayCoords',
                              'numServers', 
                              'serverHosts',
                              'serversRoundRobin',
                              'times', 
                              'update',
                              'delays',
                              'h5output', 
                              'testH5output',
                              'overwrite',
                              'verbosity', 
                              'numEvents', 
                              'userClass',
                              'testNumEvents'])

    undefinedSystemKeys = expectedSystemKeys.difference(set(system_params.keys()))
    newSystemKeys = set(system_params.keys()).difference(expectedSystemKeys)
    assert len(undefinedSystemKeys)==0, "Required keys are not in system_params: %r" % \
        (undefinedSystemKeys,)
    if len(newSystemKeys)>0 and MPI.COMM_WORLD.Get_rank()==0:
        sys.stderr.write("Warning: unexpected keys in system_params: %r\n" % (newSystemKeys,))
    if checkUserParams:
        
        assert 'colorNdarrayCoords' in user_params
        assert 'colorFineNdarrayCoords' in user_params
        assert os.path.exists(system_params['maskNdarrayCoords']), "maskNdarrayCoords file:  '%s' doesn't exist" % system_params['maskNdarrayCoords']
        assert os.path.exists(user_params['colorNdarrayCoords']), "colorNdarrayCoords file:  '%s' doesn't exist" % user_params['colorNdarrayCoords']
        assert os.path.exists(user_params['colorFineNdarrayCoords']), "colorFineNdarrayCoords file:  '%s' doesn't exist" % user_params['colorFineNdarrayCoords']
        mask = np.load(system_params['maskNdarrayCoords']).astype(np.int8)
        color = np.load(user_params['colorNdarrayCoords']).astype(np.int32)
        finecolor = np.load(user_params['colorFineNdarrayCoords']).astype(np.int32)
        
        assert mask.shape == color.shape, "mask shape=%r != color.shape=%r" % (mask.shape, color.shape)
        assert color.shape == finecolor.shape, "color.shape=%r != finecolor.shape=%r" % (color.shape, finecolor.shape)

        MAXCOLOR = 1<<14 # arbitrary, but to try to warn against corrupt files

        assert np.min(color)>=0, "color file contains values < 0"
        assert np.min(finecolor)>=0, "color file contains values < 0"
        assert np.max(color)<=MAXCOLOR, "color file contains values > %d" % MAXCOLOR
        assert np.max(finecolor)<=MAXCOLOR, "color file contains values > %d" % MAXCOLOR
        
        numFineColorThatAreZeroWhereColorIsNonZero = np.sum(finecolor[color > 0] == 0)
        assert numFineColorThatAreZeroWhereColorIsNonZero == 0, "finecolor file has pixels that are zero where color file is nonzero"
        mask_flat = mask.flatten()
        maskValues = set(mask_flat)
        assert maskValues.union(set([0,1])) == set([0,1]), "mask contains values other than 0 and 1." + \
            (" mask contains %d distinct values" % len(maskValues))
        assert 1 in maskValues, "The mask does not have the value 1, it is all 0. Elements marked with 1 are processed"
        maskNdarrayCoords = mask == 1 # is is important to convert mask to array of bool, np.bool
        masked_color = mask * color
        pixel_waste =  np.sum(masked_color <= 0)
        pixels_on = np.sum(masked_color > 0)
        if pixel_waste > 0 and MPI.COMM_WORLD.Get_rank() == 0:
            sys.stderr.write(("Warning: there are %d or %.1f%% pixels that are 0 or < 0 in the color file: "+
                             "%s that are not 0 in the mask file: %s. These pixels will still be "+
                              "procssed by workers, but not the viewer. Consider turning them off in the " + 
                              "mask file. There are %d mask included pixels that are on in the color file.\n") % \
                             (pixel_waste, 100.0*pixel_waste/float(mask_flat.shape[0]),
                              user_params['colorNdarrayCoords'],
                              system_params['maskNdarrayCoords'],
                              pixels_on))
        

def imgBoundBox(iX, iY, maskNdArrayCoords):
    '''returns bounding box in image space for ndarray mask
    
    Takes three matricies, all ndarray's
    ARGS:
      iX     gives row/dim0 in image space for each pixel in ndarray
      iY     gives row/dim1 in image space for each pixel in ndarray

      maskNdArrayCoords   0/1 mask

    returns dictionary with 'rowA', 'rowB', 'colA', 'colB' for bounding
    box of mask in image space
    '''
    assert iX.shape == iY.shape
    assert iY.shape == maskNdArrayCoords.shape
    numImageRows = np.max(iX)+1
    numImageCols = np.max(iY)+1
    fullImageShape = (numImageRows, numImageCols)
    maskImage = np.zeros(fullImageShape, dtype=np.int)
    maskImage[iX.flatten(), iY.flatten()] = maskNdArrayCoords.astype(np.int8).flatten()[:]
    rowSums = np.sum(maskImage,1)
    colSums = np.sum(maskImage,0)
    rowNonZero = np.where(rowSums>0)[0]
    colNonZero = np.where(colSums>0)[0]
    bounds = {'rowA':np.min(rowNonZero),
              'rowB':np.max(rowNonZero),
              'colA':np.min(colNonZero),
              'colB':np.max(colNonZero)}
    return fullImageShape, bounds


def replaceSubsetsWithAverage(A, labels, label2total=None):
    '''Returns matrix based on A, where each element gets average over pixels with the same label
    ARGS:
      A           - numpy array of values to average over labeled subsets
      labels      - numpy array of ints, >= 0, same shape as A, each int labels a subset
      label2total - optional dictionary of number of pixels in each label in labels
    RETURN:
      new matrix of averages values 
    '''
    assert A.shape == labels.shape
    if label2total == None:
        labelCounts = np.bincount(labels.flatten())
        label2total = {}
        for label,count in enumerate(labelCounts):
            label2total[label]=count
    groupedAverages = np.bincount(labels.flatten(), A.flatten())
    for label, count in label2total.iteritems():
        if label >=0 and label < len(groupedAverages):
            groupedAverages[label] /= float(count)
    avgA_dtype = np.float32
    if A.dtype == np.float64:
        avgA_dtype = np.float64
    avgA = np.zeros(A.size, dtype=avgA_dtype)
    avgA[:] = groupedAverages[labels.flatten()]
    avgA.resize(A.shape)
    return avgA

