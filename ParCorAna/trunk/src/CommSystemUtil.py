'''Utility functions for CommSystem
'''
import math
import logging

def checkCountsOffsets(counts, offsets, n):    
    '''Makes sure that the counts and offsets partition n. For instance
    counts=[2,2,2] offsets=[0,2,4] n=6 is correct. However
    counts=[2,2,2] offsets=[0,2,4] n=7 is incorrect,
    counts=[2,2,2] offsets=[2,4,6] n=6 is incorrect, ect
    
    Throws an exception if there is a problem
    '''
    assert sum(counts)==n, 'counts=%r offsets=%r do not partition n=%d' % (counts, offsets, n)
    assert offsets[0]==0, 'counts=%r offsets=%r do not partition n=%d' % (counts, offsets, n)
    assert len(counts)==len(offsets), 'counts=%r offsets=%r do not partition n=%d' % (counts, offsets, n)
    for i in range(1,len(counts)):
        assert offsets[i]==offsets[i-1]+counts[i-1], 'counts=%r offsets=%r do not partition n=%d' % (counts, offsets, n)
    assert offsets[-1]+counts[-1]==n, 'counts=%r offsets=%r do not partition n=%d' % (counts, offsets, n)

def divideAmongWorkers(dataLength, numWorkers):
    '''partition the amount of data as evenly as possible among the given number of workers.
    Returns counts and offsets of the partition.

    example: divideAmongWorkers(11,3) 
    returns offsets=[0,4,8]
             counts=[4,4,3]    
    '''
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
    
def makeLogger(isMaster, isViewer, isServer, rank, lvl=logging.INFO):
    '''Returns Python logger with prefix identifying master/viewer/server/worker & rnk
    If noMpi is true, returns logger with 'noMpi' prefix.
    ARGS:
     isMaster   - True if this is logger for master rank
     isViewer   - True if this is logger for viewer rank
     isServer   - True is this logger is for server rank
     rank       - rank to report
     lvl        - logging level
     noMpi      - testing, True if noMpi
    '''
    assert int(isMaster) + int(isViewer) + int(isServer) in [0,1], "more than one of isMaster, isViewer, isServer is true"
    if isMaster:
        logger = logging.getLogger('master-rnk:%d' % rank)
    elif isViewer:
        logger = logging.getLogger('viewer-rnk:%d' % rank)
    elif isServer:
        logger = logging.getLogger('server-rnk:%d' % rank)
    else:
        logger = logging.getLogger('worker-rnk:%d' % rank)
    logger.setLevel(lvl)
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
