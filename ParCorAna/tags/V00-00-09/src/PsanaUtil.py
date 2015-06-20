'''Utitility functions for working with psana.

Functions include: 

  *  Parsing/managing the dataset string.
  *  Creating psana options (same as config file) for loading a module chain
'''

import sys
import math
import psana
import numpy as np

def rangesToList(rangeString):
    '''converts a string with ranges to a list
    
    Examples:
      >>> rangesToList('1-4,5,20-24')
      [1,2,3,4,5,20,21,22,23,24]
    '''
    ranges = rangeString.split(',')
    values = set()
    for cand in ranges:
        ab = cand.split('-')
        assert len(ab) in [1,2], "invalid range string: %s" % rangeString
        if len(ab)==2:
            a,b = map(int,ab)
            values = values.union(range(a,b+1))
        else:
            a = ab[0]
            values.add(int(a))
    values = list(values)
    values.sort()
    return values

def changeDatasetStreams(dataset, newStreams):
    '''changes the streams processed in a dataset. 

    Args:
      dataset (str): a psana dataset string, as specified in psana users manual
      newStreams (list): list of ints, the new streams to use in the dataset string

    Return:
      a new dataset string. Either it will have :stream=xxx added to the
      end, or if there already was a :stream= in dataset, it ts value will be 
      replaced with newStreams.
    '''
    # need to either add streams, or replace it
    if dataset.find('stream=')<0:
        dataset += ':stream=%s' % (','.join(map(str,newStreams),))
    else:
        beforeStream, afterStream = dataset.split('stream=')
        afterParts = afterStream.split(':')
        afterParts[0] = ','.join(map(str,newStreams))
        afterStream = ':'.join(afterParts)
        dataset = 'stream='.join([beforeStream,afterStream])
    return dataset

def parseDataSetString(dataset):
    '''Return a dict of the pieces of the dataset string, and fills in defaults.

    If stream or run is in the key list, it will convert the list of ranges into
    sorted list of values.

    Example:
      >>> exp=CXI/cxi342:xtc:run=3,5:stream=3-5,8
     {'instr':'CXI', 'exp':'cxi343','xtc':True,'run':[3,5],'stream':[3,4,5,8], 
     'xtc':True, 'idx':False, 'h5':False, 'shmem':False, 'live':False}

    Notes:
      Not as robust as psana routine, this does not check for an experiment
      number that it looks up in the database.

      One potential point of confusion, shmem defaults to False in the dict if not provided, 
      but will be a string if it is
    '''
    parts = dataset.split(':')
    keyValues = {}
    for part in parts:
        eqSplit = part.split('=')
        assert len(eqSplit) in [1,2], "malformed dataset string. this part: %s of dataset=%s" % (part, dataset)
        if len(eqSplit)==2:
            keyValues[eqSplit[0]] = eqSplit[1]
        else:
            keyValues[eqSplit[0]] = True

    # check for things like exp=CXI/sss to get instrument as well as shmem=CXI
    if 'exp' in keyValues:
        expValue = keyValues['exp']
        expValueSplit = expValue.split('/')
        assert len(expValueSplit) in [1,2], "malformed exp= in datset. Should only be 0 or 1 '/' characters: exp=%s" % expValue
        if len(expValueSplit)==2:
            keyValues['exp'] = expValueSplit[1]
            keyValues['instr']=expValueSplit[0].upper()
        elif len(expValueSplit)==1:
            keyValues['exp'] = expValueSplit[0]
            assert len(expValueSplit[0])>3, "exp key=%s does not have more than 3 characters" % expValueSplit[0]
            keyValues['instr']=expValueSplit[0][0:3].upper()
    elif 'shmem' in keyValues:
        assert len(keyValues['shmem'])>=3, "shmem in datasource string, but value=%s has < 3 characters?" % keyValues['shmem']
        keyValues['instr']=keyValues['shmem'][0:3].upper()
    else:
        raise Exception("neither exp= nor shmem= appears in datasource specification=%s"%dataset)
    knownInstruments = 'AMO  CXI  DIA  MEC  MOB  SXR  USR  XCS  XPP'.split()
    assert keyValues['instr'] in knownInstruments, "Could not find KNOWN instrument in datasource string: %s. Found %s. Looking for one of %s" % \
        (dataset, keyValues['instr'],knownInstruments)

    # convert range lists for run or stream
    if 'run' in keyValues:
        keyValues['run'] = rangesToList(keyValues['run'])

    if 'stream' in keyValues:
        keyValues['stream'] = rangesToList(keyValues['stream'])
    
    if 'shmem' not in keyValues:
        keyValues['shmem'] = False

    if 'h5' not in keyValues:
        keyValues['h5'] = False

    if 'live' not in keyValues:
        keyValues['live'] = False

    if 'idx' not in keyValues:
        keyValues['idx'] = False
        
    if 'xtc' not in keyValues:
        if not keyValues['h5']:
            keyValues['xtc'] = True

    return keyValues

def readDetectorDataAndEventTimes(system_params, eventOkFunction=None):
    '''gathers all masked data from a calibrated detector, as well as the even times.
    
    Reads through all, or specified number of events, of a psana datasource. Uses a 
    mask to mask out part of the calibrated detector ndarray. 

    WARNING: If the number of events is large,
    and the mask includes many of the detector pixels, this will exhaust memory.

    Args:
      system_params (dict): the parametes for the ParCorAnaDriver. Uses parameters
                            that specify the dataset, detector data, masking, and number of events.
      eventOkFunction (function, optional): a function that takes a Psana Event as argument and
    
    Return:
      (tuple): with

         * data (ndarray): a 2D numpy array of float64 - each row is the masked data from a event
         * times (list): each element is the EventId corresponding to the data row
    '''
    class DataResizeValues(object):
        '''helper class to resize a growing data array
        '''
        def __init__(self, numevents, numelem, dataBlockLenInPixels = 10<<20):
            self.numevents = numevents
            self.blockLen = max(1,int(math.ceil(dataBlockLenInPixels / numelem)))
        def initialDataLen(self):
            if self.numevents > 0: return self.numevents
            return self.blockLen
        def growDataLen(self, data):
            return data.shape[0]+self.blockLen
        
    dset = system_params['dataset']
    numevents = system_params['numEvents']
    sanaOptions = system_params['psanaOptions']
    calibOutKey = system_params['ndarrayCalibOutKey']
    outputArrayType = system_params['outputArrayType']
    srcString = system_params['src']
    maskFile = system_params['maskNdarrayCoords']
    assert os.path.exists(maskFile), "maskfile=%s doesn't exist" % maskFile
    mask = np.load(maskFile)
    self.maskShape = mask.shape
    self.maskFlatIndexArray = 1 == mask.flatten()

    psana.setOptions(psanaOptions)
    ds = psana.DataSource(dset)
    data = None
    dataResize = None
    src = psana.Source(srcString)
    eventTimesWithDetectorData = []
    evtWithDetectorDataIdx = -1

    def nextIdxAndEvent(ds, dset, numevents):
        if parseDataSetString(dset)['idx']:
            evtIdx = -1
            for run in ds.runs():
                times = run.times()
                for tm in times:        
                    evtIdx += 1
                    if numevents > 0 and evtIdx >= numevents: 
                        return
                    evt = run.get(tm)
                    yield evtIdx, evt
        else:
            for evtIdx, evt in enumerate(ds.events()):
                if numevents > 0 and evtIdx >= numevents: 
                    return
                yield evtIdx, evt

    for evtIdx, evt in nextIdxAndEvent(ds, dset, numevents):
        eventId = evt.get(psana.EventId)
        assert eventId is not None, "no eventId in event %d" % evtIdx
        arrayData = evt.get(outputArrayType, src, calibOutKey)
        if arrayData is None:
            sys.stderr.write("WARNING: detector data not present for event %d\n" % evtIdx)
            continue
        evtWithDetectorDataIdx += 1
        assert arrayData.shape == maskShape, "mask file and array do not have same shape"
        arrayData = arrayData.flatten()
        maskedData = arrayData[maskFlatIndexArray]
        if data is None:
            dataResize = DataResizeValues(numevents, len(maskedData))
            data = np.zeros((dataResize.initialDataLen(), len(maskedData)), np.float64)
        if evtWithDetectorDataIdx >= data.shape[0]:
            data.resize((dataResize.growDataLen(), len(maskedData)), refcheck=False)
        data[evtWithDetectorDataIdx,:]=maskedData[:]
        eventTimesWithDetectorData.append(eventId.time())

    if data.shape[0] > evtWithDetectorDataIdx + 1:
        data.resize((evtWithDetectorDataIdx+1, data.shape[1]), refcheck=False)

    assert data.shape[0] == len(eventTimesWithDetectorData)
    return data, eventTimesWithDetectorData

def getSortedCountersBasedOnSecNsecAtHertz(eventTimes, hertz=120):
    '''Returns counters at given clock rate, and indicies to reorder data.

    Args:
      eventTimes (list): list of tuples - first two elements of each tuple must be seconds/nanoseconds
                         each as an int. There may be other things in the tuple
      hertz:   rate in seconds for counter

    Example:
      >>> counters, newOrder = getSortedCountersAtHertz([(3,2),(3,5),(2,0)], 120)
      newOrder = [2, 0, 1]
      counters = [0,120, 120]

    For the counters in the example, a 120hz counter has .08 sec = 80 million nanoseconds between occureneces.
    so seconds=3,nano=2 and sec=3,nano=5 both look like the same counter. Meanwhile there are 120 counter values 
    between sec=2 and sec=3.

    Returns:
      (tuple): sorted counter values and a index mapping to re-order data

        * sortedCounters: an array of int64, the 120hz counter values for all the event times
                          usually this is an array of consecutive integers, starting at 0, but
                          if may have gaps depending on the data
        * newDataOrder:   an index array to reorder event data to get the counters.
    '''
    assert len(eventTimes)>0, "no event times for getCounters"
    sec0= eventTimes[0][0]
    floatTimes = np.array([(tm[0]-sec0) + 1e-9 * tm[1] for tm in eventTimes])
    newDataOrder = np.argsort(floatTimes)
    sortedTimes = floatTimes[newDataOrder]
    sortedCounters = np.zeros(len(eventTimes), np.int64)
    oneCounterSec = 1.0/float(hertz)
    idx = 0
    numWhereFractionCloseToHalf = 0
    for prevSec, nextSec in zip(sortedTimes[0:-1],sortedTimes[1:]):
        idx += 1
        currDiff = nextSec-prevSec
        incrCounter = round(currDiff/oneCounterSec)
        fraction = currDiff/oneCounterSec - incrCounter
        if fraction >= .45 and fraction <= .55:
            numWhereFractionCloseToHalf += 1
        sortedCounters[idx]=sortedCounters[idx-1]+incrCounter
    if numWhereFractionCloseToHalf>0:
        sys.stderr.write(("In general sec/nsec from event to event should be very close to a multiple of " + \
                         " %.3f seconds apart. However %d of the events were not close. " % + \
                         " They were within 10% of boundary value where it is ambiguous\n") % \
                         (oneCounterSec, numWhereFractionCloseToHalf))
    return sortedCounters, newDataOrder

def makePsanaOptions(srcString, psanaType, ndarrayOutKey, ndarrayCalibOutKey, imageOutKey=None):
    '''returns Psana options to calibrate given detector data, and returns output array type.

    Calibrating detector images is done by loading the appropriate Psana modules. At least two
    modules are required. The first converts the LCLS DAQ type to a generic ndarray. The second
    calibrates the ndarray. Optionally one can load a third module to turn an ndarray to an image.
    
    Optional one can not calibrate and just create options to return an ndarray.

    Args:
      srcString (str):  detector src 
      psanaType (type):  detector type (such as psana.CsPad.DataV2)
      ndarrayOutKey (str):   key that ndarray producer should use for output
      ndarrayCalibOutKey (str):  key that ndarrayCalib should use as output, or None to
                                 specify no calibration
      imageOutKey (str, Optional): None means no image (default) otherwise image key

    ::

      Returns:
      psanaOptions (dict) for passing to psana.setOptions
      type (psana type for ndarray) the type of the ndarray to retrieve from event store

    '''    
    assert ndarrayCalibOutKey is None or ndarrayCalibOutKey != ndarrayOutKey, "calib output key must be different than ndarray output key"
    assert imageOutKey is None or (imageOutKey != ndarrayCalibOutKey and imageOutKey != ndarrayOutKey), "image ouput key must differ from other keys"

    type2params = {psana.CsPad2x2.ElementV1:{'NDArrProducer':'CSPadPixCoords.CSPad2x2NDArrProducer', 
                                             'outArrDims':3},     # [185, 388, 2]
                   psana.CsPad.DataV1:      {'NDArrProducer':'CSPadPixCoords.CSPadNDArrProducer',  
                                             'outArrDims':3},     # [32, 185, 388]
                   psana.CsPad.DataV2:      {'NDArrProducer':'CSPadPixCoords.CSPadNDArrProducer',
                                             'outArrDims':3},     # [32, 185, 388]
                   psana.Epix.ElementV1:    {'NDArrProducer':'ImgAlgos.EpixNDArrProducer',
                                             'outArrDims':2},     # varies
                   psana.Epix.ElementV2:    {'NDArrProducer':'ImgAlgos.EpixNDArrProducer',
                                             'outArrDims':2},     # varies
                   psana.PNCCD.FramesV1:    {'NDArrProducer':'ImgAlgos.PnccdNDArrProducer',
                                             'outArrDims':3},    # [4, 512, 512]
    }

    assert psanaType in type2params, "makePsanaOptions does not know what modules to use for type: %s" % psanaType
    ndarrProducer = type2params[psanaType]['NDArrProducer']

    dim2array = dict([(dim,eval('psana.ndarray_float64_%d'%dim)) for dim in range(1,7)])
    outArrDims = type2params[psanaType]['outArrDims']
    assert outArrDims in dim2array, "internal error: no array type defined for dim=%s" % outArrDims
    outArrayType = dim2array[outArrDims]
    
    psanaOptions = {}

    #############
    # configure ndarray producer
    psanaOptions['modules'] = ndarrProducer
    psanaOptions[ndarrProducer+'.source'] = srcString

    # specify output key. The parameter names varies between producers, so set all to cover our bases
    psanaOptions[ndarrProducer+'.outkey'] = ndarrayOutKey
    psanaOptions[ndarrProducer+'.key_out'] = ndarrayOutKey

    # specify output type for ndarray producer
    psanaOptions[ndarrProducer+'.outtype']='double'

    # specify fullsize for ndarray producers that go as data (for consistency, user can mask out pixels)
    psanaOptions[ndarrProducer+'.is_fullsize']= 'True'

    imageInputKey = ndarrayOutKey
    if ndarrayCalibOutKey is not None:
        imageInputKey = ndarrayCalibOutKey
        assert isinstance(ndarrayCalibOutKey, str) and len(ndarrayCalibOutKey)>0, "ndarrayCalibOutKey is wrong type or zero len (must be str) it is %r" % ndarrayCalibOutKey
        doCalib = True
        psanaOptions['modules'] += ' ImgAlgos.NDArrCalib'

        # specify input source, input key, and output key for calib
        psanaOptions['ImgAlgos.NDArrCalib.source'] = srcString
        psanaOptions['ImgAlgos.NDArrCalib.key_in'] = ndarrayOutKey
        psanaOptions['ImgAlgos.NDArrCalib.key_out'] = ndarrayCalibOutKey

        # specify calibrations to do
        psanaOptions['ImgAlgos.NDArrCalib.do_peds'] = True    #  pedestals subtracted if available in calib store
        psanaOptions['ImgAlgos.NDArrCalib.do_cmod'] = True    #  common mode correction is evaluated and applied
        psanaOptions['ImgAlgos.NDArrCalib.do_stat'] = True    #  bad/hot pixels in pixel_status are masked
        psanaOptions['ImgAlgos.NDArrCalib.do_mask'] = False   #  mask is applied if the file fname_mask is available (1/0 = good/bad pixels)
        psanaOptions['ImgAlgos.NDArrCalib.do_bkgd'] = False   #  normalized background is subtracted if the file fname_bkgd is available
        psanaOptions['ImgAlgos.NDArrCalib.do_gain'] = False   #  pixel_gain correction is applied if available in calib store
        psanaOptions['ImgAlgos.NDArrCalib.do_nrms'] = False   #  per-pixel threshold is applied if pixel_rms  is available in calib store
        psanaOptions['ImgAlgos.NDArrCalib.do_thre'] = False   #  low level threshold in ADU is applied
        psanaOptions['ImgAlgos.NDArrCalib.fname_bkgd'] = ''   #  input file name for background, applied if the file name is specified
        psanaOptions['ImgAlgos.NDArrCalib.fname_mask'] = ''   #  input file name for mask, applied if the file name is specified
        psanaOptions['ImgAlgos.NDArrCalib.masked_value'] = 0  #  intensity value (in ADU) substituted for masked pixels
        psanaOptions['ImgAlgos.NDArrCalib.threshold_nrms'] = 3  #  threshold as a number of sigmas to pixel_rms parameters
        psanaOptions['ImgAlgos.NDArrCalib.threshold'] = 0       #  common low level threshold in ADU below_thre_value
        psanaOptions['ImgAlgos.NDArrCalib.below_thre_value'] = 0 # intensity substituted for pixels below threshold
        # other options are bkgd_ind_min, bkgd_ind_max, bkgd_ind_inc and print_bits

    if imageOutKey is not None:
        psanaOptions['modules'] += ' ImgAlgos.NDArrImageProducer'
        # specify input source, input key, and output key for calib
        psanaOptions['ImgAlgos.NDArrImageProducer.source'] = srcString
        psanaOptions['ImgAlgos.NDArrImageProducer.key_in'] = imageInputKey
        psanaOptions['ImgAlgos.NDArrImageProducer.key_out'] = imageOutKey
        outArrayType = dim2array[2]

    return psanaOptions, outArrayType


psanaNdArrays=[psana.ndarray_float32_1,  
               psana.ndarray_float64_3 , 
               psana.ndarray_float64_2 , 
               psana.ndarray_int16_3   , 
               psana.ndarray_int32_4   , 
               psana.ndarray_int64_5   , 
               psana.ndarray_int8_6    , 
               psana.ndarray_uint32_1  , 
               psana.ndarray_uint64_2  , 
               psana.ndarray_uint8_3 ,
               psana.ndarray_float32_2 ,  
               psana.ndarray_int16_4   , 
               psana.ndarray_int32_5   , 
               psana.ndarray_int64_6   , 
               psana.ndarray_uint16_1  , 
               psana.ndarray_uint32_2  , 
               psana.ndarray_uint64_3  , 
               psana.ndarray_uint8_4 ,
               psana.ndarray_float32_3 , 
               psana.ndarray_float64_4 , 
               psana.ndarray_int16_5   , 
               psana.ndarray_int32_6   , 
               psana.ndarray_int8_1    , 
               psana.ndarray_uint16_2  , 
               psana.ndarray_uint32_3  , 
               psana.ndarray_uint64_4  , 
               psana.ndarray_uint8_5 ,
               psana.ndarray_float32_4 , 
               psana.ndarray_float64_5 , 
               psana.ndarray_int16_6   , 
               psana.ndarray_int64_1   , 
               psana.ndarray_int8_2    , 
               psana.ndarray_uint16_3  , 
               psana.ndarray_uint32_4  , 
               psana.ndarray_uint64_5  , 
               psana.ndarray_uint8_6 ,
               psana.ndarray_float32_5 , 
               psana.ndarray_float64_6 , 
               psana.ndarray_int32_1   , 
               psana.ndarray_int64_2   , 
               psana.ndarray_int8_3    , 
               psana.ndarray_uint16_4  , 
               psana.ndarray_uint32_5  , 
               psana.ndarray_uint64_6  , 
               psana.ndarray_float32_6 , 
               psana.ndarray_int16_1   , 
               psana.ndarray_int32_2   , 
               psana.ndarray_int64_3   , 
               psana.ndarray_int8_4    , 
               psana.ndarray_uint16_5  , 
               psana.ndarray_uint32_6  , 
               psana.ndarray_uint8_1   , 
               psana.ndarray_float64_1 , 
               psana.ndarray_int16_2   , 
               psana.ndarray_int32_3   , 
               psana.ndarray_int64_4   , 
               psana.ndarray_int8_5     ,
               psana.ndarray_uint16_6   ,
               psana.ndarray_uint64_1   ,
               psana.ndarray_uint8_2    ]

