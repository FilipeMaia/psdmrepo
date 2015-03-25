'''Utitility functions for working with psana.

Functions include: 

  *  Parsing/managing the dataset string.
  *  Creating psana options (same as config file) for loading a module chain
'''

import psana

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
        beforeStream, afterStream = dataSourceString.split('stream=')
        afterParts = afterStream.split(':')
        afterParts[0] = ','.map(str,newStreams)
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
    spaceSplit = dataset.split()
    commaSplit = dataset.split(',')
    assert len(spaceSplit)==1 and len(commaSplit)==1, ("parseDataSetString only implemented for a " + \
                            "single string, not space or comma separated list of files")

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
        raise Exception("neither exp= nor shmem= appears in datasource specification")
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

