import os
import inspect
import numpy as np
import psana
import h5py

class BreakOutException:
    pass

def gatherSave(dataSource,
               h5OutputFileName,
               inputOutputFunction,
               inputs,
               extraInputs=None,
               whichEvent=False,
               h5DataSetNames=('data','time','step'),
               overwrite=False,
               numEvents=None,
               status=None):
    '''Gathers data from a Psana datasource and writes it all at once to an hdf5 file. 
    (gathered data must fit in memory).
    
    ARGS:
    dataSource:          Psana datasource string, such as 'exp=cxitut13:run=22'
    h5OutputFileName:    name of hdf5 output file
    inputOutputFunction: callback function to process data from Psana event.
                         Returns event based data to write to hdf5 file.
    inputs:              dict describing inputs to extract from Psana event and
                         pass to inputOutputFunction. See below.
    extraInputs:         dict of additional args to pass to inputOutputFunction 
                         (defaults to None)
    whichEvent:          when True, inputOutputFunction receives four additional arguments:
                           runNumber   which run this event is in, zero-up counter
                           stepNumber  which step or calib cycle this event is in (zero-up)
                           eventInStep which event within the step this is
                           eventNumber event number within the dataSource
                         defaults to False.
    h5DataSetNames:      tuple of three names for the three datasets created. Defaults to
                           data:  output returned by the inputOutputFunction
                           time:  contains seconds, nanoseconds, ticks and fiducials for events
                           step:  contains run, step - which run and calibCycle the event was in
    overwrite:            set True to overwrite the output h5 file
    numEvents:            set this to process a limited number of events in the dataSource.
                          useful for debugging.
    status:               set to an integer nth and a status is reported every event
    
    An example of use would be:

    def myGather(diode, epicsPv):
       return {'aD':diode.channel0Volts(), 'pv':epicsPv.value(0) }

    gatherSave('exp=cxitut13:run=22','myout.h5', myGather, 
               inputs={'diode':(psana.Ipimb.DataV2, psana.Source('DetInfo(CxiDg4.0:Ipimb.0)')),
                       'epicsPv':(psana.Epics, 'CXI:SC1:MZM:08:ENCPOSITIONGET')})

    the keys of the inputs dict - 'diode', 'epicsPv', must exactly match the names of 
    parameters to the function myGather.

    When myGather is called, it is passed Psana object retrieved from the epics store or event store.
    The tuple (psana.Ipimb.DataV2, psana.Source('DetInfo(CxiDg4.0:Ipimb.0)') means to pass

    evt.get(psana.Ipimb.DataV2, psana.Source('DetInfo(CxiDg4.0:Ipimb.0)'))

    in for 'diode'.  Similarly the tuple 
    (psana.Epics, 'CXI:SC1:MZM:08:ENCPOSITIONGET') 
    means to get the named epics pv from the epics store.

    The result will be the file myout.h5 with the datasets 'data', 'time', 'step'
    where data has the two fields 'aD' and 'pV'.

    To control the order of the fields in the output file, have myGather return a list of
    pairs rather than a dict, i.e:

    def myGather(diode, epicsPv):
       return (('aD':diode.channel0Volts()), ('pv':epicsPv.value(0)))

    The inputOutputFunction (myGather in example above) can return an empty list or dict to
    skip storing data for the event. When storing data, the keys and types of the data must
    always be the same. For instance, returning
    {'outputA':23}
    and then 
    {'outputA':25.5}
    will produce an error as the type changed from int to float.
    Likewise returning {'outputA':23} and then {'outputA':23, 'output2':1.1} produces an error.
    
    You can wrap output in numpy types to control output, for instance:
    
    import numpy as np
    def myGather(diode, epicsPv):
       return (('aD':np.float64(aDval)), ('pv':np.uint16(pvVal)))


    The extraInputs argument allows the myGather function to do bookeeping, keep track of past
    events, or have constants passed in. For instance:
    
    myconstants = {'MOTOR_POS':23.3}

    def myGather(epicsPv, myconstants):
      return {'aD': myconstants['MOTOR_POS']*epicsPv.value(0)}

    gatherSave('exp=cxitut13:run=22','myout.h5', myGather, 
               inputs={'epicsPv':(psana.Epics, 'CXI:SC1:MZM:08:ENCPOSITIONGET')},
               extraInputs={'myconstants':myconstants})

    again the keys of extraInputs must exactly match arguments of myGather.

    Likewise if whichEvent=True, myGather must take arguments named runNumber, etc.
    '''
    dataSource = psana.DataSource(dataSource)
    epicsStore = dataSource.env().epicsStore()

    whichEventArgs = ['runNumber','stepNumber','eventInStep','eventNumber']
    if not whichEvent:
        whichEventArgs = None
    validateInputs(inputs,extraInputs,inputOutputFunction,whichEventArgs)

    blockSize = 100000
    dataDtype = None
    timeDtype = np.dtype([('seconds',np.uint32),('nanoseconds',np.uint32), 
                          ('ticks',np.uint32), ('fiducials',np.uint32)])
    posDtype = np.dtype([('runNumber',np.uint32), ('stepNumber',np.uint32)])
    dataArray = None
    timeArray = None
    posArray = None
    dataEventBuffer = None

    def createArrays():
        timeArray = np.zeros(blockSize,dtype=timeDtype)
        posArray = np.zeros(blockSize,dtype=posDtype)
        dataArray = np.zeros(blockSize,dtype=dataDtype)
        return dataArray, timeArray, posArray

    def validateData(outputs, eventId):
        names = [k for k,v in outputs]
        assert tuple(names) == dataDtype.names, ("names of outputs returned during eventId=%s" % eventId) + \
                                                (" do not match original names.\nnew names: %s\n " % tuple(names)) + \
                                                ("old names: %s" % dataDtype.names)
        valueTypeAgreement = [type(v)==dataDtype[k] for k,v in outputs]
        msg = ''
        if not all(valueTypeAgreement):
            msg += "types of values returned in h5output for eventId=%s " % eventId
            msg += "do not match types returned in first event. mismatches:\n"
            for k,v in outputs:
                if type(v) != dataDtype[k]:
                    msg += " ** k=%s value=%s original type=%s new type=%s\n" % (k, v, dataDtype[k], type(v))
        assert all(valueTypeAgreement), msg
        
    def storeToArrays(dataArray, timeArray, posArray, idx, dvals, evtId, run, calib):
        tm = evtId.time()
        timeValues = (tm[0],tm[1],evtId.ticks(), evtId.fiducials())
        timeArray[idx]=timeValues
        posValues = (run, calib)
        posArray[idx]=posValues
        dataArray[idx]=dvals

    eventNumber = -1
    storedEvents = 0
    
    if not overwrite:
        assert not os.path.exists(h5OutputFileName), "output file exists and overwrite not set"
        
    assert len(h5DataSetNames)==3, "h5DataSetNames must contain three elments"
    assert all([isinstance(x,str) for x in h5DataSetNames]), "h5DataSetNames must be a list of three names"

    bytesPerStoredEvent = None
    try:
        for runNumber,run in enumerate(dataSource.runs()):
            for calibNumber,calib in enumerate(run.steps()):
                for eventInStep,evt in enumerate(calib.events()):
                    eventNumber += 1
                    if numEvents is not None and eventNumber >= numEvents:
                        raise BreakOutException
                    inputDict = fillInputs(inputs, epicsStore, evt)
                    if len(inputDict)==0:
                        continue
                    if extraInputs is not None:
                        for ky,val in extraInputs.iteritems():
                            inputDict[ky]=val
                    eventId = evt.get(psana.EventId)
                    if whichEvent:
                        inputDict['runNumber']=run
                        inputDict['stepNumber']=calib
                        inputDict['eventInStep']=eventInStep
                        inputDict['eventNumber']=eventNumber
                        
                    h5outputs = inputOutputFunction(**inputDict)
                    if len(h5outputs)==0:
                        continue
                    if isinstance(h5outputs,dict):
                        h5outputs = [(k,v) for k,v in h5outputs.iteritems()]
                    if dataArray is None:
                        dataDtype = getDtype(h5outputs)
                        dataArray, timeArray, posArray = createArrays()
                        dataEventBuffer = np.zeros(1,dtype=dataDtype)
                        bytesPerStoredEvent = dataEventBuffer.nbytes + timeArray[0].nbytes + posArray[0].nbytes
                        
                    if storedEvents >= timeArray.size:
                        timeArray.resize(timeArray.size + blockSize)
                        posArray.resize(posArray.size + blockSize)
                        dataArray.resize(dataArray.size + blockSize)
                    validateData(h5outputs, eventId)
                    # store 
                    for idx,keyval in enumerate(h5outputs):
                        dataEventBuffer[0][idx] = keyval[1]
                    storeToArrays(dataArray, timeArray, posArray, storedEvents, 
                                  dataEventBuffer, eventId, runNumber, calibNumber)
                    storedEvents += 1
                    if (status is not None) and (status > 0) and (eventNumber % status == 0) and \
                       (bytesPerStoredEvent is not None):
                        print "run=%4.4d calib=%4.4d eventInStep=%4.4d eventNumber=%4.4d arrays: length=%4.4d" % \
                            (runNumber, calibNumber, eventInStep,eventNumber, storedEvents),
                        totalMem = memoryHumanReadable(bytesPerStoredEvent * storedEvents)
                        print " memory: %s" % totalMem
    except BreakOutException:
        pass
    # shrink arrays to number of events we stored data from
    dataArray.resize(storedEvents)
    timeArray.resize(storedEvents)
    posArray.resize(storedEvents)

    if not overwrite:
        assert not os.path.exists(h5OutputFileName), "overwrite is False but the output file %s exists" % h5OutputFileName
    fout = h5py.File(h5OutputFileName,'w')

    dataName, timeName, posName = h5DataSetNames
    H5WriteDataset(fout,dataName,dataArray)
    H5WriteDataset(fout,timeName,timeArray)
    H5WriteDataset(fout,posName,posArray)
    fout.close()

def validateInputs(inputs, extraInputs, inputOutputFunction, whichEventArgs):

    # check that inputs is a dict,
    # if whichEventArgs is not None, check that inputs keys do not include 
    # runNumber, stepNumber, etc
    # check that inputs keys are arguments to inputOutputFunction
    # if whichEventArgs, check that inputs keys do not collide with whichEventArgs
    # check that inputs values describe epics and psana objects correctly
    
    assert isinstance(inputs,dict), "inputs must be a dict"
    assert isinstance(inputOutputFunction,type(validateInputs)),  "inputOutputFunction does not "+\
                                                      "appear to be a Python function"
    userArgs = inspect.getargspec(inputOutputFunction).args
    for ky in inputs.keys():
        assert isinstance(ky,str), "keys in inputs dict must be strings"
        assert ky in userArgs, "keys of the input dict must be names of arguments to "+\
                               "the inputOutputFunction, however "+\
                               ("'%s' is in the inputs dict, but not the name of a function argument" % ky)
        userArgs.remove(ky)
        if whichEventArgs:
            assert ky not in whichEventArgs, ("inputs key %s collides with argument "+\
                                    "that will be passed via whichEvent (args passed will be %s) - "+\
                                    "change this argument name") % (ky, whichEventArgs)
    for val in inputs.values():
        assert isinstance(val,tuple) or isinstance(val,list), ("values of inputs dict "+\
                                                               "must be lists or tuples")
        assert len(val)>=2, "values of inputs dict must have at least 2 elements"
    for inputArgName, getTuple in inputs.iteritems():
        psanaType = getTuple[0]
        if psanaType == psana.Epics:
            pvName = getTuple[1]
            assert isinstance(pvName,str), "inputs epics - second item must be string with pvname"
            assert len(getTuple)==2, "inputs item for a epics pv must have only two entries"
        else:
            source = getTuple[1]
            assert isinstance(source,psana.Source), "inputs second entry must be a psana.Source"
            assert len(getTuple)==2, "inputs item for a psana type must have only two entries"

    # check that extraInput is a dict
    # check that inputs keys do not collide with extraInput
    # check that extraInput keys are in the inputOutputFunction
    if extraInputs is not None:
        assert isinstance(extraInputs,dict), "extraInput must be a dict"
        for ky in extraInputs.keys():
            assert isinstance(ky,str), "keys in extraInput dict must be strings"
            assert ky not in inputs.keys(), "extraInput key: %s collides with keys in inputs, change names" % ky
            assert ky in userArgs, ("keys of the extraInput dict must be names of arguments " + \
                                   "to the inputOutputFunction, however '%s' is in the extraInputs dict, " + \
                                   "but not the name of a function argument") % ky
            if whichEventArgs:
                assert ky not in whichEventArgs, ("extraInputs key %s collides with argument that " +\
                                                  "will be passed via whichEvent (args passed will be %s)"+\
                                                  " - change this argument name") % (ky, whichEventArgs)
            userArgs.remove(ky)

    # check that whichEventArgs are parameters to inputOutputFunction
    if whichEventArgs:
        for ky in whichEventArgs:
            assert ky in userArgs, "inputOutputFunction will be passed whichEvent args, " + \
                                   ("but arg %s is not a parameter to the function") % ky
            userArgs.remove(ky)
            
    # check that all inputOutputFunction arguments are accounted for
    assert len(userArgs)==0, "inputOutputFunction can only take arguments described by inputs, "+\
                             ("extraInputs or whichEvent - however it takes additional arguments: %s") % userArgs

def fillInputs(inputs, epicsStore, evt):
    inputDict = {}
    goodEvent = True
    for inputArgName, getTuple in inputs.iteritems():
        psanaType = getTuple[0]
        if psanaType == psana.Epics:
            pvName = getTuple[1]
            pv = epicsStore.getPV(pvName)
            assert pv is not None, ("The pvName: %s is not in the epics store - is the name "+\
                                    "spelled correctly?") % pvName
            if pv.isCtrl():
                goodEvent = False
                print "warning: pv isCtrl for %s. Possible known or new DAQ problem." % (evt.get(psana.EventId))
                print " Contact LCLS with experiment, run number and above eventid. (email pcds-help)"
                break
            inputDict[inputArgName]=pv
        else:
            source = getTuple[1]
            psanaObj = evt.get(psanaType,source)
            if psanaObj is None:
                goodEvent = False
                break
            inputDict[inputArgName]=psanaObj

    if not goodEvent:
        return {}
    return inputDict

def getDtype(outputs):
    assert isinstance(outputs,list) or isinstance(outputs,tuple), "outputs must be list or tuple"
    for xy in outputs:
        assert len(xy)==2, "each output entry must have two items - the field name and its value"

    simpleTypes = set([float,np.float,np.float16, np.float32, np.float64, np.float128,
                      int, np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64])
    dtypeList = []
    for name,value in outputs:
        assert isinstance(name,str), "output keys must be strings"
        valueType = type(value)
        assert valueType in simpleTypes, "output values can only be a simple type - a float or int. " + \
            " type for '%s' is %s" % (name,valueType)
        if valueType is float:
            valueType = np.float
        if valueType is int:
            valueType = np.int
        dtypeList.append((name,valueType))

    return np.dtype(dtypeList)
        
def H5WriteDataset(parent, dataName, dataArray):
    dataDtype = dataArray.dtype
    dataDset = parent.create_dataset(dataName, dataArray.shape, dataDtype)
    dataDset[:] = dataArray

def H5ReadDataset(filename, dsetName):
    f = h5py.File(filename,'r')
    dset = f[dsetName]
    dsetArray = dset[:]
    f.close()
    return dsetArray

def H5ReadDataTimePos(filename, dataName='data', timeName='time', posName='step'):
    dataArray = H5ReadDataset(filename,dataName)
    timeArray = H5ReadDataset(filename,timeName)
    posArray = H5ReadDataset(filename,posName)
    return dataArray, timeArray, posArray

def memoryHumanReadable(totalMem):
    if totalMem < 1e3:
        return "%d bytes" % totalMem
    elif totalMem < 1e6:
        return "%.2f Kb" % (totalMem/1024.0)
    elif totalMem < 1e9:
        return "%.2f Mb" % (totalMem/float(1<<20))
    return "%.2f Gb" % (totalMem/float(1<<30))
