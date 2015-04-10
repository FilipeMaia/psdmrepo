import psana
import numpy as np
from pprint import pprint
import StringIO
import ParCorAna.PsanaUtil as PsanaUtil

class EventData(object):
    '''Object returned by EventIter. 

    Attributes:
      sec:       seconds of event from which data was retrieved
      nsec:      nanoseconds of event from which data was retrieved
      fiducials: fiducials of event from which data was retrieved
      dataArray: a numpy data array from a detector in the event.
                 dataArray is guaranteed to be contiguous C ordered array
    '''
    def __init__(self,sec, nsec, fiducials, dataArray):
        self.sec=sec
        self.nsec=nsec
        self.fiducials = fiducials
        self.dataArray = np.ascontiguousarray(dataArray)

    def eventId(self):
        '''returns tuple (sec, nsec, fiducials) for this event data'''
        return self.sec, self.nsec, self.fiducials

class EventIter(object):
    '''Provides an iterator over psana events. 
    ::

      Used to run psana over a dataset and get specific detector data
      Decides which events this server will read from the dataset
      Provides user callbacks to filter events/preprocess data

      Args:
      dataSourceString (str): psana datset specification
      rank (int):     identifies this rank among the servers
      servers (int):  number of servers
      xCorrBaseObj:   instance of XCorrBase, used to do scatter and call user server callbacks
      system_params: dict with system keys. Keys used:
                     ['psanaOptions']        - psana configuration  
                     ['outputArrayType']     - output array to get from event store, for example
                                               psana.ndarray_float64_2
                     ['src']                 - src string for where array is in event
                     ['ndarrayCalibOutKey']  - key string for where array is in event
        ndarrayShape - the expected shape of the detector NDArray
        logger
        numEvents   - set to non zero to stop early

        rank and servers are used to distribute the events. 
        EventIter is run on  each server.

    Example:
      >>> dataIter = EventIter(args)
      >>> dataGen = dataIter.dataGenerator()
      >>> for datum in dataGen:
          # wortk with datum.sec, nsec and dataArray
    '''
    def __init__(self,dataSourceString, rank, servers, 
                 xCorrBaseObj, system_params, ndarrayShape, logger, numEvents=None):

        if numEvents is None:
            numEvents = 0
        self.numEvents = numEvents
        assert rank in servers, "serverRank=%d not in servers=%s" % (rank, servers)
        self.serverNumber = servers.index(rank)
        self.servers = servers
        self.rank = rank
        self.system_params = system_params
        self.ndarrayShape = ndarrayShape
        self.logger = logger
        self.xCorrBaseObj = xCorrBaseObj
        self.userObj = xCorrBaseObj.userObj

        psanaOptions = system_params['psanaOptions']
        psana.setOptions(psanaOptions)

        msgbuffer = StringIO.StringIO()
        pprint(psanaOptions, msgbuffer)
        logger.debug('initialized psana with options:\n%s' % msgbuffer.getvalue())

        datasetParts = PsanaUtil.parseDataSetString(dataSourceString)
        self.isIndex = datasetParts['idx']
        self.isShmem = not (datasetParts['shmem'] is False)  # datasetParts['shmem'] is False or a string
        self.isH5 = datasetParts['h5'] 
        self.isXtc = datasetParts['xtc'] 
        self.isLive = datasetParts['live']
        numServers = len(servers)
        if numServers > 1:
            if self.isH5:
                logger.warning("multiple servers in h5 mode, servers will handle the same events")
            elif (not self.isIndex) and (not self.isShmem):
                assert self.isXtc, "dataset string: %s does not appear to be xtc" % dataSourceString
                if 'streams' in datasetParts:
                    streams = datasetParts['streams']
                    daqStreams = [stream for stream in streams if stream < 80]
                    ctrlStreams = [stream for stream in streams if stream >= 80]
                else:
                    daqStreams = range(80)
                    ctrlStreams = range(80,160)
                thisServerDaqStreams = [daqStreams[k] for k in range(self.serverNumber, len(daqStreams), numServers)]
                thisServerStreams = thisServerDaqStreams + ctrlStreams
                thisServerStreams.sort()
                dataSourceString = PsanaUtil.changeDatasetStreams(dataSourceString, thisServerDaqStreams)
                logger.debug("server %d of %d, changed streams. new dataset=%s" % (self.serverNumber, numServers, dataSourceString))
        self.dataSourceString = dataSourceString

        self.called = False

        self.goodEvents = 0
        self.badEvents = 0
        self.printedNoDataWarning = False

        self.getType = self.system_params['outputArrayType']
        self.getSrc = psana.Source(self.system_params['src'])
        self.getKey = self.system_params['ndarrayCalibOutKey']
        if self.getKey is None:
            self.logger.warning("system_params['ndarrayCalibOutKey'] is None. using uncalibrated key: 'ndarrayProducerOutKey'")
            self.getKey = self.system_params['ndarrayProducerOutKey']
        assert self.getKey is not None, "Neither 'ndarrayCalibOutKey' nor 'ndarrayProducerOutKey' is set in system_params"

    def eventOk(self, evt):
        return self.userObj.serverEventOk(evt)

    def getDataArray(self, evt):
        dataArray = evt.get(self.getType, self.getSrc, self.getKey)
        if dataArray is None:
            self.badEvents += 1
            if self.badEvents > 20 and self.goodEvents == 0 and (not self.printedNoDataWarning):
                msg = "ERROR: more than 20 events with no data. Trying to get: type=%s src=%s key=%s. " % (self.getType, self.getSrc, self.getKey)
                msg += "current event keys are:\n  %s"  % '\n   '.join(map(str,evt.keys()))
                self.printedNoDataWarning = True
                raise Exception(msg)
            return None

        assert dataArray.shape == self.ndarrayShape, "EventIter.getDataArray: " + \
            ("shape ERROR. evt ndarray shape=%s != %s (the expected ndarray shape usual from the mask file)" % \
             (dataArray.shape, self.ndarrayShape))

        dataArray = self.userObj.serverFinalDataArray(dataArray, evt)
        return dataArray # may be None, or modified copy

    def sendToWorkers(self, datum):
        self.xCorrBaseObj.serverWorkersScatter(serverFullDataArray = datum.dataArray)

    def abortFromMaster(self):
        pass

    def getEventDataToYield(self, evt):
        if not self.eventOk(evt):
            return None
        dataArray = self.getDataArray(evt)
        if dataArray is None:
            return None
        eventId = evt.get(psana.EventId)
        assert eventId is not None
        sec, nsec = eventId.time()
        fiducials = eventId.fiducials()
        return EventData(sec, nsec, fiducials, dataArray)

    def dataGenerator(self):
        '''yields a sequence of EventData.

        Each element yielded is valid. Stops numEvents hit, or all data read.

        Example:
          >>>  for evtData in eventIter.dataGenerator():
                 print "got detector data at sec=%d nsec=%d fid=%d of shape=%r" % \
                  (evtData.sec, evtData.nsec, evtData.fiducials, evtData.dataArray.shape)
        '''
        assert not self.called, "cannot call EventIter dataGenerator twice"
        self.called = True

        ds = psana.DataSource(self.dataSourceString)
        
        if self.isIndex:
            evtStride = len(self.servers)
            evtStart = self.servers.index(self.rank)
            for run in ds.runs():
                times = run.times()
                numEvents = len(times)
                if self.numEvents > 0:
                    numEvents = min(self.numEvents, numEvents)
                for tmIdx in range(evtStart, len(times), evtStride):
                    if tmIdx >= numEvents:
                        break
                    evt = run.event(times[tmIdx])
                    eventDataToYield = self.getEventDataToYield(evt)
                    if eventDataToYield is not None:
                        yield eventDataToYield
        else:
            for idx, evt in enumerate(ds.events()):
                if self.numEvents > 0 and idx >= self.numEvents:
                    break
                eventDataToYield = self.getEventDataToYield(evt)
                if eventDataToYield is not None:
                    yield eventDataToYield
                    
