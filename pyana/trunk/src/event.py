#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

"""
This module is a collection of classes and methods to deal with the event data and everything related to it.
"""


import logging
import numpy
import os
import tempfile
from string import Template

import pyana
from pyana.histo import HistoMgr
from pypdsdata import xtc
from pypdsdata import epics
from pypdsdata import Error
from pypdsdata import acqiris
from pypdsdata import cspad
from pypdsdata import gsc16ai
from pypdsdata import pnccd
from pypdsdata import princeton

_log = logging.getLogger("pyana.event")



def _addr( address ):
    """Parse address string and returns dictionary with keys 
    (detector, detId, device, devId), some or all values can be None.
    If address type is DetInfo then returns dictionary with single key
    address."""
    
    if isinstance(address, (xtc.DetInfo, xtc.BldInfo)):
        return dict(address=address)

    # if it is a string matching BLD type then return Bld source
    try:
        return dict(address=xtc.BldInfo(address))
    except:
        pass
    
    det = None
    detId = None
    dev = None
    devId = None
    
    detdev = address.split('|')
    
    if len(detdev) > 2 :
        raise Error("Unrecognized address format: "+address)

    w = detdev[0].split('-')
    if len(w) > 2 :
        raise Error("Unrecognized detector format in address: "+address)
    if w[0] and w[0] != '*' :
        det = getattr(xtc.DetInfo.Detector, w[0], None)
        if det is None : raise Error("Unknown detector name in address "+address)
    if len(w) == 2 and w[1] and w[1] != '*' :
        try :
            detId = int(w[1])
        except :
            raise Error("Unrecognized detector ID in address "+address)
        
    if len(detdev) == 2 :
        w = detdev[1].split('-')
        if len(w) > 2 :
            raise Error("Unrecognized device format in address: "+address)
        if w[0] and w[0] != '*' :
            dev = getattr(xtc.DetInfo.Device, w[0], None)
            if dev is None : raise Error("Unknown device name in address "+address)
        if len(w) == 2 and w[1] and w[1] != '*' :
            try :
                devId = int(w[1])
            except :
                raise Error("Unrecognized device ID in address "+address)
    
    return dict(detector=det, detId=detId, device=dev, devId=devId)

class Event(object):
    
    """Class encapsulating event data. Instance of this class contains all 
    data belonging to currently processed event.

    Clients typically use generic get() method or one of the specific
    getXYZ() methods to get access to event data. TYpically event object
    contains all the data that are read from XTC file (some data from XTC 
    is stored in environment). Additionally users can add their own objects
    to event using put() method which allows sharing of the user data  
    between several modules.
    
    Event has few additional methods which provide generic information
    about event, such as event time, run number, damage mask, etc. 
    """
    
    # Set of constants specifying event processing status
    Normal = 0
    Skip = 1
    Stop = 2
    Terminate = 3
    

    def __init__( self, dg, run = None, expNum = None, env = None ):
        
        self.m_dg = dg
        self.m_run = run
        self.m_expNum = expNum
        self.m_env = env
        self.m_userData = {}
        self.m_status = Event.Normal
        

    def run(self) :
        """ self.run() -> int
        
        Returns run number as integer value. Run number is extracted from the file names being processed. 
        When the file name cannot be determined or has unknown format the above method will return None.
        """
        return self.m_run

    def expNum(self) :
        """ self.expNum() -> int
        
        Returns experiment number as integer value. EXperiment number is extracted from the file names being processed. 
        When the file name cannot be determined or has unknown format the above method will return None.
        """
        return self.m_expNum

    def seq(self) :
        """ self.seq() -> Sequence
        
        Returns _pdsdata.xtc.Sequence object corresponding to current event.
        """
        return self.m_dg.seq

    def env(self) :
        """ self.env() -> Env
        
        Returns environment object. User code does not typically need to use this method as user code 
        receives environment object via method arguments.
        """
        return self.m_dg.env

    def damage(self) :
        """ self.damage() -> int
        
        Returns object of type _pdsdata.xtc.Damage representing damage mask. The damage mask is returned for 
        the top-level XTC, individual sub-objects may contain different damage masks.
        """
        return self.m_dg.xtc.damage

    def getTime(self) :
        """ self.getTime() -> ClockTime
        
        Returns _pdsdata.xtc.ClockTime object, equivalent to dg.seq.clock().
        """
        return self.m_dg.seq.clock()


    def keys(self):
        """self.keys() -> list of tuples
        
        Returns the list of tuples (TypeId, Src), one tuple for every object contained in the 
        event. For user objects first item in tuple would be the key, second item is None.
        """
        result = []
        if self.m_dg:
            for x in self._xtcGenerator(self.m_dg.xtc):
                # filter out Epics too
                if x.contains.id() not in [xtc.TypeId.Type.Id_Xtc, xtc.TypeId.Type.Id_Epics]:
                    result.append((x.contains, x.src))

        for k in self.m_userData.iterkeys():
            result.append((k, None))

        return result


    def find(self, **kw):
        """self.find(**kw) -> list of object
        
        Returns possibly empty list of data objects contained in the event. This method accepts a number of arguments, 
        but all arguments are optional. If no arguments are given then a list of all data objects is returned. 
        If some arguments are given then only those objects that satisfy a particular criteria are returned. 
        
        The list of possible keyword arguments: 
        ``typeId`` - accepts enum xtc.TypeId.Type, only return objects which have that TypeId;
        ``version`` - accepts number, only return objects whose type version number is equal to number;
        ``level`` - accepts one of xtc.Level values, only returns objects originated at that level;
        ``detector`` - accepts enum xtc.DetInfo.Detector values, only returns objects produced by this detector;
        ``detId`` - accepts number, only returns objects produced by this detector ID;
        ``device`` - accepts enum xtc.DetInfo.Device values, only returns objects produced by this device;
        ``devId`` - accepts number, only returns objects produced by this device ID;
        ``address`` - xtc.DetInfo object or an address string.
    
        The parameters ``address`` and any of the ``detector``, ``detId``, ``device``, ``devId`` are incompatible, 
        specify only one or another.
        """

        objects = [x.payload() for x in Event._filter(self.m_dg.xtc,**kw)]
        return filter(None, objects)

    def findXtc(self, **kw):
        """self.findXtc(**kw) -> list of XTCs
        
        Returns the list (possibly empty) of Xtc objects matching the criteria.
        Accepts keyword parameters same as find() """

        return list( Event._filter(self.m_dg.xtc, **kw) )

    def findFirstXtc(self, **kw):
        """self.findFirstXtc(**kw) -> XTC object
        
        Returns the first Xtc object matching the criteria or None.
        Accepts keyword parameters same as find()."""
        
        for x in Event._filter(self.m_dg.xtc, **kw):
            return x
        return None

    def findFirst(self, **kw):
        """self.findFirst(**kw) -> object
        
        Returns first data object (XTC's payload) matching the criteria or None.
        Accepts keyword parameters same as find()."""
        
        for x in Event._filter(self.m_dg.xtc, **kw):
            obj = x.payload()
            if obj: return obj
        return None

    def get(self, key, address=None):
        """self.get(key, address=None) -> object
        
        Generic get method, retrieves detector data or user data. 
        If first argument (``key``) is an integer then it is assumed to be 
        TypeId value (such as xtc.TypeId.Type.Id_AcqWaveform), second
        argument in this case is an address string or DetInfo object.
        Otherwise the first argument is assumed to be a key (usually a 
        string) for user data added to event with put() method. """
        
        if isinstance(key,int):
            # integer key is TypeId value, must be event data
            for x in Event._filter(self.m_dg.xtc, typeId=key, address=address):
                obj = self._payload(x)
                if obj: return obj
        else :
            default = address
            return self.m_userData.get(key, default)


    def _payload(self, xtcObj):
        """ Extract payload from XTC and if necessary wrap an object into some wrapper class. """

        # get payload
        obj = xtcObj.payload()
        if not obj: return None

        # find a wrapper method for typeId
        typeId = xtcObj.contains.id()
        wrapper = None
        if typeId == xtc.TypeId.Type.Id_Gsc16aiData:
            wrapper = self._wrapGsc16aiData
        elif typeId == xtc.TypeId.Type.Id_AcqWaveform:
            wrapper = self._wrapAcqirisData
        elif typeId == xtc.TypeId.Type.Id_pnCCDframe:
            wrapper = self._wrapPnCcdData
        elif typeId == xtc.TypeId.Type.Id_PrincetonFrame:
            wrapper = self._wrapPrincetonData
        elif typeId == xtc.TypeId.Type.Id_CspadElement:
            wrapper = self._wrapCsPadQuads

        if wrapper:
            return wrapper(obj, typeId, xtcObj.contains.version(), xtcObj.src)
        else:
            # no need to wrap
            return obj


    def _wrapGsc16aiData(self, obj, typeId, typeVersion, src):
        """ Wrapper method for Gsc16ai data """
        
        cfg = self.m_env.getConfig(xtc.TypeId.Type.Id_Gsc16aiConfig, address=src)
        if not cfg : raise Error("cannot find Gsc16ai config for address %s" % src )    
        return gsc16ai.DataV1( obj, cfg )

    def _wrapAcqirisData(self, obj, typeId, typeVersion, src):
        """ Wrapper method for Acqiris data """
        
        cfg = self.m_env.getAcqConfig(address=src)
        if not cfg : raise Error("cannot find Acqiris config for address %s" % src )

        data = []        
        hcfg = cfg.horiz()
        channel = 0
        while True:
            
            vcfg = cfg.vert(channel)
            data.append(acqiris.DataDescV1(obj, hcfg, vcfg))
            
            channel += 1
            if channel >= cfg.nbrChannels(): break
            
            obj = obj.nextChannel(hcfg)

        return data

    def _wrapPnCcdData(self, obj, typeId, typeVersion, src):
        """ Wrapper method for PNCCD data"""
        
        cfg = self.m_env.getPnCCDConfig(address=src)
        if not cfg : raise Error("cannot find PnCCD config for address %s" % src )

        # get all frames
        frames = []
        numLinks = cfg.numLinks()
        while True:
            frames.append(obj)
            if len(frames) >= numLinks : break
            obj = obj.next(cfg)
        
        return pnccd.FrameV1( frames, cfg )

    def _wrapPrincetonData(self, obj, typeId, typeVersion, src):
        """ Wrapper method for Princeton data"""

        cfg = self.m_env.getPrincetonConfig(address=src)
        if not cfg : raise Error("cannot find Princeton config for address %s" % src )

        return princeton.FrameV1( obj, cfg )

    def _wrapCsPadQuads(self, obj, typeId, typeVersion, src):
        """ Wrapper method for cspad data"""
        
        cfg = self.m_env.getConfig(typeId=xtc.TypeId.Type.Id_CspadConfig, address=src)
        if not cfg : raise Error("cannot find CsPad config for address %s" % src )

        # get all elements
        quads = []
        numQuads = cfg.numQuads()
        while True:
            quads.append(cspad.wrapElement(obj, cfg))
            if len(quads) >= numQuads : break
            obj = obj.next(cfg)
        
        # quads may be unordered in XTC, clients prefer them ordered
        quads.sort(cmp = lambda x, y: cmp(x.quad(), y.quad()))
        return quads

    def getAcqValue(self, address, channel, env):
        """self.getAcqValue(address, channel, env) -> object
        
        Returns Acqiris data object of type pypdsdata.acqiris.DataDescV* for specific device and channel. 
        If address given is not very specific then the first matching object is returned.
        
        Parameters: 
        ``address`` - xtc.DetInfo object or an address string;
        ``channel`` - channel number from 0 to max number of channels;
        ``env`` - environment object containing Acqiris configuration object.

        Channel number is an integer number, total number of channels can be extracted from the Acqiris configuration object.
        """

        _log.debug("Event.getAcqValue: address=%s channel=%s", address, channel)

        xtcObj = self.findFirstXtc( typeId=xtc.TypeId.Type.Id_AcqWaveform, address=address )
        if not xtcObj : return None
        obj = xtcObj.payload()

        # get config object
        cfg = env.getAcqConfig(address=xtcObj.src)
        if not cfg : 
            raise Error("cannot find Acqiris config for address %s" % xtcObj.src )

        # check channel number
        if channel < 0 or channel >= cfg.nbrChannels() :
            raise Error("invalid channel number: %d" % channel )

        hcfg = cfg.horiz()
        vcfg = cfg.vert(channel)
        
        # move to specific channel
        for i in range(channel) :
            obj = obj.nextChannel(hcfg)
        
        return acqiris.DataDescV1( obj, hcfg, vcfg )
    
    def getEBeam(self):
        """self.getEBeam() -> object
        
        Returns data object of type pypdsdata.bld.BldDataEBeamV* whichever is present in the event.
        This method is equivalent to ``self.get(xtc.TypeId.Type.Id_EBeam)``.
        """
        return self.findFirst(typeId=xtc.TypeId.Type.Id_EBeam )

    def getEvrData(self, address):
        """self.getEvrData(address) -> object
        
        Returns data object of type pypdsdata.evr.DataV* for given address.
        Parameters: ``address`` xtc.DetInfo object or an address string.

        This method is equivalent to ``evt.get(xtc.TypeId.Type.Id_EvrData, address)``.
        """
        return self.findFirst( typeId=xtc.TypeId.Type.Id_EvrData, address=address )

    def getFeeGasDet(self):
        """self.getFeeGasDet() -> list of floats
        
        Returns the list of 4 floating numbers [f_11_ENRC, f_12_ENRC, f_21_ENRC, f_22_ENRC] 
        obtained from bld.BldDataFEEGasDetEnergy object.
        """
        obj = self.findFirst( typeId=xtc.TypeId.Type.Id_FEEGasDetEnergy )
        if obj :
            return [obj.f_11_ENRC, obj.f_12_ENRC, obj.f_21_ENRC, obj.f_22_ENRC]

    def getFrameValue(self, address):
        """self.getFrameValue(address) -> object
        
        Returns frame data object of type pypdsdata.camera.FrameV* for specific device. If address given is 
        not very specific then the first matching object is returned.

        Parameters:
        ``address`` - xtc.DetInfo object or an address string.

        This method is equivalent to ``evt.get(xtc.TypeId.Type.Id_Frame, address)``.
        """
        return self.findFirst( typeId=xtc.TypeId.Type.Id_Frame, address=address )

    def getIpimbValue(self, address):
        """self.getIpimbValue(address) -> object
        
        Returns frame data object of type pypdsdata.ipimb.DataV* for specific device. If address given 
        is not very specific then the first matching object is returned.

        Parameters:
        ``address`` - xtc.DetInfo object or an address string.

        This method is equivalent to ``evt.get(xtc.TypeId.Type.Id_IpimbData, address)``.
        """
        return self.findFirst( typeId=xtc.TypeId.Type.Id_IpimbData, address=address )

    getOpal1kValue = getFrameValue
    """This methods is an alias for getFrameValue()."""

    def getPhaseCavity(self):
        """self.getPhaseCavity() -> object
        
        Returns data object of type pypdsdata.bld.BldDataPhaseCavity.
        
        This method is equivalent to ``evt.get(xtc.TypeId.Type.Id_PhaseCavity, address)``.
        """
        return self.findFirst( typeId=xtc.TypeId.Type.Id_PhaseCavity )

    def getPnCcdValue(self, address, env):
        """self.getPnCcdValue(address, env) -> object
        
        Returns pnCCD data object of type pypdsdata.pnccd.FrameV* for specific device. 
        If address given is not very specific then the first matching object is returned.

        Parameters:
        ``address`` - xtc.DetInfo object or an address string;
        ``env`` - environment object containing pnCCD configuration object.
        
        This method is equivalent to ``evt.get(xtc.TypeId.Type.Id_pnCCDframe, address)``.
        """
        
        _log.debug("Event.getPnCcdValue: address=%s", address)

        xtcObj = self.findFirstXtc( typeId=xtc.TypeId.Type.Id_pnCCDframe, address=address )
        if not xtcObj : return None

        return self._wrapPnCcdData(xtcObj.payload(), xtcObj.contains.id(), xtcObj.contains.version(), xtcObj.src)

    def getPrincetonValue(self, address, env):
        """self.getPrincetonValue(address, env) -> object
        
        Returns Princeton frame object of type pypdsdata.princeton.Frame* for specific device. 
        If address given is not very specific then the first matching object is returned.

        Parameters:
        ``address`` - xtc.DetInfo object or an address string;
        ``env`` - environment object containing Acqiris configuration object.
                
        This method is equivalent to ``evt.get(xtc.TypeId.Type.Id_PrincetonFrame, address)``.
        """

        _log.debug("Event.getPrincetonValue: address=%s", address)

        xtcObj = self.findFirstXtc( typeId=xtc.TypeId.Type.Id_PrincetonFrame, address=address )
        if not xtcObj : return None

        return self._wrapPrincetonData(xtcObj.payload(), xtcObj.contains.id(), xtcObj.contains.version(), xtcObj.src)

    def getCsPadQuads(self, address, env):
        """self.getCsPadQuads(address, env) -> list
        
        Returns list of objects of type pypdsdata.cspad.ElementV* for specific device address. 
        If address given is not very specific then the first matching object is returned.

        Parameters:
        ``address`` - xtc.DetInfo object or an address string;
        ``env`` - environment object containing CsPad configuration object.

        The size of the list is determined by the CsPad configuration (``numQuads()`` method of the configuration object).
        """
        
        _log.debug("Event.getCsPadElements: address=%s", address)

        xtcObj = self.findFirstXtc( typeId=xtc.TypeId.Type.Id_CspadElement, address=address )
        if not xtcObj : return None

        return self._wrapCsPadQuads(xtcObj.payload(), xtcObj.contains.id(), xtcObj.contains.version(), xtcObj.src)


    def put(self, data, key):
        """self.put(data : object, key : object)
        
        Add user data to the event, key identifies the data. User data can be any kind 
        of Python object. Key is usually a string but can have any type except integer type.
        Use the same key as the first argument of ``get()`` method to retrieve stored object.
        This mechanism allows sharing of data between different modules.
        """
        
        self.m_userData[key] = data

    def status(self):
        """
        Returns processing status for this event, value returned would be one of 
        Event.Normal, Event.Skip, Event.Stop, or Event.Terminate.
        
        Regular modules should not need to check status, only special kind of modules 
        like output modules may be interested in the status value.
        """
        return self.m_status

    def setStatus(self, status):
        """
        Sets processing status for this event, status value should be one of 
        Event.Normal, Event.Skip, Event.Stop, or Event.Terminate.
        """
        self.m_status = status

    #
    # Private methods not to be used by clients directly
    #

    @staticmethod
    def _filter(xtcObj, typeId=None, version=None, level=None, address=None, 
                detector=None, detId=None, device=None, devId=None):
        """ generator which produces Xtc objects matching specified criteria.
        
            typeId   - one of xtc.TypeId.Type.Something values
            version  - TypeId version number, if missing return any version
            level    - one of xtc.Level.Something values
            detector - one of xtc.DetInfo.Detector.Something values
            detId    - detector ID number
            device   - one of xtc.DetInfo.Device.Something values
            devId    - device ID number
            address  - xtc.DetInfo object or address string
            
            address cannot be used together with any of  detector/detId/device/devId
        """

        if address is not None and (detector is not None or 
                detId is not None or device is not None or devId is not None ) :
            raise ValueError("Event._filter: address cannot be specified with other arguments")

        if address is not None :
            addrdict = _addr(address)
            address = addrdict.get('address')
            detector = addrdict.get('detector')
            detId = addrdict.get('detId')
            device = addrdict.get('device')
            devId = addrdict.get('devId')

        for x in Event._xtcGenerator( xtcObj ) :

            # check typeId
            if typeId is not None :
                if x.contains.id() != typeId : continue

            # check type version
            if version is not None :
                if x.contains.version() != version : continue

            src = x.src
                
            # check level
            if level is not None :
                if src.level() != level : continue

            # check DetInfo
            if address is not None :
                
                if not isinstance(src, (xtc.DetInfo, xtc.BldInfo)) : continue
                if src != address : continue

            else :
                
                # check detector
                if detector is not None :
                    if not isinstance(src, xtc.DetInfo) : continue
                    if src.detector() != detector : continue
                if detId is not None :
                    if not isinstance(src, xtc.DetInfo) : continue
                    if src.detId() != detId : continue
                    
                # check device
                if device is not None :
                    if not isinstance(src, xtc.DetInfo) : continue
                    if src.device() != device : continue
                if devId is not None :
                    if not isinstance(src, xtc.DetInfo) : continue
                    if src.devId() != devId : continue
                
            yield x

    # Generator method for datagram contents, does tree traversal
    @staticmethod
    def _xtcGenerator( xtcObj ):
        if xtcObj.contains.id() == xtc.TypeId.Type.Id_Xtc :
            yield xtcObj
            for child in xtcObj :
                for x in Event._xtcGenerator(child) :
                    yield x
        else :
            # skip damaged data
            if xtcObj.damage.value() == 0 :
                yield xtcObj
                    
#
# class that tracks values of all Epics channels
#
class EpicsStore(object):
    """
    Instance of this class contains current status of all EPICS channels. 
    It is updated from event data on every new event.
    """
    
    def __init__ (self):
        
        self.m_id2name = {}
        self.m_name2epics = {}
        
    def update(self, evt):
        """ self.update(evt)
        
        This method updates environment EPICS data from event object. 
        """
        
        epicsData = evt.findXtc(typeId=xtc.TypeId.Type.Id_Epics)
        for extc in epicsData :
            
            src = extc.src
            e = extc.payload()
            if e is None:
                
                guessId = (src.log(), src.phy(), lastPvId+1)
                guessName = self.m_id2name.get(id, None)
                if guessName:
                    _log.warning("Zero-size EPICS data found in the event, expected name is %s", guessName)
                else:
                    _log.warning("Zero-size EPICS data found in the event")
                
            else:

                lastPvId = e.iPvId
                id = (src.log(), src.phy(), e.iPvId)
                
                if epics.dbr_type_is_CTRL(e.iDbrType) :
                    # store mapping from name to ID
                    name = e.sPvName
                    self.m_id2name[id] = name
                else:
                    name = self.m_id2name.get(id, None)
                    
                self.m_name2epics[name] = e

    def value(self, name):
        """ self.value(name) -> EPICS object
        
        Parameters: ``name`` - name of the EPICS channel.
        
        This is the primary method to access EPICS information in pyana jobs.
        Returns current value of the EPICS channel with the given name. The type of returned data 
        is either epics.EpicsPvCtrl or epics.EpicsPvTime.
        """
        return self.m_name2epics.get(name, None)

class Env(object):
    """
    Class encapsulating all environment data such as configuration objects,
    job options, histogram manager, etc.
    """
    
    def __init__ (self, jobName="pyana", hmgr=None, subproc=-1, calibDir="", expNameProvider=None):
        """If subproc is negative it means main process, non-negative
        would mean subprocess"""
        
        self.m_jobName = jobName
        self.m_hmgr = hmgr
        self.m_subproc = subproc
        self.m_calibDir = Template(calibDir)
        self.m_expNameProvider = expNameProvider
        
        self.m_epics = EpicsStore()

        # store for configuration objects
        self.m_config = {}

        # in subprocess mode store the names of the files
        self.m_files = {}

    def jobName(self):
        """ self.jobName() -> string
        
        Returns job name.
        """
        return self.m_jobName
    
    def subprocess(self):
        """ self.subprocess() -> int
        
        Returns sub-process number. In case of multi-processing job it will be a non-negative number 
        ranging from 0 to a total number of sub-processes. In case of single-process job it will return -1.
        """
        return self.m_subproc
    
    def jobNameSub(self):
        """ self.jobNameSub() -> string
        
        Returns combination of job name and subprocess index as a string 
        which is unique for all subprocesses in a job.
        """
        if self.m_subproc < 0 :
            return self.m_jobName
        else :
            return "%s-%d" % (self.m_jobName, self.m_subproc)
    
    def instrument(self):
        """
        self.instrument() -> string
        
        Returns instrument name, or empty string if name is not known
        """
        if self.m_expNameProvider: return self.m_expNameProvider.instrument()
        return ""
    
    def experiment(self):
        """
        self.experiment() -> string
        
        Returns experiment name, or empty string if name is not known
        """
        if self.m_expNameProvider: return self.m_expNameProvider.experiment()
        return ""
    
    def calibDir(self):
        """
        self.calibDir() -> string
        
        Returns name of the calibration directory for current experiment.
        """
        return self.m_calibDir.substitute(instrument=self.instrument(), experiment=self.experiment())
    
    def hmgr(self):
        """ self.hmgr() -> pyana.histo.HistoMgr
        
        Returns histogram manager object."""
        
        if not self.m_hmgr:
            if self.m_subproc < 0 :
                # instantiate histo manager only when needed
                self.m_hmgr = HistoMgr( file=self.m_jobName+".root" )
            else :
                # instantiate histo manager (memory-based)
                self.m_hmgr = HistoMgr()

        return self.m_hmgr

    def mkfile(self, filename, mode='w', bufsize=-1):
        """self.mkfile(filename, mode='w', bufsize=-1) -> file
        
        Opens file for writing output data. This is pyana's alternative for Python open() function 
        which supports multi-processing. If user needs the data in this file to be merged with the 
        files produced by other processes then mkfile() has to be used in place of open(). This method 
        takes same parameters as regular open() call.
        
        In case of single-process job this method is equivalent to a regular Python open() method. 
        In case of multi-processing when this method is called from a sub-process then the file is 
        created somewhere in a temporary location (with unique name). At the end of the job files 
        from all sub-processes are merged into one file with name filename and the temporary 
        files are deleted. Merging is performed by concatenation, merge order is unspecified. 
        """

        if self.m_subproc < 0 :
            # in regular job just open the file
            return open(filename, mode, bufsize)
        else :
            # in child open temporary file and record its name
            fd, tmpname = tempfile.mkstemp(dir=os.path.dirname(filename))
            self.m_files[filename] = tmpname
            return os.fdopen(fd, mode, bufsize)

    def epicsStore(self):
        """ self.epicsStore() -> EpicsStore
        
        This is the primary method for user code to access EPICS data. 
        It returns event.EpicsStore object which can be used to retrieve the state of the 
        individual EPICS channels.
        """
        
        return self.m_epics
    
    def update(self, evt):
        """ self.update(evt)
        
        Parameters: ``evt`` - event object of type event.Event.
        
        This method updates environment contents with selected data from event object. 
        This is equivalent to calling env.updateEpics() and env.updateConfig().
        
        This method is not supposed to be called from user code, pyana takes care of all updates itself.
        """
        self.updateConfig(evt)
        self.updateEpics(evt)

    def updateEpics(self, evt):
        """ self.updateEpics(evt)
        
        This method updates environment EPICS data from event object. 
        """        
        # epics
        self.m_epics.update( evt )

    def updateConfig(self, evt):
        """ self.updateConfig(evt)
        
        This method copies configuration objects from event object into environment.
        """
        
        # store config objects 
        if evt.seq().service() in [xtc.TransitionId.Configure, xtc.TransitionId.BeginCalibCycle] :
            
            # get all XTCs at Source level and store their payload
            for level in (xtc.Level.Source, xtc.Level.Control):
                for x in evt.findXtc(level=level):
                    typeid = x.contains.id()
                    if typeid not in [xtc.TypeId.Type.Id_Epics, xtc.TypeId.Type.Id_Xtc, xtc.TypeId.Type.Any]:
                        # don't store Epics data as config
                        try :
                            cfgObj = x.payload()
                            self._storeConfig(typeid, x.src, cfgObj)
                        except Error, e:
                            _log.error('Failed to extract config object of type %s: %s', xtc.contains, e )

    def configKeys(self):
        """self.configKeys() -> list of tuples
        
        Returns the list of tuples (TypeId, Src), one tuple for every config object 
        contained in the environment.
        """
        result = []
        for k, v in self.m_config.iteritems():
            for vk in v.iterkeys():
                result.append((k, vk))
                
        return result
                    
    def getConfig(self, typeId, address=None):
        """ self.getConfig(typeId, address=None) -> config object
        
        Parameters: 
        ``typeId`` - one of the enum xtc.TypeId.Type values; 
        ``address`` - xtc.DetInfo object or an address string; 
        return type - configuration object or None.
    
        This is a generic method for finding a configuration object of given type. 
        If address is not given or is not very specific then the first matching object is returned.
        """
        _log.debug("Env.getConfig: %s %s", typeId, address)
        return self._getConfig(typeId=typeId, address=address)

    def getAcqConfig(self, address=None):
        """ self.getAcqConfig(address=None) -> config object
        
        Parameters: 
        ``address`` - xtc.DetInfo object or an address string; 
        return type - configuration object or None.
    
        Returns Acqiris configuration object for a given device address. If more than one object 
        is matched by the parameters then first arbitrary object is returned. 
        None is returned if no object is found.
        This method is equivalent to ``getConfig(xtc.TypeId.Type.Id_AcqConfig, address)``.
        """
        _log.debug("Env.getAcqConfig: %s", address)
        return self._getConfig(typeId=xtc.TypeId.Type.Id_AcqConfig, address=address)

    def getGsc16aiConfig(self, address=None):
        """ self.getGsc16aiConfig(address=None) -> config object
        
        Parameters: 
        ``address`` - xtc.DetInfo object or an address string; 
        return type - configuration object or None.
    
        Returns Gsc16ai configuration object for a given device address. If more than one object 
        is matched by the parameters then first arbitrary object is returned. 
        None is returned if no object is found.
        This method is equivalent to ``getConfig(xtc.TypeId.Type.Id_Gsc16aiConfig, address)``.
        """
        _log.debug("Env.getGsc16aiConfig: %s", address)
        return self._getConfig(typeId=xtc.TypeId.Type.Id_Gsc16aiConfig, address=address)

    def getOpal1kConfig(self, address=None):
        """ self.getOpal1kConfig(address=None) -> config object
        
        Parameters: 
        ``address`` - xtc.DetInfo object or an address string; 
        return type - configuration object or None.
    
        Returns Opal1k configuration object for a given device address. If more than one object 
        is matched by the parameters then first arbitrary object is returned. 
        None is returned if no object is found.
        This method is equivalent to ``getConfig(xtc.TypeId.Type.Id_Opal1kConfig, address)``.
        """
        _log.debug("Env.getOpal1kConfig: %s", address)
        return self._getConfig(typeId=xtc.TypeId.Type.Id_Opal1kConfig, address=address)

    def getPnCCDConfig(self, address=None):
        """ self.getPnCCDConfig(address=None) -> config object
        
        Parameters: 
        ``address`` - xtc.DetInfo object or an address string; 
        return type - configuration object or None.
    
        Returns PnCCD configuration object for a given device address. If more than one object 
        is matched by the parameters then first arbitrary object is returned. 
        None is returned if no object is found.
        This method is equivalent to ``getConfig(xtc.TypeId.Type.Id_pnCCDconfig, address)``.
        """
        _log.debug("Env.getPnCCDConfig: %s", address)
        return self._getConfig(typeId=xtc.TypeId.Type.Id_pnCCDconfig, address=address)

    def getPrincetonConfig(self, address=None):
        """ self.getPrincetonConfig(address=None) -> config object
        
        Parameters: 
        ``address`` - xtc.DetInfo object or an address string; 
        return type - configuration object or None.
    
        Returns Princeton configuration object for a given device address. If more than one object 
        is matched by the parameters then first arbitrary object is returned. 
        None is returned if no object is found.
        This method is equivalent to ``getConfig(xtc.TypeId.Type.Id_PrincetonConfig, address)``.
        """
        _log.debug("Env.getPrincetonConfig: %s", address)
        return self._getConfig(typeId=xtc.TypeId.Type.Id_PrincetonConfig, address=address)

    def result(self):
        """ self.result() -> dict
        
        Returns complete result of processing from a subprocess. This method
        is used by framework to collect result data from sub-processes, user code
        should never call this method.
        """
        if self.m_subproc >= 0 :

            # send back all histogram and all file names
            histos = []
            if self.m_hmgr : histos = self.m_hmgr.histos()
            
            return dict ( files = self.m_files,
                          histos = histos )

    def finish(self):
        """ self.finish()
        
        Finish job. This method is used by framework to finalize data collection, user code
        should never call this method.
        """
        if self.m_hmgr : self.m_hmgr.close()

    # ==================
    #  internal methods
    # ==================
    def _storeConfig(self, typeId, detInfo, cfgObj ):
        """ self._storeConfig(typeId, detInfo, cfgObj)

        Store one configuration object.
        """
        _log.debug("Env._storeConfig: typeId=%s detinfo=%s", typeId, detInfo)
        self.m_config.setdefault(typeId, {})[detInfo] = cfgObj        
        
    def _getConfig(self, typeId=None, detector=None, detId=None, device=None, devId=None, address=None):
        """ self._getConfig(typeId=None, detector=None, detId=None, device=None, devId=None, address=None) -> config object

        Find and return configuration object.
        """
        
        if address is not None :
            addrdict = _addr(address)
            address = addrdict.get('address')
            detector = addrdict.get('detector')
            detId = addrdict.get('detId')
            device = addrdict.get('device')
            devId = addrdict.get('devId')

        configMap = self.m_config.get(typeId, {})
        if address :
            return configMap.get(address, None)
        else:
            for k, v in configMap.iteritems() :
                if detector is not None and k.detector() != detector : continue
                if detId is not None and k.detId() != detId : continue
                if device is not None and k.device() != device : continue
                if devId is not None and k.devId() != devId : continue
                return v
