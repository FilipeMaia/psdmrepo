#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

import logging
import numpy
import os
import tempfile

from pyana.histo import HistoMgr
from pypdsdata import xtc
from pypdsdata import epics
from pypdsdata import Error
from pypdsdata import acqiris
from pypdsdata import cspad
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

#
# Class encapsulating event data 
#
class Event(object):
    
    
    def __init__( self, dg, run = -1 ):
        
        self.m_dg = dg
        self.m_run = run
        self.m_userData = {}

    def run(self) :
        return self.m_run

    def seq(self) :
        return self.m_dg.seq

    def env(self) :
        return self.m_dg.env

    def damage(self) :
        return self.m_dg.xtc.damage

    def getTime(self) :
         return self.m_dg.seq.clock()

    def findXtc(self, **kw):
        """ Returns the list of Xtc objects matching the criteria.
        Accepts keyword parameters same as _filter() """

        return list( Event._filter(self.m_dg.xtc, **kw) )

    def find(self, **kw):
        """ Returns the list of payload objects matching the criteria.
        Accepts keyword parameters same as _filter() """

        objects = [x.payload() for x in Event._filter(self.m_dg.xtc,**kw)]
        return filter(None, objects)

    def findFirstXtc(self, **kw):
        """ Returns the first Xtc object matching the criteria.
        Accepts keyword parameters same as _filter() """
        
        for x in Event._filter(self.m_dg.xtc, **kw):
            return x
        return None

    def findFirst(self, **kw):
        """ Returns first data object (XTC's payload) matching the criteria.
        Accepts keyword parameters same as _filter() """
        
        for x in Event._filter(self.m_dg.xtc, **kw):
            obj = x.payload()
            if obj: return obj
        return None

    def get(self, key, address=None):
        """Generic get method, retieves detector data or user data"""
        if isinstance(key,int):
            return self.findFirst( typeId=key, address=address )
        else :
            default = address
            return self.m_userData.get(key, default)

    def getAcqValue(self, address, channel, env):
        """ returns Acquiris data for specific detector and channel, 
        address is the DetInfo object or address string. 
        This methods requires environment object to get access to 
        Acqiris configuration object. Returned object is an iterator
        which returns tuples - timestamp and voltage value"""

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
        """Returns bld.BldDataEBeam or bld.BldDataEBeamV0 object whichever is present"""
        return self.findFirst( typeId=xtc.TypeId.Type.Id_EBeam )

    def getEvrData(self, address):
        """Returns EvrData for specific address"""
        return self.findFirst( typeId=xtc.TypeId.Type.Id_EvrData, address=address )

    def getFeeGasDet(self):
        """Returns list of 4 floating numbers"""
        obj = self.findFirst( typeId=xtc.TypeId.Type.Id_FEEGasDetEnergy )
        if obj :
            return [obj.f_11_ENRC, obj.f_12_ENRC, obj.f_21_ENRC, obj.f_22_ENRC]

    def getFrameValue(self, address):
        """Returns Frame data for specific address"""
        return self.findFirst( typeId=xtc.TypeId.Type.Id_Frame, address=address )

    def getIpimbValue(self, address):
        """Returns Ipimb data for specific address"""
        return self.findFirst( typeId=xtc.TypeId.Type.Id_IpimbData, address=address )

    getOpal1kValue = getFrameValue

    def getPhaseCavity(self):
        """Returns bld.BldDataPhaseCavity object"""
        return self.findFirst( typeId=xtc.TypeId.Type.Id_PhaseCavity )

    def getPnCcdValue(self, address, env):
        """ returns PnCCDFrameV1 for specific address"""
        
        _log.debug("Event.getPnCcdValue: address=%s", address)

        xtcObj = self.findFirstXtc( typeId=xtc.TypeId.Type.Id_pnCCDframe, address=address )
        if not xtcObj : return None
        frame = xtcObj.payload()

        # get config object
        cfg = env.getPnCCDConfig(address=xtcObj.src)
        if not cfg : 
            raise Error("cannot find PnCCD config for address %s" % xtcObj.src )

        # get all frames
        frames = []
        numLinks = cfg.numLinks()
        while True:
            frames.append(frame)
            if len(frames) >= numLinks : break
            frame = frame.next(cfg)
        
        return pnccd.FrameV1( frames, cfg )

    def getPrincetonValue(self, address, env):
        """ returns Acquiris data for specific detector and channel, 
        address is the DetInfo object or address string."""

        _log.debug("Event.getPrincetonValue: address=%s", address)

        xtcObj = self.findFirstXtc( typeId=xtc.TypeId.Type.Id_PrincetonFrame, address=address )
        if not xtcObj : return None
        obj = xtcObj.payload()

        # get config object
        cfg = env.getPrincetonConfig(address=xtcObj.src)
        if not cfg : 
            raise Error("cannot find Princeton config for address %s" % xtcObj.src )

        return princeton.FrameV1( obj, cfg )
    
    def getCsPadQuads(self, address, env):
        """ returns CsPadElement for specific address"""
        
        _log.debug("Event.getCsPadElements: address=%s", address)

        xtcObj = self.findFirstXtc( typeId=xtc.TypeId.Type.Id_CspadElement, address=address )
        if not xtcObj : return None
        quad = xtcObj.payload()

        # get config object
        cfg = env.getConfig(typeId=xtc.TypeId.Type.Id_CspadConfig, address=xtcObj.src)
        if not cfg : 
            raise Error("cannot find CsPad config for address %s" % xtcObj.src )

        # get all elements
        quads = []
        numQuads = cfg.numQuads()
        while True:
            quads.append(cspad.wrapElement(quad, cfg))
            if len(quads) >= numQuads : break
            quad = quad.next(cfg)
        
        # quads may be unordered in XTC, clients prefer them ordered
        quads.sort(cmp = lambda x, y: cmp(x.quad(), y.quad()))
        return quads

    def put(self, data, key):
        """ Add user data to the event, key identifies the data """
        self.m_userData[key] = data


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
    
    def __init__ (self):
        
        self.m_name2id = {}
        self.m_id2epics = []
        
    def update(self, evt):
        
        epicsData = evt.find(typeId=xtc.TypeId.Type.Id_Epics)
        for e in epicsData :
            if epics.dbr_type_is_CTRL(e.iDbrType) :
                # store mapping from name to ID
                self.m_name2id[e.sPvName] = e.iPvId
            if e.iPvId >= len(self.m_id2epics) :
                self.m_id2epics.extend( [None]*(e.iPvId+1-len(self.m_id2epics)))
            self.m_id2epics[e.iPvId] = e

    def value(self, name):
        id = self.m_name2id.get(name, None)
        if id is None : return None
        return self.m_id2epics[id]

#
# Environment
#
class Env(object):
    
    def __init__ (self, jobName="pyana", hmgr=None, subproc=-1 ):
        """If subproc is negative it means main process, non-negative
        would mean subprocess"""
        
        self.m_jobName = jobName
        self.m_hmgr = hmgr
        self.m_subproc = subproc
        
        self.m_epics = EpicsStore()

        # store for configuration objects
        self.m_config = {}

        # in subprocess mode store the names of the files
        self.m_files = {}

    def jobName(self):
        return self.m_jobName
    
    def subprocess(self):
        return self.m_subproc
    
    def jobNameSub(self):
        if self.m_subproc < 0 :
            return self.m_jobName
        else :
            return "%s-%d" % (self.m_jobName, self.m_subproc)
    
    def hmgr(self):
        "Returns histogram manager object"
        
        if not self.m_hmgr:
            if self.m_subproc < 0 :
                # instantiate histo manager only when needed
                self.m_hmgr = HistoMgr( file=self.m_jobName+".root" )
            else :
                # instantiate histo manager (memory-based)
                self.m_hmgr = HistoMgr()

        return self.m_hmgr

    def mkfile(self, filename, mode='w', bufsize=-1):

        if self.m_subproc < 0 :
            # in regular job just open the file
            return open(filename, mode, bufsize)
        else :
            # in child open temporary file and record its name
            fd, tmpname = tempfile.mkstemp()
            self.m_files[filename] = tmpname
            return os.fdopen(fd, mode, bufsize)

    def epicsStore(self):
        return self.m_epics
    
    def update(self, evt):
        """ Update environment with some info from event """
        self.updateConfig(evt)
        self.updateEpics(evt)

    def updateEpics(self, evt):
        """ Update environment with epics info from event """        
        # epics
        self.m_epics.update( evt )

    def updateConfig(self, evt):
        """ Update environment with config info from event """
        
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
                    
    def getConfig(self, typeId, address=None):
        """Generic getConfig method"""
        _log.debug("Env.getConfig: %s %s", typeId, address)
        return self._getConfig(typeId=typeId, address=address)

    def getAcqConfig(self, address=None):
        _log.debug("Env.getAcqConfig: %s", address)
        return self._getConfig(typeId=xtc.TypeId.Type.Id_AcqConfig, address=address)

    def getOpal1kConfig(self, address=None):
        _log.debug("Env.getOpal1kConfig: %s", address)
        return self._getConfig(typeId=xtc.TypeId.Type.Id_Opal1kConfig, address=address)

    def getPnCCDConfig(self, address=None):
        _log.debug("Env.getPnCCDConfig: %s", address)
        return self._getConfig(typeId=xtc.TypeId.Type.Id_pnCCDconfig, address=address)

    def getPrincetonConfig(self, address=None):
        _log.debug("Env.getPrincetonConfig: %s", address)
        return self._getConfig(typeId=xtc.TypeId.Type.Id_PrincetonConfig, address=address)

    def result(self):
        """returns complete result of processing from a subprocess"""
        if self.m_subproc >= 0 :

            # send back all histogram and all file names
            histos = []
            if self.m_hmgr : histos = self.m_hmgr.histos()
            
            return dict ( files = self.m_files,
                          histos = histos )

    def finish(self):
        if self.m_hmgr : self.m_hmgr.close()

    # ==================
    #  internal methods
    # ==================
    def _storeConfig(self, typeId, detInfo, cfgObj ):
        _log.debug("Env._storeConfig: typeId=%s detinfo=%s", typeId, detInfo)
        self.m_config.setdefault(typeId, {})[detInfo] = cfgObj        
        
    def _getConfig(self, typeId=None, detector=None, detId=None, device=None, devId=None, address=None):
        
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
