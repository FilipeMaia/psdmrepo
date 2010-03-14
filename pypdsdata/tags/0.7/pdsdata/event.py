#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

import logging
import numpy

from pdsdata import xtc
from pdsdata import epics
from pdsdata import Error
from pdsdata import acqiris
from pdsdata import pnccd

_log = logging.getLogger("pdsdata.event")



def _addr( address ):
    """Parse address string and returns dictionary with keys 
    (detector, detId, device, devId), some or all values can be None"""
    
    if type(address) == xtc.DetInfo :
        return dict(detInfo=address)
    
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
    
    
    def __init__( self, dg ):
        
        self.m_dg = dg
        

    def seq(self) :
        return self.m_dg.seq

    def env(self) :
        return self.m_dg.env

    def getTime(self) :
         return self.m_dg.seq.clock()

    def findXtc(self, **kw):
        """ Returns the list of Xtc objects matching the criteria.
        Accepts keyword parameters same as _filter() """

        return list( Event._filter(self.m_dg.xtc, **kw) )

    def find(self, **kw):
        """ Returns the list of payload objects matching the criteria.
        Accepts keyword parameters same as _filter() """

        return [ x.payload() for x in Event._filter(self.m_dg.xtc,**kw) ]

    def findFirstXtc(self, **kw):
        """ Returns the first Xtc object matching the criteria.
        Accepts keyword parameters same as _filter() """
        
        gen = Event._filter(self.m_dg.xtc, **kw)
        try :
            return gen.next()
        except StopIteration:
            return None

    def findFirst(self, **kw):
        """ Returns the list of Xtc objects matching the criteria.
        Accepts keyword parameters same as _filter() """
        
        gen = Event._filter(self.m_dg.xtc, **kw)
        try :
            return gen.next().payload()
        except StopIteration:
            return None


    def getAcqConfig(self, address):
        """ returns AcqConfig for specific address"""
        
        addrdict = _addr(address)
        return self.findFirst( typeId=xtc.TypeId.Type.Id_AcqConfig, **addrdict )

    def getAcqValue(self, address, channel, env):
        """ returns Acquiris data for specific detector and channel, 
        detector is one of xtc.DetInfo.Detector.Value constants. 
        This methods requires environment object to get access to 
        Acqiris configuration object. Returned object is an iterator
        which returns tuples - timestamp and voltage value"""

        _log.debug("Event.getAcqValue: address=%s channel=%s", address, channel)

        addrdict = _addr(address)

        xtcObj = self.findFirstXtc( typeId=xtc.TypeId.Type.Id_AcqWaveform, **addrdict )
        if not xtcObj : return None
        obj = xtcObj.payload()

        # get config object
        cfg = env.getAcqConfig(detInfo=xtcObj.src)
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
    
    def getPnCcdValue(self, address, env):
        """ returns PnCCDFrameV1 for specific address"""
        
        _log.debug("Event.getPnCcdValue: address=%s", address)

        addrdict = _addr(address)

        xtcObj = self.findFirstXtc( typeId=xtc.TypeId.Type.Id_pnCCDframe, **addrdict )
        if not xtcObj : return None
        frame = xtcObj.payload()

        # get config object
        cfg = env.getPnCCDConfig(detInfo=xtcObj.src)
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

    def getOpal1kConfig(self, address):
        """Returns opal1k.ConfigV1 object"""
        addrdict = _addr(address)
        return self.findFirst( typeId=xtc.TypeId.Type.Id_Opal1kConfig, **addrdict )

    def getFeeGasDet(self):
        """Returns list of 4 floating numbers"""
        obj = self.findFirst( typeId=xtc.TypeId.Type.Id_FEEGasDetEnergy )
        if obj :
            return [obj.f_11_ENRC, obj.f_12_ENRC, obj.f_21_ENRC, obj.f_22_ENRC]

    def getPhaseCavity(self):
        """Returns bld.BldDataPhaseCavity object"""
        return self.findFirst( typeId=xtc.TypeId.Type.Id_PhaseCavity )

    def getEBeam(self):
        """Returns bld.BldDataEBeam or bld.BldDataEBeamV0 object whichever is present"""
        return self.findFirst( typeId=xtc.TypeId.Type.Id_EBeam )

    #
    # Private methods not to be used by clients directly
    #

    @staticmethod
    def _filter(xtcObj, typeId=None, version=None, level=None,
                detector=None, detId=None, device=None, devId=None, detInfo=None):
        """ generator which produces Xtc objects matching specified criteria.
        
            typeId   - one of xtc.TypeId.Type.Something values
            version  - TypeId version number, if missing return any version
            level    - one of xtc.Level.Something values
            detector - one of xtc.DetInfo.Detector.Something values
            detId    - detector ID number
            device   - one of xtc.DetInfo.Device.Something values
            devId    - device ID number
        """

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
            if detInfo is not None :
                
                if not isinstance(src, xtc.DetInfo) : continue
                if src != detInfo : continue

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
        yield xtcObj
        if xtcObj.contains.id() == xtc.TypeId.Type.Id_Xtc :
            for child in xtcObj :
                for x in Event._xtcGenerator(child) :
                    yield x
                    
                    
#
# class that tracks values of all Epics channels
#
class EpicsStore(object):
    
    def __init__ (self):
        
        self.m_name2id = {}
        self.m_id2epics = {}
        
    def update(self, evt):
        
        epicsData = evt.find(typeId=xtc.TypeId.Type.Id_Epics)
        for e in epicsData :
            if epics.dbr_type_is_CTRL(e.iDbrType) :
                # store mapping from name to ID
                self.m_name2id[e.sPvName] = e.iPvId
            self.m_id2epics[e.iPvId] = e

    def value(self, name):
        return self.m_id2epics.get(self.m_name2id.get(name, None), None)

#
# Environment
#
class Env(object):
    
    def __init__ (self, jobName="pyana" ):
        
        self.m_jobName = jobName
        
        self.m_epics = EpicsStore()

        # store for configuration objects
        self.m_config = {}

    def jobName(self):
        return self.m_jobName

    def epicsStore(self):
        return self.m_epics
    
    def update(self, evt):
        """ Update environment with some info from event """
        
        # epics
        self.m_epics.update( evt )

        # store config objects 
        if evt.seq().service() == xtc.TransitionId.Configure :
            
            types = [xtc.TypeId.Type.Id_AcqConfig, 
                     xtc.TypeId.Type.Id_pnCCDconfig,
                     xtc.TypeId.Type.Id_Opal1kConfig]
            
            for typeId in types :
                for x in evt.findXtc( typeId=typeId ) :
                    self._storeConfig(typeId, x.src, x.payload())
                    
    def getAcqConfig(self, **kw):
        _log.debug("Env.getAcqConfig: %s", kw)
        return self._getConfig(xtc.TypeId.Type.Id_AcqConfig, **kw)

    def getPnCCDConfig(self, **kw):
        _log.debug("Env.getPnCCDConfig: %s", kw)
        return self._getConfig(xtc.TypeId.Type.Id_pnCCDconfig, **kw)

    def getOpal1kConfig(self, **kw):
        _log.debug("Env.getOpal1kConfig: %s", kw)
        return self._getConfig(xtc.TypeId.Type.Id_Opal1kConfig, **kw)

    def _storeConfig(self, typeId, detInfo, cfgObj ):
        _log.debug("Env._storeConfig: typeId=%s detinfo=%s", typeId, detInfo)
        self.m_config.setdefault(typeId, {})[detInfo] = cfgObj        
        
    def _getConfig(self, typeId=None, detector=None, detId=None, device=None, devId=None, detInfo=None):
        
        configMap = self.m_config.get(typeId, {})
        if detInfo :
            return configMap.get(detInfo, None)
        else:
            for k, v in configMap.iteritems() :
                if detector is not None and k.detector() != detector : continue
                if detId is not None and k.detId() != detId : continue
                if device is not None and k.device() != device : continue
                if devId is not None and k.devId() != devId : continue
                return v
        