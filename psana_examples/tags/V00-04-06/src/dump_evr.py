#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Psana user analysis module dump_evr...
#
#------------------------------------------------------------------------

"""Psana python module which dumps EVR data.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andy Salnikov
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------
from psana import *

#----------------------------------
# Local non-exported definitions --
#----------------------------------

def _dumpfields(obj, fields):
    """ generic method to dump values of object attributes """
    return ' '.join(["{0}={1}".format(f, getattr(obj, f)()) for f in fields])


def _pulseConfigV0(i, cfg):
    fields = ['pulse', 'polarity', 'prescale', 'delay', 'width']
    return "  pulse config #{0}: {1}".format(i, _dumpfields(cfg, fields))

def _pulseConfigV3(i, cfg):
    fields = ['pulseId', 'polarity', 'prescale', 'delay', 'width']
    return "  pulse config #{0}: {1}".format(i, _dumpfields(cfg, fields))

def _pulseConfig(i, cfg):
    if isinstance(cfg, EvrData.PulseConfig): return _pulseConfigV0(i, cfg)
    if isinstance(cfg, EvrData.PulseConfigV3): return _pulseConfigV3(i, cfg)


def _outputMapConfigV0(i, cfg):
    fields = ['source', 'source_id', 'conn', 'conn_id', 'map']
    return "  output config #{0}: {1}".format(i, _dumpfields(cfg, fields))

def _outputMapConfigV2(i, cfg):
    fields = ['source', 'source_id', 'conn', 'conn_id', 'module', 'map']
    return "  output config #{0}: {1}".format(i, _dumpfields(cfg, fields))

def _outputMapConfig(i, cfg):
    if isinstance(cfg, EvrData.OutputMap): return _outputMapConfigV0(i, cfg)
    if isinstance(cfg, EvrData.OutputMapV2): return _outputMapConfigV2(i, cfg)


def _eventCodeConfigV3(i, cfg):
    fields = ['code', 'isReadout', 'isTerminator', 'maskTrigger', 'maskSet', 'maskClear']
    return "  event code #{0}: {1}".format(i, _dumpfields(cfg, fields))
        
def _eventCodeConfigV4(i, cfg):
    fields = ['code', 'isReadout', 'isTerminator', 'reportDelay', 'reportWidth', 'maskTrigger', 'maskSet', 'maskClear']
    return "  event code #{0}: {1}".format(i, _dumpfields(cfg, fields))

def _eventCodeConfigV5(i, cfg):
    fields = ['code', 'isReadout', 'isCommand', 'isLatch', 'reportDelay', 'reportWidth', 'maskTrigger', 'maskSet', 'maskClear', 'desc']
    return "  event code #{0}: {1}".format(i, _dumpfields(cfg, fields))

def _eventCodeConfigV6(i, cfg):
    fields = ['code', 'isReadout', 'isCommand', 'isLatch', 'reportDelay', 'reportWidth', 'maskTrigger', 'maskSet', 'maskClear', 'readoutGroup', 'desc']
    return "  event code #{0}: {1}".format(i, _dumpfields(cfg, fields))

def _eventCodeConfig(i, cfg):
    if isinstance(cfg, EvrData.EventCodeV3): return _eventCodeConfigV3(i, cfg)
    if isinstance(cfg, EvrData.EventCodeV4): return _eventCodeConfigV4(i, cfg)
    if isinstance(cfg, EvrData.EventCodeV5): return _eventCodeConfigV5(i, cfg)
    if isinstance(cfg, EvrData.EventCodeV6): return _eventCodeConfigV6(i, cfg)

def _sequencerConfig(cfg):
    fields = ['sync_source', 'beam_source', 'length', 'cycles']
    return "  seq_config: {0}".format(_dumpfields(cfg, fields))

def _fifoevent(i, fifo):
    fields = ['timestampHigh', 'timestampLow', 'eventCode']
    return "  fifo event #{0}: {1}".format(i, _dumpfields(fifo, fields))

#---------------------
#  Class definition --
#---------------------
class dump_evr (object) :
    '''Class whose instance will be used as a user analysis module.'''

    #----------------
    #  Constructor --
    #----------------
    def __init__(self):
        '''Class constructor. Does not need any parameters'''

        self.m_src = self.configSrc('source', 'DetInfo(:Evr)')

    #-------------------
    #  Public methods --
    #-------------------

    def beginRun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        config = env.configStore().get(EvrData.Config, self.m_src)
        if config:
        
            print "%s: %s" % (config.__class__.__name__, self.m_src)

            for tup in enumerate(config.pulses()): print _pulseConfig(*tup)
            for tup in enumerate(config.output_maps()): print _outputMapConfig(*tup)
            try:
                for tup in enumerate(config.eventcodes()): print _eventCodeConfig(*tup)
            except:
                pass
            try:
                print _sequencerConfig(config.seq_config())
            except:
                pass

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        data = evt.get(EvrData.Data, self.m_src)
        if data:
            print "{0}: numFifoEvents={1}".format(data.__class__.__name__, data.numFifoEvents())
            for tup in enumerate(data.fifoEvents()): print _fifoevent(*tup)

