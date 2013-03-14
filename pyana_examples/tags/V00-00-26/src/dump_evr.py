#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module dump_evr...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

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
from pypdsdata import xtc

#----------------------------------
# Local non-exported definitions --
#----------------------------------

def _printPulseConfig(i, pcfg):
    print "  pulse config #%d: pulse=%d polarity=%d prescale=%d delay=%d width=%d" % \
        (i, pcfg.pulse(), pcfg.polarity(), pcfg.prescale(), pcfg.delay(), pcfg.width())

def _printPulseConfigV3(i, pcfg):
    print "  pulse config #%d: pulseId=%d polarity=%d prescale=%d delay=%d width=%d" % \
        (i, pcfg.pulseId(), pcfg.polarity(), pcfg.prescale(), pcfg.delay(), pcfg.width())

def _printOutputMap(i, ocfg):
    
    print "  output config #%d: source=%s source_id=%d conn=%s conn_id=%d" % \
        (i, ocfg.source(), ocfg.source_id(), ocfg.conn(), ocfg.conn_id())

def _printOutputMapV2(i, ocfg):
    
    print "  output config #%d: source=%s source_id=%d conn=%s conn_id=%d module=%d" % \
        (i, ocfg.source(), ocfg.source_id(), ocfg.conn(), ocfg.conn_id(), ocfg.module())

def _printEventCodeV3(i, ecfg):
    
    print "  event code #%d: code=%d isReadout=%d isTerminator=%d maskTrigger=%d maskSet=%d maskClear=%d" % \
        (i, ecfg.code(), ecfg.isReadout(), ecfg.isTerminator(), ecfg.maskTrigger(), ecfg.maskSet(), ecfg.maskClear())

def _printEventCodeV4(i, ecfg):
    
    print "  event code #%d: code=%d isReadout=%d isTerminator=%d reportDelay=%d reportWidth=%d maskTrigger=%d maskSet=%d maskClear=%d" % \
        (i, ecfg.code(), ecfg.isReadout(), ecfg.isTerminator(), ecfg.reportDelay(), ecfg.reportWidth(), \
         ecfg.maskTrigger(), ecfg.maskSet(), ecfg.maskClear())

def _printEventCodeV5(i, ecfg):
    
    print "  event code #%d: code=%d isReadout=%d isCommand=%d isLatch=%d reportDelay=%d reportWidth=%d maskTrigger=%d maskSet=%d maskClear=%d description=%s" % \
        (i, ecfg.code(), ecfg.isReadout(), ecfg.isCommand(), ecfg.isLatch(), ecfg.reportDelay(), ecfg.reportWidth(), \
         ecfg.maskTrigger(), ecfg.maskSet(), ecfg.maskClear(), ecfg.desc())

def _printEventCodeV6(i, ecfg):
    
    print "  event code #%d: code=%d isReadout=%d isCommand=%d isLatch=%d reportDelay=%d reportWidth=%d maskTrigger=%d maskSet=%d maskClear=%d readoutGroup=%d description=%s" % \
        (i, ecfg.code(), ecfg.isReadout(), ecfg.isCommand(), ecfg.isLatch(), ecfg.reportDelay(), ecfg.reportWidth(), \
         ecfg.maskTrigger(), ecfg.maskSet(), ecfg.maskClear(), ecfg.readoutGroup(), ecfg.desc())

def _printFIFOEvent(i, f):
    
    print "    fifo event #%d TimestampHigh=%d TimestampLow=%d EventCode=%d" % \
        (i, f.TimestampHigh, f.TimestampLow, f.EventCode)


    

#---------------------
#  Class definition --
#---------------------
class dump_evr (object) :
    """Class whose instance will be used as a user analysis module. """

    #--------------------
    #  Class variables --
    #--------------------
    
    # usual convention is to prefix static variables with s_
    s_staticVariable = 0

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, source="" ) :
        """Class constructor takes the name of the data source.

        @param source   data source
        """
        
        self.m_src = source

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :

        logging.info( "dump_evr.beginjob() called" )

        config = env.getConfig(xtc.TypeId.Type.Id_EvrConfig, self.m_src)
        if config:
            className = config.__class__.__name__
            print "%s: %s" % (className, self.m_src)
            
            
            if className == "ConfigV1":
                
                print "  npulses =", config.npulses()
                print "  noutputs =", config.noutputs()
                
                for i in range(config.npulses()): _printPulseConfig(i, config.pulse(i))
                for i in range(config.noutputs()): _printOutputMap(i, config.output_map(i))


            if className == "ConfigV2":
                
                print "  npulses =", config.npulses()
                print "  noutputs =", config.noutputs()
                print "  beam =", config.beam()
                print "  rate =", config.rate()
                
                for i in range(config.npulses()): _printPulseConfig(i, config.pulse(i))
                for i in range(config.noutputs()): _printOutputMap(i, config.output_map(i))


            if className == "ConfigV3":
                
                print "  npulses =", config.npulses()
                print "  noutputs =", config.noutputs()
                print "  neventcodes =", config.neventcodes()
                
                for i in range(config.npulses()): _printPulseConfigV3(i, config.pulse(i))
                for i in range(config.noutputs()): _printOutputMap(i, config.output_map(i))
                for i in range(config.neventcodes()): _printEventCodeV3(i, config.eventcode(i))


            if className == "ConfigV4":
                
                print "  npulses =", config.npulses()
                print "  noutputs =", config.noutputs()
                print "  neventcodes =", config.neventcodes()
                
                for i in range(config.npulses()): _printPulseConfigV3(i, config.pulse(i))
                for i in range(config.noutputs()): _printOutputMap(i, config.output_map(i))
                for i in range(config.neventcodes()): _printEventCodeV4(i, config.eventcode(i))


            if className == "ConfigV5":
                
                print "  npulses =", config.npulses()
                print "  noutputs =", config.noutputs()
                print "  neventcodes =", config.neventcodes()
                
                for i in range(config.npulses()): _printPulseConfigV3(i, config.pulse(i))
                for i in range(config.noutputs()): _printOutputMap(i, config.output_map(i))
                for i in range(config.neventcodes()): _printEventCodeV5(i, config.eventcode(i))

                scfg = config.seq_config()
                print "  seq_config: sync_source=%s beam_source=%s length=%d cycles=%d" % \
                    (scfg.sync_source(), scfg.beam_source(), scfg.length(), scfg.cycles())


            if className == "ConfigV6":
                
                print "  npulses =", config.npulses()
                print "  noutputs =", config.noutputs()
                print "  neventcodes =", config.neventcodes()
                
                for i in range(config.npulses()): _printPulseConfigV3(i, config.pulse(i))
                for i in range(config.noutputs()): _printOutputMapV2(i, config.output_map(i))
                for i in range(config.neventcodes()): _printEventCodeV5(i, config.eventcode(i))

                scfg = config.seq_config()
                print "  seq_config: sync_source=%s beam_source=%s length=%d cycles=%d" % \
                    (scfg.sync_source(), scfg.beam_source(), scfg.length(), scfg.cycles())


            if className == "ConfigV7":
                
                print "  npulses =", config.npulses()
                print "  noutputs =", config.noutputs()
                print "  neventcodes =", config.neventcodes()
                
                for i in range(config.npulses()): _printPulseConfigV3(i, config.pulse(i))
                for i in range(config.noutputs()): _printOutputMapV2(i, config.output_map(i))
                for i in range(config.neventcodes()): _printEventCodeV6(i, config.eventcode(i))

                scfg = config.seq_config()
                print "  seq_config: sync_source=%s beam_source=%s length=%d cycles=%d" % \
                    (scfg.sync_source(), scfg.beam_source(), scfg.length(), scfg.cycles())


    def event( self, evt, env ) :

        data = evt.get(xtc.TypeId.Type.Id_EvrData, self.m_src)
        if data:
            
            className = data.__class__.__name__
            print "%s: %s" % (className, self.m_src)

            print "  numFifoEvents =", data.numFifoEvents()
            for i in range(data.numFifoEvents()): _printFIFOEvent(i, data.fifoEvent(i))
            

    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param evt    event data object
        @param env    environment object
        """
        
        logging.info( "dump_evr.endjob() called" )
