#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: dump_simple.py 2622 2011-11-11 14:35:00Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  Class dump_acqiris
#
#------------------------------------------------------------------------

"""Example module for accessing Acqiris data.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: dump_acqiris.py 2622 2011-11-11 14:35:00Z salnikov@SLAC.STANFORD.EDU $

@author Andy Salnikov
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 2622 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

import numpy

#---------------------
#  Class definition --
#---------------------
class dump_acqiris (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, source="" ) :
        self.m_src = source

    #-------------------
    #  Public methods --
    #-------------------
    def beginCalibCycle( self, evt, env ) :
        self.source = env.configStr("source", "DetInfo(:Acqiris)")
        config = env.configStore().get("Psana::Acqiris::Config", self.source)
        if not config:
            return
        
        print "%s: %s" % (config.__class__.__name__, self.m_src)
            
        print "  nbrBanks =", config.nbrBanks(),
        print "channelMask =", config.channelMask(),
        print "nbrChannels =", config.nbrChannels(),
        print "nbrConvertersPerChannel =", config.nbrConvertersPerChannel()
     
        h = config.horiz()
        print "  horiz: sampInterval =", h.sampInterval(),
        print "delayTime =", h.delayTime(),
        print "nbrSegments =", h.nbrSegments(),
        print "nbrSamples =", h.nbrSamples()

        nch = config.nbrChannels()
        vert = config.vert()
        for ch in range(nch):
            v = vert[ch]
            print "  vert(%d):" % ch,
            print "fullScale =", v.fullScale()
            print "slope =", v.slope()
            print "offset =", v.offset()
            print "coupling =", v.coupling()
            print "bandwidth=", v.bandwidth()

    def event( self, evt, env ) :
        acqData = evt.get("Psana::Acqiris::DataDesc", self.source)
        if not acqData:
            return

        # find matching config object
        acqConfig = env.configStore().get("Psana::Acqiris::Config", self.source)

        # loop over channels
        nchan = acqData.data_shape()[0];
        for chan in range(nchan):
            elem = acqData.data(chan);
            v = acqConfig.vert()[chan]
            slope = v.slope()
            offset = v.offset()

            print "Acqiris::DataDescV1: channel=%d" % chan ### XXX should print real class name instead
            print "  nbrSegments=%d" % elem.nbrSegments()
            print "  nbrSamplesInSeg=%d" % elem.nbrSamplesInSeg()
            print "  indexFirstPoint=%d" % elem.indexFirstPoint()

            timestamps = elem.timestamp()
            waveforms = elem.waveforms()

            # loop over segments
            for seg in range(elem.nbrSegments()):
                print "  Segment #%d" % seg
                print "    timestamp=%d" % timestamps[seg].pos()
                print "    data=[",
                size = min(elem.nbrSamplesInSeg(), 32)
                for i in range(size):
                    print "%f," % (waveforms[seg][i]*slope + offset),
                print "...]"
