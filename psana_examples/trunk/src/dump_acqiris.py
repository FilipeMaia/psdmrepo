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
        for ch in range(nch):
            try:
                v = config.vert(ch)
                print "  vert(%d):" % ch,
                print "fullScale =", v.fullScale(),
                print "slope =", v.slope(),
                print "offset =", v.offset(),
                print "coupling =", v.coupling(),
                print "bandwidth=", v.bandwidth()
            except:
                print "  [ ERROR fetching vert(%d) ]" % ch

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
            print "%s: %s: channel = %d" % (elem.__class__.__name__, self.m_src, chan)
            print "  nbrSegments =", elem.nbrSegments()
            print "  nbrSamplesInSeg =", elem.nbrSamplesInSeg()

            v_array = acqConfig.vert();
            """
            v = v_array.at(chan);
            print "v=", v
            slope = v.slope();
            print "slope=", slope
            offset = v.offset();
            print "offset=", offset
            """

            timestamp = elem.timestamp()
            print "type(timestamp)=", type(timestamp)
            print "timestamp.ndim=", timestamp.ndim
            print "timestamp.shape=", timestamp.shape
            print "timestamp.dtype=", timestamp.dtype
            print "timestamp.itemsize=", timestamp.itemsize
            print "timestamp.size=", timestamp.size
            """
            print "timestamp[0]=", timestamp[0]
            print "timestamp=", timestamp
            """
            #print "elem.timestamp()=", elem.timestamp()
            #print "  timestamps =", [elem.timestamp(seg) for seg in range(elem.nbrSegments())]

            wf = elem.waveforms()
            print "  waveform [len=%d] = %s" % (len(wf), wf)
