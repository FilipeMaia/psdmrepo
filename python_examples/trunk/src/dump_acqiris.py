#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: dump_acqiris.py 2622 2011-11-11 14:35:00Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  Pyana user analysis module dump_princeton...
#
#------------------------------------------------------------------------

"""Example module for accessing SharedIpimb data.

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
from pypdsdata import xtc

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
    def beginjob( self, evt, env ) :
        
        src = env.configStr("source", "DetInfo(:Acqiris)")
        print 'env.configStr("source", "DetInfo(:Acqiris)") ->', src

        foundSrc_list = []
        config = env.configStore().get("Psana::Acqiris::ConfigV1", env.Source(src), foundSrc_list);
        foundSrc = foundSrc_list[0]
        print "foundSrc: log()=%x, phy()=%x" % (foundSrc.log(), foundSrc.phy())
        if config:
        
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
                v = config.vert()[ch]
                print "  vert(%d):" % ch,
                print "fullScale =", v.fullScale(),
                print "slope =", v.slope(),
                print "offset =", v.offset(),
                print "coupling =", v.coupling(),
                print "bandwidth=", v.bandwidth()


    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        print "... self.m_src=", self.m_src
        acqData = evt.get("Psana::Acqiris::DataDesc", env.Source(self.m_src))

        nchan = acqData.data_shape()[0];
        for chan in range(nchan):
            elem = acqData.data(chan)

            print "%s: %s: channel = %d" % (elem.__class__.__name__, self.m_src, chan)
            print "  nbrSegments =", elem.nbrSegments()
            print "  nbrSamplesInSeg =", elem.nbrSamplesInSeg()
            print "  timestamps =", [elem.timestamp()[seg].pos() for seg in range(elem.nbrSegments())]
            wf = elem.waveforms()
            print "  waveform [len=%d] = %s" % (len(wf), wf)


    def endjob( self, evt, env ) :
        
        pass

