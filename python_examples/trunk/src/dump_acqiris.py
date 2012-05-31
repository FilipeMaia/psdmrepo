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
        """Class constructor takes the name of the data source.

        @param source   data source
        """
        
        self.m_src = source

    #-------------------
    #  Public methods --
    #-------------------
    def beginjob( self, evt, env ) :
        
        config = env.getConfig(xtc.TypeId.Type.Id_AcqConfig, self.m_src)
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
                v = config.vert(ch)
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

        acqData = evt.get(xtc.TypeId.Type.Id_AcqWaveform, self.m_src)

        for chan, elem in enumerate(acqData):

            print "%s: %s: channel = %d" % (elem.__class__.__name__, self.m_src, chan)
            print "  nbrSegments =", elem.nbrSegments()
            print "  nbrSamplesInSeg =", elem.nbrSamplesInSeg()
            print "  timestamps =", [elem.timestamp(seg) for seg in range(elem.nbrSegments())]
            wf = elem.waveform()
            print "  waveform [len=%d] = %s" % (len(wf), wf)


    def endjob( self, evt, env ) :
        
        pass

