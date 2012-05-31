#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: dump_timepix.py 3076 2012-03-14 21:51:06Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  Pyana user analysis module dump_timepix...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: dump_timepix.py 3076 2012-03-14 21:51:06Z salnikov@SLAC.STANFORD.EDU $

@author Andy Salnikov
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 3076 $"
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

#---------------------
#  Class definition --
#---------------------
class dump_timepix (object) :
    """example analysis module which dumps Timepix objects"""

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
        
        config = env.getConfig(xtc.TypeId.Type.Id_TimepixConfig, self.m_src)
        if config:
        
            print "%s: %s" % (config.__class__.__name__, self.m_src)
            
            print "  readoutSpeed =", config.readoutSpeed()
            print "  triggerMode =", config.triggerMode()
            try:
                # ConfigV1
                print "  shutterTimeout =", config.shutterTimeout()
            except:
                pass
            try:
                # ConfigV2
                print "  timepixSpeed =", config.timepixSpeed()
            except:
                pass
            print "  dac0Ikrum =", config.dac0Ikrum()
            print "  dac0Disc =", config.dac0Disc()
            print "  dac0Preamp =", config.dac0Preamp()
            print "  dac0BufAnalogA =", config.dac0BufAnalogA()
            print "  dac0BufAnalogB =", config.dac0BufAnalogB()
            print "  dac0Hist =", config.dac0Hist()
            print "  dac0ThlFine =", config.dac0ThlFine()
            print "  dac0ThlCourse =", config.dac0ThlCourse()
            print "  dac0Vcas =", config.dac0Vcas()
            print "  dac0Fbk =", config.dac0Fbk()
            print "  dac0Gnd =", config.dac0Gnd()
            print "  dac0Ths =", config.dac0Ths()
            print "  dac0BiasLvds =", config.dac0BiasLvds()
            print "  dac0RefLvds =", config.dac0RefLvds()
            print "  dac1Ikrum =", config.dac1Ikrum()
            print "  dac1Disc =", config.dac1Disc()
            print "  dac1Preamp =", config.dac1Preamp()
            print "  dac1BufAnalogA =", config.dac1BufAnalogA()
            print "  dac1BufAnalogB =", config.dac1BufAnalogB()
            print "  dac1Hist =", config.dac1Hist()
            print "  dac1ThlFine =", config.dac1ThlFine()
            print "  dac1ThlCourse =", config.dac1ThlCourse()
            print "  dac1Vcas =", config.dac1Vcas()
            print "  dac1Fbk =", config.dac1Fbk()
            print "  dac1Gnd =", config.dac1Gnd()
            print "  dac1Ths =", config.dac1Ths()
            print "  dac1BiasLvds =", config.dac1BiasLvds()
            print "  dac1RefLvds =", config.dac1RefLvds()
            print "  dac2Ikrum =", config.dac2Ikrum()
            print "  dac2Disc =", config.dac2Disc()
            print "  dac2Preamp =", config.dac2Preamp()
            print "  dac2BufAnalogA =", config.dac2BufAnalogA()
            print "  dac2BufAnalogB =", config.dac2BufAnalogB()
            print "  dac2Hist =", config.dac2Hist()
            print "  dac2ThlFine =", config.dac2ThlFine()
            print "  dac2ThlCourse =", config.dac2ThlCourse()
            print "  dac2Vcas =", config.dac2Vcas()
            print "  dac2Fbk =", config.dac2Fbk()
            print "  dac2Gnd =", config.dac2Gnd()
            print "  dac2Ths =", config.dac2Ths()
            print "  dac2BiasLvds =", config.dac2BiasLvds()
            print "  dac2RefLvds =", config.dac2RefLvds()
            print "  dac3Ikrum =", config.dac3Ikrum()
            print "  dac3Disc =", config.dac3Disc()
            print "  dac3Preamp =", config.dac3Preamp()
            print "  dac3BufAnalogA =", config.dac3BufAnalogA()
            print "  dac3BufAnalogB =", config.dac3BufAnalogB()
            print "  dac3Hist =", config.dac3Hist()
            print "  dac3ThlFine =", config.dac3ThlFine()
            print "  dac3ThlCourse =", config.dac3ThlCourse()
            print "  dac3Vcas =", config.dac3Vcas()
            print "  dac3Fbk =", config.dac3Fbk()
            print "  dac3Gnd =", config.dac3Gnd()
            print "  dac3Ths =", config.dac3Ths()
            print "  dac3BiasLvds =", config.dac3BiasLvds()
            print "  dac3RefLvds =", config.dac3RefLvds()
            
            try:
                # ConfigV2
                print "  chipCount =", config.chipCount()
                print "  driverVersion =", config.driverVersion()
                print "  firmwareVersion =", config.firmwareVersion()
                print "  pixelThreshSize =", config.pixelThreshSize()
                print "  pixelThresh =", config.pixelThresh()
                print "  chip names =", config.chip0Name(), config.chip1Name(), config.chip2Name(), config.chip3Name()
                print "  chip IDs =", config.chip0ID(), config.chip1ID(), config.chip2ID(), config.chip3ID()
            except:
                pass


    def event( self, evt, env ) :

        data = evt.get(xtc.TypeId.Type.Id_TimepixData, self.m_src)
        if data:
            print "%s: %s" % (data.__class__.__name__, self.m_src)
            
            print "  timestamp =", data.timestamp()
            print "  frameCounter =", data.frameCounter()
            print "  lostRows =", data.lostRows()
            
            img = data.data()
            print "  data.shape =", img.shape
            print "  data =", img

    def endjob( self, evt, env ) :
        
        pass
