#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module dump_shared_ipimb...
#
#------------------------------------------------------------------------

"""Example module for accessing SharedIpimb data.

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

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class dump_shared_ipimb (object) :
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
        
        pass

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        data = evt.get(xtc.TypeId.Type.Id_SharedIpimb, self.m_src)
        if data:
            print "%s: %s" % (data.__class__.__name__, self.m_src)
            
            ipimbData = data.ipimbData
            print "  Ipimb.Data:"
            print "    triggerCounter =", ipimbData.triggerCounter()
            print "    config =", ipimbData.config0(), ipimbData.config1(), ipimbData.config2()
            print "    channel =", ipimbData.channel0(), ipimbData.channel1(), \
                ipimbData.channel2(), ipimbData.channel3()
            print "    volts =", ipimbData.channel0Volts(), ipimbData.channel1Volts(), \
                ipimbData.channel2Volts(), ipimbData.channel3Volts()

            ipimbConfig = data.ipimbConfig
            print "  Ipimb.Config:";
            print "    triggerCounter =", ipimbConfig.triggerCounter()
            print "    serialID =", ipimbConfig.serialID()
            print "    chargeAmpRange =", ipimbConfig.chargeAmpRange()
            print "    calibrationRange =", ipimbConfig.calibrationRange()
            print "    resetLength =", ipimbConfig.resetLength()
            print "    resetDelay =", ipimbConfig.resetDelay()
            print "    chargeAmpRefVoltage =", ipimbConfig.chargeAmpRefVoltage()
            print "    calibrationVoltage =", ipimbConfig.calibrationVoltage()
            print "    diodeBias =", ipimbConfig.diodeBias()
            print "    status =", ipimbConfig.status()
            print "    errors =", ipimbConfig.errors()
            print "    calStrobeLength =", ipimbConfig.calStrobeLength()
            print "    trigDelay =", ipimbConfig.trigDelay()
            
            ipmFexData = data.ipmFexData
            print "  Lusi.IpmFex:";
            print "    sum =", ipmFexData.sum
            print "    xpos =", ipmFexData.xpos
            print "    ypos =", ipmFexData.ypos
            print "    channel =", ipmFexData.channel


    def endjob( self, evt, env ) :
        
        pass

