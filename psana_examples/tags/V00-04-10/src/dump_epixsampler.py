#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Psana user analysis module dump_epixsampler...
#
#------------------------------------------------------------------------

"""Example psana module to dump epixsampler data

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

#---------------------
#  Class definition --
#---------------------
class dump_epixsampler (object) :
    '''Class whose instance will be used as a user analysis module.'''

    #----------------
    #  Constructor --
    #----------------
    def __init__(self):
        self.m_src = self.configSrc('source', ':EpixSampler')

    #-------------------
    #  Public methods --
    #-------------------

    def beginRun( self, evt, env ) :
        config = env.configStore().get(EpixSampler.Config, self.m_src)
        if config:
            print "dump_epixsampler: %s: %s" % (config.__class__.__name__, self.m_src)
            print "  version =", config.version()
            print "  runTrigDelay =", config.runTrigDelay()
            print "  daqTrigDelay =", config.daqTrigDelay()
            print "  daqSetting =", config.daqSetting()
            print "  adcClkHalfT =", config.adcClkHalfT()
            print "  adcPipelineDelay =", config.adcPipelineDelay()
            print "  digitalCardId0 =", config.digitalCardId0()
            print "  digitalCardId1 =", config.digitalCardId1()
            print "  analogCardId0 =", config.analogCardId0()
            print "  analogCardId1 =", config.analogCardId1()
            print "  numberOfChannels =", config.numberOfChannels()
            print "  samplesPerChannel =", config.samplesPerChannel()
            print "  baseClockFrequency =", config.baseClockFrequency()
            print "  testPatternEnable =", config.testPatternEnable()

    def event( self, evt, env ) :

        data = evt.get(EpixSampler.Element, self.m_src)
        if not data:
            return

        print "dump_epixsampler: %s: %s" % (data.__class__.__name__, self.m_src)
        print "  vc =", data.vc()
        print "  lane =", data.lane()
        print "  acqCount =", data.acqCount()
        print "  frameNumber =", data.frameNumber()
        print "  ticks =", data.ticks()
        print "  fiducials =", data.fiducials()
        print "  temperatures =", data.temperatures()
        print "  frame =", data.frame()
