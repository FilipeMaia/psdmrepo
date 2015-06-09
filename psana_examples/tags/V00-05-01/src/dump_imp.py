#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Psana user analysis module dump_imp...
#
#------------------------------------------------------------------------

"""Example module for psana which dumps Imp data.

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
class dump_imp (object) :
    '''Class whose instance will be used as a user analysis module.'''


    #----------------
    #  Constructor --
    #----------------
    def __init__(self):
        self.m_src = self.configSrc('source', ':Imp')

    #-------------------
    #  Public methods --
    #-------------------

    def beginRun( self, evt, env ) :

        config = env.configStore().get(Imp.Config, self.m_src)
        if config:
            print "dump_imp: %s: %s" % (config.__class__.__name__, self.m_src)
            print "  range =", config.range()
            print "  calRange =", config.calRange()
            print "  reset =", config.reset()
            print "  biasData =", config.biasData()
            print "  calData =", config.calData()
            print "  biasDacData =", config.biasDacData()
            print "  calStrobe =", config.calStrobe()
            print "  numberOfSamples =", config.numberOfSamples()
            print "  trigDelay =", config.trigDelay()
            print "  adcDelay =", config.adcDelay()

    def event( self, evt, env ) :

        data = evt.get(Imp.Element, self.m_src)
        if not data:
            return

        print "dump_imp: %s: %s" % (data.__class__.__name__, self.m_src)

        print "  vc =", data.vc()
        print "  lane =", data.lane()
        print "  frameNumber =", data.frameNumber()
        print "  range =", data.range()

        laneStatus = data.laneStatus()
        print "  laneStatus.linkErrCount =", laneStatus.linkErrCount()
        print "  laneStatus.linkDownCount =", laneStatus.linkDownCount()
        print "  laneStatus.cellErrCount =", laneStatus.cellErrCount()
        print "  laneStatus.rxCount =", laneStatus.rxCount()
        print "  laneStatus.locLinked =", laneStatus.locLinked()
        print "  laneStatus.remLinked =", laneStatus.remLinked()
        print "  laneStatus.zeros =", laneStatus.zeros()
        print "  laneStatus.powersOkay =", laneStatus.powersOkay()
        
        for i, sample in enumerate(data.samples()):
            print "  sample[%d]: channels = %s" % (i, sample.channels())

