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
from pypdsdata import xtc
from pypdsdata import imp

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
        self.m_src = self.configSrc('source', '*-*|Imp-*')
        
        self.n_samples = 0

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :

        config = env.getConfig(xtc.TypeId.Type.Id_ImpConfig, self.m_src)
        if config:
            print "dump_imp: %s: %s" % (config.__class__.__name__, self.m_src)
            print "  Range =", config.get(imp.ConfigV1.Registers.Range)
            print "  Cal_range =", config.get(imp.ConfigV1.Registers.Cal_range)
            print "  Reset =", config.get(imp.ConfigV1.Registers.Reset)
            print "  Bias_data =", config.get(imp.ConfigV1.Registers.Bias_data)
            print "  Cal_data =", config.get(imp.ConfigV1.Registers.Cal_data)
            print "  BiasDac_data =", config.get(imp.ConfigV1.Registers.BiasDac_data)
            print "  Cal_strobe =", config.get(imp.ConfigV1.Registers.Cal_strobe)
            print "  NumberOfSamples =", config.get(imp.ConfigV1.Registers.NumberOfSamples)
            print "  TrigDelay =", config.get(imp.ConfigV1.Registers.TrigDelay)
            print "  Adc_delay =", config.get(imp.ConfigV1.Registers.Adc_delay)

            self.n_samples = config.get(imp.ConfigV1.Registers.NumberOfSamples)

    def event( self, evt, env ) :

        data = evt.get(xtc.TypeId.Type.Id_ImpData, self.m_src)
        if not data:
            return

        print "dump_imp: %s: %s" % (data.__class__.__name__, self.m_src)

        print "  vc =", data.vc()
        print "  lane =", data.lane()
        print "  frameNumber =", data.frameNumber()
        print "  range =", data.range()
        print "  laneStatus =", data.laneStatus()
        
        for i in range(self.n_samples):
            print "  sample[%d]: channels = %s" % (i, data.sample(i).channels())

    def endjob( self, evt, env ) :
        pass
