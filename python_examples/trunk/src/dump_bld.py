#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: dump_bld.py 3140 2012-03-28 00:29:50Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  Pyana user analysis module dump_timepix...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: dump_bld.py 3140 2012-03-28 00:29:50Z salnikov@SLAC.STANFORD.EDU $

@author Andy Salnikov
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 3140 $"
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
from pypdsdata import bld

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#---------------------
#  Class definition --
#---------------------
class dump_bld (object) :
    """example analysis module which dumps BLD data"""

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
        
        self.event(evt, env)

    def event( self, evt, env ) :

        # dump BldDataFEEGasDetEnergy
        data = evt.get(xtc.TypeId.Type.Id_FEEGasDetEnergy, self.m_src)
        if data:
            print "%s: %s" % (data.__class__.__name__, self.m_src)            
            print "  f_11_ENRC =", data.f_11_ENRC
            print "  f_12_ENRC =", data.f_12_ENRC
            print "  f_21_ENRC =", data.f_21_ENRC
            print "  f_22_ENRC =", data.f_22_ENRC
            
        # dump BldDataEBeamV* objects
        data = evt.get(xtc.TypeId.Type.Id_EBeam, self.m_src)
        if data:
            print "%s: %s" % (data.__class__.__name__, self.m_src)            
            print "  uDamageMask =", data.uDamageMask
            print "  fEbeamCharge =", data.fEbeamCharge
            print "  fEbeamL3Energy =", data.fEbeamL3Energy
            print "  fEbeamLTUPosX =", data.fEbeamLTUPosX
            print "  fEbeamLTUPosY =", data.fEbeamLTUPosY
            print "  fEbeamLTUAngX =", data.fEbeamLTUAngX
            print "  fEbeamLTUAngY =", data.fEbeamLTUAngY
            if type(data) == bld.BldDataEBeamV1:
                # this only exists in V1
                print "  fEbeamPkCurrBC2 =", data.fEbeamPkCurrBC2
            if type(data) == bld.BldDataEBeamV2:
                # this only exists in V2
                print "  fEbeamPkCurrBC2 =", data.fEbeamPkCurrBC2
                print "  fEbeamEnergyBC2 =", data.fEbeamEnergyBC2
            if type(data) == bld.BldDataEBeamV3:
                # this only exists in V3
                print "  fEbeamPkCurrBC2 =", data.fEbeamPkCurrBC2
                print "  fEbeamEnergyBC2 =", data.fEbeamEnergyBC2
                print "  fEbeamPkCurrBC1 =", data.fEbeamPkCurrBC1
                print "  fEbeamEnergyBC1 =", data.fEbeamEnergyBC1

        # dump BldDataPhaseCavity
        data = evt.get(xtc.TypeId.Type.Id_PhaseCavity, self.m_src)
        if data:
            print "%s: %s" % (data.__class__.__name__, self.m_src)            
            print "  fFitTime1 =", data.fFitTime1
            print "  fFitTime2 =", data.fFitTime2
            print "  fCharge1 =", data.fCharge1
            print "  fCharge2 =", data.fCharge2
            

    def endjob( self, evt, env ) :
        
        pass
