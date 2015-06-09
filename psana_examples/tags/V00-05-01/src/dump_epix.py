#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Psana user analysis module dump_epix...
#
#------------------------------------------------------------------------

"""Example psana module to dump epix data

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
class dump_epix (object) :
    '''Class whose instance will be used as a user analysis module.'''

    #----------------
    #  Constructor --
    #----------------
    def __init__(self):
        self.m_src = self.configSrc('source', ':Epix')

    #-------------------
    #  Public methods --
    #-------------------

    def beginRun( self, evt, env ) :
        config = env.configStore().get(Epix.Config, self.m_src)
        if config:
            print "dump_epix: %s: %s" % (config.__class__.__name__, self.m_src)
            print "  version =", config.version()
            print "  runTrigDelay =", config.runTrigDelay()
            print "  daqTrigDelay =", config.daqTrigDelay()
            print "  dacSetting =", config.dacSetting()
            print "  asicGR =", config.asicGR()
            print "  asicAcq =", config.asicAcq()
            print "  asicR0 =", config.asicR0()
            print "  asicPpmat =", config.asicPpmat()
            print "  asicPpbe =", config.asicPpbe()
            print "  asicRoClk =", config.asicRoClk()
            print "  asicGRControl =", config.asicGRControl()
            print "  asicAcqControl =", config.asicAcqControl()
            print "  asicR0Control =", config.asicR0Control()
            print "  asicPpmatControl =", config.asicPpmatControl()
            print "  asicPpbeControl =", config.asicPpbeControl()
            print "  asicR0ClkControl =", config.asicR0ClkControl()
            print "  prepulseR0En =", config.prepulseR0En()
            print "  adcStreamMode =", config.adcStreamMode()
            print "  testPatternEnable =", config.testPatternEnable()
            print "  acqToAsicR0Delay =", config.acqToAsicR0Delay()
            print "  asicR0ToAsicAcq =", config.asicR0ToAsicAcq()
            print "  asicAcqWidth =", config.asicAcqWidth()
            print "  asicAcqLToPPmatL =", config.asicAcqLToPPmatL()
            print "  asicRoClkHalfT =", config.asicRoClkHalfT()
            print "  adcReadsPerPixel =", config.adcReadsPerPixel()
            print "  adcClkHalfT =", config.adcClkHalfT()
            print "  asicR0Width =", config.asicR0Width()
            print "  adcPipelineDelay =", config.adcPipelineDelay()
            print "  prepulseR0Width =", config.prepulseR0Width()
            print "  prepulseR0Delay =", config.prepulseR0Delay()
            print "  digitalCardId0 =", config.digitalCardId0()
            print "  digitalCardId1 =", config.digitalCardId1()
            print "  analogCardId0 =", config.analogCardId0()
            print "  analogCardId1 =", config.analogCardId1()
            print "  lastRowExclusions =", config.lastRowExclusions()
            print "  numberOfAsicsPerRow =", config.numberOfAsicsPerRow()
            print "  numberOfAsicsPerColumn =", config.numberOfAsicsPerColumn()
            print "  numberOfRowsPerAsic =", config.numberOfRowsPerAsic()
            print "  numberOfPixelsPerAsicRow =", config.numberOfPixelsPerAsicRow()
            print "  baseClockFrequency =", config.baseClockFrequency()
            print "  asicMask =", config.asicMask()
            print "  asicPixelTestArray =", config.asicPixelTestArray()
            print "  asicPixelMaskArray =", config.asicPixelMaskArray()
            print "  numberOfRows =", config.numberOfRows()
            print "  numberOfColumns =", config.numberOfColumns()
            print "  numberOfAsics =", config.numberOfAsics()
            
            for i in range(config.numberOfAsics()):
                aconfig = config.asics(i)
                print "    Epix.AsicConfigV1 #%d" % i
                print "      monostPulser =", aconfig.monostPulser()
                print "      dummyTest =", aconfig.dummyTest()
                print "      dummyMask =", aconfig.dummyMask()
                print "      pulser =", aconfig.pulser()
                print "      pbit =", aconfig.pbit()
                print "      atest =", aconfig.atest()
                print "      test =", aconfig.test()
                print "      sabTest =", aconfig.sabTest()
                print "      hrTest =", aconfig.hrTest()
                print "      digMon1 =", aconfig.digMon1()
                print "      digMon2 =", aconfig.digMon2()
                print "      pulserDac =", aconfig.pulserDac()
                print "      Dm1En =", aconfig.Dm1En()
                print "      Dm2En =", aconfig.Dm2En()
                print "      slvdSBit =", aconfig.slvdSBit()
                print "      VRefDac =", aconfig.VRefDac()
                print "      TpsTComp =", aconfig.TpsTComp()
                print "      TpsMux =", aconfig.TpsMux()
                print "      RoMonost =", aconfig.RoMonost()
                print "      TpsGr =", aconfig.TpsGr()
                print "      S2dGr =", aconfig.S2dGr()
                print "      PpOcbS2d =", aconfig.PpOcbS2d()
                print "      Ocb =", aconfig.Ocb()
                print "      Monost =", aconfig.Monost()
                print "      FastppEnable =", aconfig.FastppEnable()
                print "      Preamp =", aconfig.Preamp()
                print "      PixelCb =", aconfig.PixelCb()
                print "      S2dTComp =", aconfig.S2dTComp()
                print "      FilterDac =", aconfig.FilterDac()
                print "      TC =", aconfig.TC()
                print "      S2d =", aconfig.S2d()
                print "      S2dDacBias =", aconfig.S2dDacBias()
                print "      TpsTcDac =", aconfig.TpsTcDac()
                print "      TpsDac =", aconfig.TpsDac()
                print "      S2dTcDac =", aconfig.S2dTcDac()
                print "      S2dDac =", aconfig.S2dDac()
                print "      TestBe =", aconfig.TestBe()
                print "      IsEn =", aconfig.IsEn()
                print "      DelExec =", aconfig.DelExec()
                print "      DelCckReg =", aconfig.DelCckReg()
                print "      RowStartAddr =", aconfig.RowStartAddr()
                print "      RowStopAddr =", aconfig.RowStopAddr()
                print "      ColStartAddr =", aconfig.ColStartAddr()
                print "      ColStopAddr =", aconfig.ColStopAddr()
                print "      chipID =", aconfig.chipID()

    def event( self, evt, env ) :

        data = evt.get(Epix.Element, self.m_src)
        if not data:
            return

        print "dump_epix: %s: %s" % (data.__class__.__name__, self.m_src)
        print "  vc =", data.vc()
        print "  lane =", data.lane()
        print "  acqCount =", data.acqCount()
        print "  frameNumber =", data.frameNumber()
        print "  ticks =", data.ticks()
        print "  fiducials =", data.fiducials()
        print "  frame =", data.frame()
        print "  excludedRows =", data.excludedRows()
        print "  temperatures =", data.temperatures()
        print "  lastWord =", data.lastWord()
