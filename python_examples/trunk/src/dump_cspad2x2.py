#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: dump_cspad2x2.py 2862 2012-02-07 00:54:34Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  Pyana user analysis module dump_cspad2x2...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: dump_cspad2x2.py 2862 2012-02-07 00:54:34Z salnikov@SLAC.STANFORD.EDU $

@author Andy Salnikov
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 2862 $"
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
class dump_cspad2x2 (object) :
    """Class whose instance will be used as a user analysis module. """

    #--------------------
    #  Class variables --
    #--------------------
    
    # usual convention is to prefix static variables with s_
    s_staticVariable = 0

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
 
        config = env.getConfig(xtc.TypeId.Type.Id_Cspad2x2Config, self.m_src)
        if not config:
            return
        
        print "dump_cspad2x2: %s: %s" % (config.__class__.__name__, self.m_src)
        print "  tdi =", config.tdi()
        prot = config.protectionThreshold()
        print "  protectionThreshold: adcThreshold =", prot.adcThreshold, "pixelCountThreshold =", prot.pixelCountThreshold 
        print "  protectionEnable =", config.protectionEnable()
        print "  inactiveRunMode =", config.inactiveRunMode()
        print "  activeRunMode =", config.activeRunMode()
        print "  payloadSize =", config.payloadSize()
        print "  badAsicMask =", config.badAsicMask()
        print "  asicMask =", config.asicMask()
        print "  roiMask =", config.roiMask()
        print "  numAsicsRead =", config.numAsicsRead()
        print "  numAsicsStored =", config.numAsicsStored()
        print "  concentratorVersion =", config.concentratorVersion()
        quad = config.quad()
        print "  quad:"
        print "    shiftSelect =", quad.shiftSelect()
        print "    edgeSelect =", quad.edgeSelect()
        print "    readClkSet =", quad.readClkSet()
        print "    readClkHold =", quad.readClkHold()
        print "    dataMode =", quad.dataMode()
        print "    prstSel =", quad.prstSel()
        print "    acqDelay =", quad.acqDelay()
        print "    intTime =", quad.intTime()
        print "    digDelay =", quad.digDelay()
        print "    ampIdle =", quad.ampIdle()
        print "    injTotal =", quad.injTotal()
        print "    rowColShiftPer =", quad.rowColShiftPer()
        print "    ampReset =", quad.ampReset()
        print "    digCount =", quad.digCount()
        print "    digPeriod =", quad.digPeriod()
        print "    PeltierEnable =", quad.PeltierEnable()
        print "    kpConstant =", quad.kpConstant()
        print "    kiConstant =", quad.kiConstant()
        print "    kdConstant =", quad.kdConstant()
        print "    humidThold =", quad.humidThold()
        print "    setPoint =", quad.setPoint()
        print "    ro =", quad.ro()
        print "    dp =", quad.dp()
        print "    gm =", quad.gm()


    def event( self, evt, env ) :

        elem = evt.get(xtc.TypeId.Type.Id_Cspad2x2Element, self.m_src)
        if not elem :
            print '*** cspad MiniElement information is missing ***'
            return

        # dump information about quadrants
        print "dump_cspad2x2: %s: %s" % (elem.__class__.__name__, self.m_src)
        print "    quadrant: %d" % elem.quad()
        print "    virtual_channel: %s" % elem.virtual_channel()
        print "    lane: %s" % elem.lane()
        print "    tid: %s" % elem.tid()
        print "    acq_count: %s" % elem.acq_count()
        print "    op_code: %s" % elem.op_code()
        print "    seq_count: %s" % elem.seq_count()
        print "    ticks: %s" % elem.ticks()
        print "    fiducials: %s" % elem.fiducials()
        print "    frame_type: %s" % elem.frame_type()
        print "    sb_temp: %s" % map(elem.sb_temp, range(4))

        # image data as 3-dimentional array
        data = elem.data()
        print "    Data shape: %s" % str(data.shape)
        print "    Data:", data

    def endjob( self, env ) :

        logging.info( "dump_cspad2x2.endjob() called" )
