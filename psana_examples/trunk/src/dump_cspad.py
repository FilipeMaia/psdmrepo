#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: dump_cspad.py 2622 2011-11-11 14:35:00Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  Module myana...
#
#------------------------------------------------------------------------

"""User analysis job for XTC data.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: dump_cspad.py 2622 2011-11-11 14:35:00Z salnikov@SLAC.STANFORD.EDU $

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 2622 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psana import *

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class dump_cspad ( object ) :
    """Example analysis module which dumps ControlConfig object info."""

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor. """
        
        self.m_src = self.configSrc('source', "")

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :

        config = env.configStore().get(CsPad.Config, self.m_src)
        if not config:
            return
        
        print "dump_cspad: %s: %s" % (config.__class__.__name__, self.m_src)
        print "  concentratorVersion =", config.concentratorVersion();
        print "  runDelay =", config.runDelay();
        print "  eventCode =", config.eventCode();
        print "  inactiveRunMode =", config.inactiveRunMode();
        print "  activeRunMode =", config.activeRunMode();
        print "  tdi =", config.tdi();
        print "  payloadSize =", config.payloadSize();
        print "  badAsicMask0 =", config.badAsicMask0();
        print "  badAsicMask1 =", config.badAsicMask1();
        print "  asicMask =", config.asicMask();
        print "  quadMask =", config.quadMask();
        print "  numAsicsRead =", config.numAsicsRead();
        print "  numQuads =", config.numQuads();
        try:
            # older versions may not have all methods
            print "  roiMask = [%s]" % ', '.join([hex(config.roiMask(q)) for q in range(4)])
            print "  numAsicsStored = %s" % str(map(config.numAsicsStored, range(4)))
        except:
            pass
        try:
            print "  sections = %s" % str(map(config.sections, range(4)))
        except:
            pass


    def event( self, evt, env ) :

        data = evt.get(CsPad.Data, self.m_src)
        if not data:
            return
        
        nQuads = data.quads_shape()[0]

        # dump information about quadrants
        print "dump_cspad: %s: %s" % (data.quads(0).__class__.__name__, self.m_src)
        print "  Number of quadrants: %d" % nQuads
        for i in range(nQuads):
            q = data.quads(i)
            
            print "  Quadrant #%d" % q.quad()
            print "    virtual_channel = %s" % q.virtual_channel()
            print "    lane = %s" % q.lane()
            print "    tid = %s" % q.tid()
            print "    acq_count = %s" % q.acq_count()
            print "    op_code = %s" % q.op_code()
            print "    quad = %s" % q.quad()
            print "    seq_count = %s" % q.seq_count()
            print "    ticks = %s" % q.ticks()
            print "    fiducials = %s" % q.fiducials()
            print "    frame_type = %s" % q.frame_type()
            print "    sb_temp = %s" % q.sb_temp()
            print "    common_mode = %s" % [q.common_mode(i) for i in range(q.data().shape[0])]
            # image data as 3-dimentional array
            print "    data shape = {}".format(q.data().shape)
            print "    data = %s" % q.data()
