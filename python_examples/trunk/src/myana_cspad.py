#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: myana_cspad.py 1253 2010-10-14 22:27:28Z salnikov $
#
# Description:
#  Module myana...
#
#------------------------------------------------------------------------

"""User analysis job for XTC data.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: myana_cspad.py 1253 2010-10-14 22:27:28Z salnikov $

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 1253 $"
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
from pypdsdata.xtc import *

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class myana_cspad ( object ) :
    """Example analysis module which dumps ControlConfig object info."""

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, address = None ) :
        """Constructor. """
        self.address = address

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :

        config = env.getConfig(TypeId.Type.Id_CspadConfig, self.address)
        if not config:
            print '*** cspad config object is missing ***'
            return
        
        quads = range(4)
        
        print "Cspad configuration"
        print "  N quadrants   : %d" % config.numQuads()
        print "  Quad mask     : %#x" % config.quadMask()
        print "  payloadSize   : %d" % config.payloadSize()
        print "  badAsicMask0  : %#x" % config.badAsicMask0()
        print "  badAsicMask1  : %#x" % config.badAsicMask1()
        print "  asicMask      : %#x" % config.asicMask()
        print "  numAsicsRead  : %d" % config.numAsicsRead()
        try:
            # older versions may not have all methods
            print "  roiMask       : [%s]" % ', '.join([hex(config.roiMask(q)) for q in quads])
            print "  numAsicsStored: %s" % str(map(config.numAsicsStored, quads))
        except:
            pass
        print "  sections      : %s" % str(map(config.sections, quads))


    def event( self, evt, env ) :

        quads = evt.getCsPadQuads(self.address, env)
        if not quads :
            print '*** cspad information is missing ***'
            return

        # dump information about quadrants
        print "Number of quadrants: %d" % len(quads)
        for q in quads:
            
            print "  Quadrant %d" % q.quad()
            print "    virtual_channel: %s" % q.virtual_channel()
            print "    lane: %s" % q.lane()
            print "    tid: %s" % q.tid()
            print "    acq_count: %s" % q.acq_count()
            print "    op_code: %s" % q.op_code()
            print "    seq_count: %s" % q.seq_count()
            print "    ticks: %s" % q.ticks()
            print "    fiducials: %s" % q.fiducials()
            print "    frame_type: %s" % q.frame_type()
            print "    sb_temp: %s" % map(q.sb_temp, range(4))

            # image data as 3-dimentional array
            data = q.data()
            print "    Data shape: %s" % str(data.shape)
            
        
    def endjob( self, env ) :
        pass

        