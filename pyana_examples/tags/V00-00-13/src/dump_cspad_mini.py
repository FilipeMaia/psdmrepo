#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module dump_cspad_mini...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

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
class dump_cspad_mini (object) :
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
 
       logging.info( "dump_cspad_mini.beginjob() called" )


    def event( self, evt, env ) :

        elem = evt.get(xtc.TypeId.Type.Id_Cspad2x2Element, self.m_src)
        if not elem :
            print '*** cspad MiniElement information is missing ***'
            return

        # dump information about quadrants
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
        print data

    def endjob( self, env ) :

        logging.info( "dump_cspad_mini.endjob() called" )
