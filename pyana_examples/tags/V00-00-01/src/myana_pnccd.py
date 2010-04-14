#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module myana...
#
#------------------------------------------------------------------------

"""User analysis job for XTC data.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
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
from pdsdata import *

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class myana_pnccd ( object ) :
    """Example analysis module which dumps pnCCD info."""

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor. """

        self.count=0

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        self.configs = {}
        for x in evt.findXtc(typeId=xtc.TypeId.Type.Id_pnCCDconfig) :
            self.configs[x.src] = x.payload()

    def beginrun( self, evt, env ) :
        pass

    def event( self, evt, env ) :
        
        for x in evt.findXtc(typeId=xtc.TypeId.Type.Id_pnCCDframe) :
            
            cfg = self.configs[x.src]
            print "event %d: detInfo=%s numLinks=%d payloadSizePerLink=%d" % \
                    ( self.count, x.src, cfg.numLinks(), cfg.payloadSizePerLink() )
            
            frame = x.payload()
            for i in range(cfg.numLinks()) :
                print  "    link %d: shape=%s specialWord=%d frameNumber=%d timeStampHi=%d timeStampLo=%d sizeofData=%d" % \
                    (i, frame.data(cfg).shape, frame.specialWord(), frame.frameNumber(), 
                     frame.timeStampHi(), frame.timeStampLo(), frame.sizeofData(cfg) )
                frame = frame.next(cfg)
        
            
            # dump also combined object
            frame = evt.getPnCcdValue(x.src, env)
            print  "    combined: shape=%s specialWord=%d frameNumber=%d timeStampHi=%d timeStampLo=%d sizeofData=%d" % \
                (frame.data().shape, frame.specialWord(), frame.frameNumber(), 
                 frame.timeStampHi(), frame.timeStampLo(), frame.sizeofData() )
        
        
        self.count += 1
        
    def endrun( self, env ) :
        pass

    def endjob( self, env ) :
        pass
