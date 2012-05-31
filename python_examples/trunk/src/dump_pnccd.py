#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: dump_pnccd.py 2622 2011-11-11 14:35:00Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  Module myana...
#
#------------------------------------------------------------------------

"""User analysis job for XTC data.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: dump_pnccd.py 2622 2011-11-11 14:35:00Z salnikov@SLAC.STANFORD.EDU $

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
from pypdsdata import *

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class dump_pnccd ( object ) :
    """Example analysis module which dumps pnCCD info."""

    #--------------------
    #  Class variables --
    #--------------------

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

        config = env.getConfig(xtc.TypeId.Type.Id_pnCCDconfig, self.m_src)
        if config:
        
            print "%s: %s" % (config.__class__.__name__, self.m_src)
            print "  numLinks =", config.numLinks()
            print "  payloadSizePerLink =", config.payloadSizePerLink()
            
            try:
                #these methods exist only in V2
                print "  numChannels = %s" % config.numChannels()
                print "  numRows =", config.numRows()
                print "  numSubmoduleChannels =", config.numSubmoduleChannels()
                print "  numSubmoduleRows =", config.numSubmoduleRows()
                print "  numSubmodules =", config.numSubmodules()
                print "  camexMagic =", config.camexMagic()
                print "  info =", config.info()
                print "  timingFName =", config.timingFName()
            except:
                pass

    def beginrun( self, evt, env ) :
        pass

    def event( self, evt, env ) :
        
        frame = evt.get(xtc.TypeId.Type.Id_pnCCDframe, self.m_src)
        if frame:
            print "%s: %s" % (frame.__class__.__name__, self.m_src)
            print "  specialWord =", frame.specialWord()
            print "  frameNumber =", frame.frameNumber()
            print "  timeStampHi =", frame.timeStampHi()
            print "  timeStampLo =", frame.timeStampLo()
            data = frame.data()
            print "  data.shape =", data.shape
            print "  data.dtype =", data.dtype
            print "  data =", data
        
        
    def endrun( self, env ) :
        pass

    def endjob( self, env ) :
        pass
