#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module dump_princeton...
#
#------------------------------------------------------------------------

"""Example module for accessing SharedIpimb data.

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

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class dump_princeton (object) :
    """Class whose instance will be used as a user analysis module. """

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
        
        config = env.getPrincetonConfig(self.m_src)
        if config:
        
            print "%s: %s" % (config.__class__.__name__, self.m_src)
            
            print "  width =", config.width();
            print "  height =", config.height();
            print "  orgX =", config.orgX();
            print "  orgY =", config.orgY();
            print "  binX =", config.binX();
            print "  binY =", config.binY();
            print "  exposureTime =", config.exposureTime();
            print "  coolingTemp =", config.coolingTemp();
            print "  readoutSpeedIndex =", config.readoutSpeedIndex();
            print "  readoutEventCode =", config.readoutEventCode();
            print "  delayMode =", config.delayMode();
            print "  frameSize =", config.frameSize();

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        frame = evt.get(xtc.TypeId.Type.Id_PrincetonFrame, self.m_src)
        if frame:
            print "%s: %s" % (frame.__class__.__name__, self.m_src)
            
            print "  shotIdStart =", frame.shotIdStart()
            print "  readoutTime =", frame.readoutTime()
            
            img = frame.data()
            print "  data.shape =", img.shape
            print "  data =", img


    def endjob( self, evt, env ) :
        
        pass

