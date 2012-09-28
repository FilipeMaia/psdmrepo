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
class myana_control_config ( object ) :
    """Example analysis module which dumps ControlConfig object info."""

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor. """
        pass

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        
        pass

    def begincalibcycle( self, evt, env ) :

        # PV control info is present in both Configure and BeginCalibCycle
        # transition so it should be available in beginjob() and 
        # begincalibcycle(), but we are interested in calibcycle values mostly.
        # The same information is available in even() as usual.
        
        print "begincalibcycle() run=%s time=%s" % (evt.run(), evt.getTime()) 

        # control data is supposed to be in environment
        control = env.getConfig(TypeId.Type.Id_ControlConfig)
        if not control :
            print '  Control information is missing'
        else :
            
            # dump pv controls
            for i in range(control.npvControls()) :
                pv = control.pvControl(i)
                print "  Control PV %d: %s [%d] = %f" % (i, pv.name(), pv.index(), pv.value())

            # dump pv monitors
            for i in range(control.npvMonitors()) :
                pv = control.pvMonitor(i)
                print "  Monitor PV %d: %s [%d] High Limit %f  Low Limit %f" % \
                    (i, pv.name(), pv.index(), pv.loValue(), pv.hiValue())
                    
            try:
                # this can only work with V2
                for i in range(control.npvLabels()) :
                    lbl = control.pvLabel(i)
                    print "  PV Label %d: %s -> %s" % (i, lbl.name(), lbl.value())
            except:
                pass
        

    def event( self, evt, env ) :

        # nothing to do here, but event() must be present
        pass
        
    def endjob( self, env ) :
        pass

        