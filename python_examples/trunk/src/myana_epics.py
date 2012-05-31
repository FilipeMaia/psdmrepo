#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: myana_epics.py 1419 2011-01-23 22:37:04Z salnikov $
#
# Description:
#  Pyana user analysis module myana_epics...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: myana_epics.py 1419 2011-01-23 22:37:04Z salnikov $

@author Andrei Salnikov
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 1419 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class myana_epics (object) :
    """Example analysis module which accesses EPICS data. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, pv = "BEAM:LCLS:ELEC:Q") :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param pv   Name of the EPICS PV to dump
        """
        self.m_pv = pv

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :

        # Preferred way to log information is via logging package
        logging.info( "myana_epics.beginjob() called" )

        # Use environment object to access EPICS data        
        pv = env.epicsStore().value(self.m_pv)
        if not pv:
            logging.warning('EPICS PV %s does not exist', self.m_pv)
        else:
            
            # Returned value should be of the type epics.EpicsPvCtrl.
            # The code here demonstrates few members accessible for that type.
            # For full list of members see Pyana Ref. Manual.
            print "PV %s: id=%d type=%d size=%d status=%s severity=%s values=%s" % \
                (self.m_pv, pv.iPvId, pv.iDbrType, pv.iNumElements, 
                 pv.status, pv.severity, pv.values)

    def event( self, evt, env ) :

        # Use environment object to access EPICS data        
        pv = env.epicsStore().value(self.m_pv)
        if not pv:
            logging.warning('EPICS PV %s does not exist', self.m_pv)
        else:
            
            # Returned value should be of the type epics.EpicsPvTime.
            # The code here demonstrates few members accessible for that type.
            # For full list of members see Pyana Ref. Manual.
            print "PV %s: id=%d type=%d size=%d status=%s severity=%s values=%s stamp=%s" % \
                (self.m_pv, pv.iPvId, pv.iDbrType, pv.iNumElements, 
                 pv.status, pv.severity, pv.values, pv.stamp)

    def endjob( self, env ) :
        pass
