#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $
#
# Description:
#  Pyana user analysis module pyana_epics...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $

@author Ingrid Ofte
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 1095 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------
from utilities import PyanaOptions

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class pyana_epics (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, pv = "BEAM:LCLS:ELEC:Q") :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param pv   Name(s) of the EPICS PV(s) to dump
        """

        opt = PyanaOptions()
        self.m_pvs = opt.getOptStrings( pv )
        print "Init epics: "
        print self.m_pvs


    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job. It should
        do a one-time initialization possible extracting values from event
        data (which is a Configure object) or environment.

        @param evt    event data object
        @param env    environment object
        """

        # Preferred way to log information is via logging package
        logging.info( "pyana_epics.beginjob() called" )

        # Use environment object to access EPICS data
        for m_pv in self.m_pvs :
            pv = env.epicsStore().value( m_pv )
            if not pv:
                logging.warning('EPICS PV %s does not exist', m_pv)
            else:

                # Returned value should be of the type epics.EpicsPvCtrl.
                # The code here demonstrates few members accessible for that type.
                # For full list of members see Pyana Ref. Manual.
                print "PV %s: id=%d type=%d size=%d status=%s severity=%s values=%s" % \
                      (m_pv, pv.iPvId, pv.iDbrType, pv.iNumElements,
                       pv.status, pv.severity, pv.values)
            
            
    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "pyana_epics.beginrun() called" )

    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "pyana_epics.begincalibcycle() called" )

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        # Use environment object to access EPICS data
        for m_pv in self.m_pvs :
            pv = env.epicsStore().value( m_pv )
            if not pv:
                logging.warning('EPICS PV %s does not exist', m_pv)
            else:

                # Returned value should be of the type epics.EpicsPvTime.
                # The code here demonstrates few members accessible for that type.
                # For full list of members see Pyana Ref. Manual.
                print "PV %s: id=%d type=%d size=%d status=%s severity=%s values=%s" % \
                      (m_pv, pv.iPvId, pv.iDbrType, pv.iNumElements,
                       pv.status, pv.severity, pv.values)

                                                                                    

    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        
        logging.info( "pyana_epics.endcalibcycle() called" )

    def endrun( self, env ) :
        """This optional method is called if present at the end of the run.
        
        @param env    environment object
        """
        
        logging.info( "pyana_epics.endrun() called" )

    def endjob( self, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        
        logging.info( "pyana_epics.endjob() called" )
