#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module dump_epics...
#
#------------------------------------------------------------------------

"""Pyana module which dumps EPICS data.

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
from pypdsdata import epics

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class dump_epics (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :

        pass

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :

        estore = env.epicsStore()
        print "====== PV names ======"
        for pv in sorted(estore.pvNames()):
            
            alias = estore.alias(pv)
            if alias is None:
                print "  PV name: %s" % (pv,)
            else:
                print "  PV name: %-24s alias: %s" % (pv, alias)

        print "====== PV values ======"
        for name in sorted(estore.names()):
            
            pv = estore.value(name)
            print "  name: %-30s iDbrType: %s  status: %d  severity: %d  units: '%s'" % (name, epics.dbr_text[pv.iDbrType], pv.status, pv.severity, pv.units),
            if pv.iNumElements > 1:
                print " values: %s" % (pv.values, )
            else:
                print " value: %s" % (pv.value, )
            print "    lower_alarm_limit: %s  upper_alarm_limit: %s" % (pv.lower_alarm_limit, pv.upper_alarm_limit,),
            print " lower_ctrl_limit: %s  upper_ctrl_limit: %s" % (pv.lower_ctrl_limit, pv.upper_ctrl_limit,)
            print "    lower_warning_limit: %s  upper_warning_limit: %s" % (pv.lower_warning_limit, pv.upper_warning_limit,),
            print " lower_disp_limit: %s  upper_disp_limit: %s" % (pv.lower_disp_limit, pv.upper_disp_limit,)

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        estore = env.epicsStore()
        print "====== PV values ======"
        for name in sorted(estore.names()):
            
            pv = estore.value(name)
            print "  name: %-30s status: %d  severity: %d" % (name, pv.status, pv.severity),
            if pv.iNumElements > 1:
                print " values: %s" % (pv.values,)
            else:
                print " value: %s" % (pv.value,)

    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param evt    event data object
        @param env    environment object
        """
        
        logging.info( "dump_epics.endjob() called" )
