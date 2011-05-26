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
import numpy as np
import matplotlib.pyplot as plt

from pypdsdata import xtc

from utilities import PyanaOptions
from utilities import EpicsData


#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class pyana_epics (object) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, pv_names = None,
                   plot_every_n  = None,
                   accumulate_n  = "0",
                   fignum = "1" ) :
        """Class constructor
        All parameters are passed as strings
        @param pv_names       List of name(s) of the EPICS PV(s) to plot
        @param plot_every_n   Frequency for plotting. If n=0, no plots till the end
        @param accumulate_n   Accumulate all (0) or reset the array every n shots
        @param fignum         Matplotlib figure number
        """

        opt = PyanaOptions()
        self.pv_names = opt.getOptStrings(pv_names)
        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.accumulate_n = opt.getOptInteger(accumulate_n)
        self.mpl_num = opt.getOptInteger(fignum)

        # other
        self.n_shots = None
        self.accu_start = None        

        # lists to fill numpy arrays
        self.initlists()
        
    def initlists(self):
        self.time = []
        self.shotnmb = []
        self.value = {}
        for pv_name in self.pv_names :
            self.value[pv_name] = []

    def resetlists(self):
        self.accu_start = self.n_shots
        del self.time[:]
        del self.shotnmb[:]
        for pv_name in self.pv_names :
            del self.value[pv_name][:]
                
        
    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        # Preferred way to log information is via logging package
        logging.info( "pyana_epics.beginjob() called")

        self.n_shots = 0
        self.accu_start = 0
        

        # Use environment object to access EPICS data
        for pv_name in self.pv_names :

            pv = env.epicsStore().value( pv_name )
            if not pv:
                logging.warning('EPICS PV %s does not exist', pv_name)
            else :

                # The returned value should be of the type epics.EpicsPvCtrl.
                print "PV %s: id=%d type=%d size=%d status=%s severity=%s values=%s" % \
                      (pv_name, pv.iPvId, pv.iDbrType, pv.iNumElements,
                       pv.status, pv.severity, pv.values)
        
    def event( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """

        self.n_shots += 1
        logging.info( "pyana_epics.event() called (%d)"%self.n_shots )

        # Use environment object to access EPICS data
        for pv_name in self.pv_names :
            pv = env.epicsStore().value( pv_name )
            if not pv:
                logging.warning('EPICS PV %s does not exist', pv_name)
            else:

                ## The returned value should be of the type epics.EpicsPvTime.
                self.value[pv_name].append( pv.values )


    def endjob( self, evt, env ) :
        """
        @param evt    optional
        @param env    environment object
        """
        
        logging.info( "pyana_epics.endjob() called" )


        for pv_name in self.pv_names :

            array = np.float_( self.value[pv_name] )
            print "For epics PV %s, array of shape"%pv_name, np.shape(array)
