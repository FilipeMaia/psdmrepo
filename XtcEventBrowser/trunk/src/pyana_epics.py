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

import numpy as np
import matplotlib.pyplot as plt

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
    def __init__ ( self, pv = "BEAM:LCLS:ELEC:Q",
                   plot_every_n = None,
                   plot_number = 1 ) :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param pv   Name(s) of the EPICS PV(s) to dump
        """

        opt = PyanaOptions()
        self.pv_names = opt.getOptStrings( pv )
        print "Init epics: "
        print self.pv_names

        # structure for storing the data
        # dictionary = { 'pvname': datalist[calibcycle] }
        self.pv_data_all = {}
        self.pv_data_calib = {}
        for pv_name in self.pv_names :
            self.pv_data_all[pv_name] = [] # one list for the whole job            
            self.pv_data_calib[pv_name] = None

        self.n_shot   = 0 # event number (resets every calibration cycle)
        self.n_calib  = 0 # calibration number (resets every run)
        self.n_run    = 0 # run number (resets every begin job / configure )
        self.n_config = 0 # configuration number





    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job.
        It will also be called in any Xtc Configure transition, if 
        configurations have changed since last it was run.

        @param evt    event data object
        @param env    environment object
        """

        # Preferred way to log information is via logging package
        self.n_run = 0
        self.n_config += 1
        logging.info( "pyana_epics.beginjob() called (%d)"%self.n_config )

        # Use environment object to access EPICS data
        for pv_name in self.pv_names :
            pv = env.epicsStore().value( pv_name )
            if not pv:
                logging.warning('EPICS PV %s does not exist', pv_name)
            else:
                pass
                ## Returned value should be of the type epics.EpicsPvCtrl.
                ## The code here demonstrates few members accessible for that type.
                ## For full list of members see Pyana Ref. Manual.
                # print "PV %s: id=%d type=%d size=%d status=%s severity=%s values=%s" % \
                #       (pv_name, pv.iPvId, pv.iDbrType, pv.iNumElements,
                #        pv.status, pv.severity, pv.values)
                             

            
    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        self.n_calib = 0
        self.n_run += 1
        logging.info( "pyana_epics.beginrun() called (%d)"%self.n_run )

        for pv_name in self.pv_names :
            pv = env.epicsStore().value( pv_name )
            if not pv:
                logging.warning('EPICS PV %s does not exist', pv_name)
            else:
                pass
                
    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """

        self.n_shot = 0
        self.n_calib += 1
        logging.info( "pyana_epics.begincalibcycle() called (%d)"%self.n_calib )


        for pv_name in self.pv_names :

            self.pv_data_calib[pv_name] = []

            pv = env.epicsStore().value( pv_name )
            if not pv:
                logging.warning('EPICS PV %s does not exist', pv_name)
            else:
                pass


    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        self.n_shot += 1
        logging.info( "pyana_epics.event() called (%d)"%self.n_shot )

        # Use environment object to access EPICS data
        for pv_name in self.pv_names :
            pv = env.epicsStore().value( pv_name )
            if not pv:
                logging.warning('EPICS PV %s does not exist', pv_name)
            else:

                ## Returned value should be of the type epics.EpicsPvTime.
                ## The code here demonstrates few members accessible for that type.
                ## For full list of members see Pyana Ref. Manual.
                # print "PV %s: id=%d type=%d size=%d status=%s severity=%s values=%s" % \
                #       (pv_name, pv.iPvId, pv.iDbrType, pv.iNumElements,
                #       pv.status, pv.severity, pv.values)

                # add pv value to the list
                if pv.iNumElements > 1 :
                    print pv_name, " has more than one element per event:"
                    print pv.values
                    exit(1)
                for values in pv.values :
                    self.pv_data_calib[pv_name].append( values )


    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        
        logging.info( "pyana_epics.endcalibcycle() called" )
        print "Report from calibcycle #%d"%self.n_calib
        for pv_name in self.pv_names :

            array = np.float_( self.pv_data_calib[pv_name] )

            self.pv_data_all[pv_name].append( array )


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

        for pv_name in self.pv_names :

            # make a 2d array
            array = np.array( self.pv_data_all[pv_name] )

            # for each calib cycle (axis=0), get the mean vaue of all events (axis=1)
            meanarray = np.sum(array,axis=1)
            print np.shape(meanarray)
            self.make_graph(meanarray)                            


    def make_histogram(self, array):
        pass
    
    def make_graph(self, array, fignum=600, suptitle=""):
        print "Make graph from array ", np.shape(array)

        fig = plt.figure(num=(fignum+10), figsize=(8,8) )
        fig.suptitle(suptitle)
        
        ax1 = fig.add_subplot(111)
        plt.plot(array)
        plt.title('')
        plt.xlabel('Shot#',horizontalalignment='left') # the other right
        plt.ylabel('Value',horizontalalignment='right')
        plt.show()
