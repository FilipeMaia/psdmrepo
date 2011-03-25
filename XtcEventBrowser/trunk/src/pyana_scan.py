#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $
#
# Description:
#  Pyana user analysis module pyana_scan...
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


#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class pyana_scan (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self,
                   input_epics = "BEAM:LCLS:ELEC:Q",
                   input_scalars = None,
                   plot_every_n = None,
                   fignum = "1" ) :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param input_epics      Name(s) of other scalars to correlate in scan
        @param input_scalars    Name(s) of other scalars to correlate in scan
        @param plot_every_n     Frequency for plotting. If n=0, no plots till the end
        @param fignum           Matplotlib figure number
        """
        opt = PyanaOptions()
        self.input_epics = opt.getOptStrings(input_epics)
        self.input_scalars = opt.getOptStrings(input_scalars)
        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.mpl_num = opt.getOptInteger(fignum)
        
        # count number of begin/end job (configure transitions)
        self.n_jobs = 0 

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called at an xtc Configure transition
        Assume only one Configure per job.
        Typically you should process only one run per job. 

        @param evt    event data object
        @param env    environment object
        """

        # data counters
        self.n_jobs += 1 # number of jobs / configurations
        self.n_runs =  0 # number of runs in this job             
            
    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """
        self.n_runs += 1

        # initialize calibcycle data
        self.n_ccls = 0
        self.ccls_nevts  = []
        self.ccls_ctrl = {}   # to hold ControlPV names and values
        self.ccls_scalars = {} # to hold epics and other scalar mean and std

        print "Processing run number ", evt.run()
        logging.info( "pyana_scan.beginrun() called (%d)"%self.n_runs )

                
    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """
        self.n_ccls += 1

        # initialize event data
        self.n_evts = 0
        self.evts_scalars = {}
        print "Begin calibcycle ", self.n_ccls

        # control.ConfigV1 element
        ctrl_config = env.getConfig(xtc.TypeId.Type.Id_ControlConfig)

        nControls = ctrl_config.npvControls()
        for ic in range (0, nControls ):
            #
            cpv = ctrl_config.pvControl(ic)
            name = cpv.name()
            value = cpv.value()
            
            if name not in self.ccls_ctrl.keys() :
                self.ccls_ctrl[name] = []

            # store the value
            self.ccls_ctrl[name].append(value)


    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.
        @param evt    event data object
        @param env    environment object
        """
        self.n_evts += 1
        logging.info( "pyana_scan.event() called (%d)"%self.n_evts )

        # Use environment object to access EPICS data
        for epv_name in self.input_epics :

            if epv_name not in self.evts_scalars.keys() :
                self.evts_scalars[epv_name] = []

            # store the value
            epv = env.epicsStore().value(epv_name).value
            if not epv:
                logging.warning('EPICS PV %s does not exist', pv_name)

            self.evts_scalars[epv_name].append(epv)

        # Other scalars in the event
        for scalar in self.input_scalars :
            if scalar not in self.evts_scalars.keys() :
                self.evts_scalars[scalar] = []

            #scalar = evt.get(scalar)
            ipmFex = evt.get(xtc.TypeId.Type.Id_IpmFex, scalar )
            self.evts_scalars[scalar].append(ipmFex.sum)


    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        print "End calibcycle %d had %d events " % (self.n_ccls, self.n_evts)
        logging.info( "pyana_scan.endcalibcycle() called" )
        
        self.ccls_nevts.append(self.n_evts)

        # process the chunk of events collected in this scan cycle
        for name, list in self.evts_scalars.iteritems() :
            arr = np.array(list)
            mean =  np.mean(arr)
            std = np.std(arr)

            if name not in self.ccls_scalars.keys() :
                self.ccls_scalars[name] = []

            self.ccls_scalars[name].append( np.array([mean,std]) )

    def endrun( self, env ) :
        """This optional method is called if present at the end of the run.
        
        @param env    environment object
        """
        print "End run %d had %d calibcycles " % (self.n_runs, self.n_ccls)

        #arr = np.array(self.ccls_nevts)

        for ctrl, values in self.ccls_ctrl.iteritems() : 
            ctrl_array = np.array(values)
            self.make_graph1(ctrl_array,fignum=1,xtitle='Scan step',ytitle=ctrl)

            for sc_name, sc_list in self.ccls_scalars.iteritems() :
                mean_std_arr = np.array( sc_list )

                self.make_profile(ctrl_array,
                                  mean_std_arr[:,0], xtitle = ctrl, 
                                  yerr=mean_std_arr[:,1], ytitle = sc_name,
                                  fignum=2 )

        plt.show()

        
        return
        
        logging.info( "pyana_scan.endrun() called" )

    def endjob( self, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        logging.info( "pyana_scan.endjob() called" )
        print "End job %d had %d runs " % (self.n_jobs, self.n_runs)


    def make_histogram(self, array):
        pass
    
    def make_graph1(self, array, fignum=600, xtitle="",ytitle="",suptitle=""):
        print "Make graph from array ", np.shape(array)

        fig = plt.figure(num=fignum, figsize=(8,8) )
        fig.suptitle(suptitle)
        
        ax1 = fig.add_subplot(111)
        plt.plot(array,'bo:')
        plt.title('')
        plt.xlabel(xtitle,horizontalalignment='left') # the other right
        plt.ylabel(ytitle,horizontalalignment='right')
        plt.draw()

    def make_graph2(self, array1, array2, fignum=600, xtitle="",ytitle="",suptitle=""):

        fig = plt.figure(num=fignum, figsize=(8,8) )
        fig.suptitle(suptitle)
        
        ax1 = fig.add_subplot(111)
        plt.plot(array1,array2,'bo:')
        plt.title('')
        plt.xlabel(xtitle,horizontalalignment='left') # the other right
        plt.ylabel(ytitle,horizontalalignment='right')
        plt.draw()

    def make_profile(self, array1, array2, xerr=None, yerr=None, fignum=600, xtitle="",ytitle="",suptitle=""):

        fig = plt.figure(num=fignum, figsize=(8,8) )
        fig.suptitle(suptitle)
        
        ax1 = fig.add_subplot(111)
        plt.errorbar(array1,array2,xerr=xerr, yerr=yerr)
        plt.title('')
        plt.xlabel(xtitle,horizontalalignment='left') # the other right
        plt.ylabel(ytitle,horizontalalignment='right')
        plt.draw()

