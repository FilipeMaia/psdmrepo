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
class pyana_xppscan (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self,
                   controlpv = None,
                   input_epics = None,
                   input_scalars = None,
                   ipimb_addresses = "",
                   plot_every_n = None,
                   fignum = "1" ) :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param controlpv        Name(s) of control PVs to use
                                if none given, use whatever we find in the event. 
        @param input_epics      Name(s) of other scalars to correlate in scan
        @param input_scalars    Name(s) of other scalars to correlate in scan
        @param ipimb_addresses  list of IPIMB addresses
        @param plot_every_n     Frequency for plotting. If n=0, no plots till the end
        @param fignum           Matplotlib figure number
        """
        opt = PyanaOptions()
        self.controlpv = opt.getOptStrings(controlpv)
        self.input_epics = opt.getOptStrings(input_epics)
        self.input_scalars = opt.getOptStrings(input_scalars)
        self.ipimb_addresses = opt.getOptStrings(ipimb_addresses)
        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.mpl_num = opt.getOptInteger(fignum)

        print self.ipimb_addresses

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
        self.n_runs =  0 # number of runs in this job             
            
    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """
        self.n_runs += 1

        print "Processing run number ", evt.run()
        logging.info( "pyana_xppscan.beginrun() called (%d)"%self.n_runs )

        # initialize calibcycle data
        self.n_ccls = 0
        self.ccls_nevts  = []
        self.ccls_ctrl = {}   # to hold ControlPV names and values
        self.ccls_scalars = {} # to hold epics and other scalar mean and std

        self.fex_channels = {}

        for addr in self.ipimb_addresses :
            self.fex_channels[addr] = list()
            
        # histogram of ipm
        self.ipm_channels = None
                
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
        logging.info( "pyana_xppscan.event() called (%d)"%self.n_evts )

        # Use environment object to access EPICS data
        for epv_name in self.input_epics :

            # at first event, make a list for each scalar, to store event data
            if epv_name not in self.evts_scalars.keys() :
                print epv_name
                self.evts_scalars[epv_name] = []

            # store the value
            epv = env.epicsStore().value(epv_name)
            if not epv:
                logging.warning('EPICS PV %s does not exist', epv_name)
            else :
                self.evts_scalars[epv_name].append(epv.value)

        for ipimb in self.ipimb_addresses :
            ipmFex = evt.get(xtc.TypeId.Type.Id_IpmFex, ipimb)

            self.fex_channels[ipimb].append(ipmFex.channel)


        # Other scalars in the event
        for scalar in self.input_scalars :

            # at first event, make a list for each scalar, to store event data
            if scalar not in self.evts_scalars.keys() :
                self.evts_scalars[scalar] = []


            """Here's a lot of hardcoded stuff. If you want other
            things plotted to evaluate the motor scan, you need
            to edit this code.
            """
            if scalar.find("Ipimb")>=0 :
                ipmFex = evt.get(xtc.TypeId.Type.Id_IpmFex, scalar )
                if ipmFex :
                    self.evts_scalars[scalar].append(ipmFex.sum)
                #else:
                #    self.evts_scalars[scalar].append(-99.0)
                    
            elif scalar.find("EBeam")>= 0 :
                ebeam = evt.getEBeam()
                if ebeam:
                    self.evts_scalars[scalar].append(ebeam.fEbeamL3Energy)
                #else :
                #    self.evts_scalars[scalar].append(-99.0)

            elif scalar.find("FeeGasDetEnergy")>= 0 :
                fee_energy_array = evt.getFeeGasDet()
                if fee_energy_array:
                    self.evts_scalars[scalar].append( np.sum(fee_energy_array) )
                #else :
                #    self.evts_scalars[scalar].append(-99.0)

            elif scalar.find("PhaseCavity")>= 0 :
                pc = evt.getPhaseCavity()
                if pc:
                    val = (pc.fCharge1 - pc.fCharge2) / (pc.fFitTime1 - pc.fFitTime2)
                    self.evts_scalars[scalar].append( val )
                #else :
                #    self.evts_scalars[scalar].append(-99.0)


    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        print "End calibcycle %d had %d events " % (self.n_ccls, self.n_evts)
        logging.info( "pyana_xppscan.endcalibcycle() called" )
        
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
        This is where we draw plots: One window showing several plots
        
        @param env    environment object
        """
        logging.info( "pyana_xppscan.endrun() called" )
        print "End run %d had %d calibcycles " % (self.n_runs, self.n_ccls)

        self.get_limits(fignum=2, suptitle="Click to select limits")
        self.make_plots(fignum=1, suptitle="All in one")        


    def endjob( self, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        logging.info( "pyana_xppscan.endjob() called" )
        print "End job had %d runs " % (self.n_runs)

    def filtvec(self, vec,lims):
        return ((vec>min(lims))&(vec<max(lims)))

    def get_limits(self, fignum=1, suptitle=""):

        plt.clf()

        fig = plt.figure(num=fignum, figsize=(10,10))
        fig.suptitle(suptitle)

        lims = np.zeros((4,2),dtype="float")
        for addr in self.ipimb_addresses :

            channels = np.float_(self.fex_channels[addr])

            ax2 = fig.add_subplot(1,2,1)
            xaxis = np.arange( 0, len(self.fex_channels[addr]) )
            plt.loglog(channels[:,0],channels[:,1],'o')
            plt.xlabel('Channel0')
            plt.ylabel('Channel1')
            plt.title(addr)
        
            ax3 = fig.add_subplot(1,2,2)
            xaxis = np.arange( 0, len(self.fex_channels[addr]) )
            plt.loglog(channels[:,2],channels[:,3],'o')
            plt.xlabel('Channel0')
            plt.ylabel('Channel1')
            plt.title(addr)

            plt.axes(ax2)
            plt.hold(True)
            lims[0:2,:] = plt.ginput(2)
            fbool = (self.filtvec(channels[:,0],lims[0:2,0]))&self.filtvec(channels[:,1],lims[0:2,1])
            plt.loglog(channels[fbool,0],channels[fbool,1],'or')

            plt.axes(ax3)
            plt.hold(True)
            lims[2:4,:] = plt.ginput(2)
            fbool = (self.filtvec(channels[:,2],lims[2:4,0]))&self.filtvec(channels[:,3],lims[2:4,1])
            plt.loglog(channels[fbool,2],channels[fbool,3],'or')

            print np.shape(lims[0:2,:]), np.shape(lims[2:4,:])
            print lims
            print lims[0:2,:]
            print lims[2:4,:]

        plt.draw()
        

    def make_plots(self, fignum=1, suptitle=""):

        plt.clf()
        
        nplots = 1 + len(self.ccls_scalars)
        ncols = 1
        nrows = 1
        if nplots == 2: ncols = 2
        if nplots == 3: ncols = 3
        if nplots == 4: ncols = 2; nrows = 2
        if nplots > 4:
            ncols = 3
            nrows = nplots / 3
            if nplots%3 > 0 : nrows += 1

        height=3.5
        if nrows * 3.5 > 14 : height = 14/nrows
        width=height*1.3
        print "width, height = ", width, height
        print "Have %d variables to be plotted, layout = %d x %d" % (nplots, nrows,ncols)
                
        #fig = plt.figure(num=fignum, figsize=(height*nrows,width*ncols) )
        fig = plt.figure(num=fignum, figsize=(10,10))
        fig.suptitle(suptitle)

        pos = 0
        for ctrl, values in self.ccls_ctrl.iteritems() : 
            ctrl_array = np.array(values)

            pos += 1
            ax1 = fig.add_subplot(nrows,ncols,pos)

            plt.plot(ctrl_array,'bo:')
            plt.title('')
            plt.xlabel("Scan step",horizontalalignment='left') # the other right
            plt.ylabel(ctrl,horizontalalignment='right')
            plt.draw()

            for sc_name, sc_list in self.ccls_scalars.iteritems() :
                
                pos += 1
                axn = fig.add_subplot(nrows,ncols,pos)

                mean_std_arr = np.array( sc_list )

                plt.errorbar(ctrl_array,
                             mean_std_arr[:,0],
                             yerr=mean_std_arr[:,1])
                plt.title('')
                plt.xlabel(ctrl,horizontalalignment='left') # the other right
                plt.ylabel(sc_name,horizontalalignment='right')
                plt.draw()


        plt.draw()

        
