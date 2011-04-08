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

        filtbool = {}
        for addr in self.ipimb_addresses :
            channels = np.float_(self.fex_channels[addr])
            plt.title(addr)
            lims = self.get_limits(channels,fignum=2,method="corrfrac")

            filtbool[addr] = self.get_filter(channels, lims)
            print len(filtbool[addr])

        #self.make_plots(fignum=1, suptitle="All in one")        


    def endjob( self, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        logging.info( "pyana_xppscan.endjob() called" )
        print "End job had %d runs " % (self.n_runs)


    def get_filter(self,channels,lims):
        """Return a filtbool array, the element-wise AND of
        boolean filter array for each channel.        
        @return filtbool
        @param  channels
        @param  lims
        """
        nev,nch = np.shape(channels)

        filtbool = []
        for stepNO in range (0,self.n_ccls):
            tfiltbool = np.ones((1,nev),bool)
            for chNO in range (0,4):
                tv = channels[:,chNO]
                if not np.any( lims[chNO,:]==[0,0] ):
                    tfiltbool = np.logical_and( tfiltbool, self.filtvec(tv,lims[chNO,:]) )

            filtbool.append( tfiltbool )

        return filtbool

    
    def filtvec(self, vec,lims):
        """Helper function:
        returns a filter vector (of True and False values) based on
        @param vec       - input vector and
        @param lims      - acceptance limits
        """
        above_minimum = (vec > min(lims))
        below_maximum = (vec < max(lims))

        return np.logical_and( above_minimum, below_maximum )



    def get_limits(self, channels, fignum=1, method="corrfrac"):
        """Get limits from graphical input
        @return  lims     - [[x,y],[x,y],[x,y],[x,y]] limit coordinates, one row for each channel 

        @param fignum
        @param channels       array of IPM channels
        """
        nev, nch = np.shape(channels)
        print "%d channels with %d events"%(nch,nev)

        plt.clf()
        fig = plt.figure(num=fignum, figsize=(7,7))
        
        nplots = nch
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

        lims = None
        if method=="corrfrac":    lims = self.get_limits_corrfrac(channels,fig)
        if method=="correlation": lims = self.get_limits_correlation(channels,fig)
        if method=="automatic": lims = self.get_limits_automatic(channels,fig)
        if method=="channelhist": lims = self.get_limits_channelhist(channels,fig)

        print "limits: ", lims
        return lims

    def get_limits_correlation(self, channels, fig):
        """Get limits from graphical input
        @return  lims      - [[x,y],[x,y],[x,y],[x,y]] limit coordinates, one row for each channel 
        @param channels    - array of channels form a single IPIMB
        @param fig         - pointer to the figure
        """
        lims = np.zeros((4,2),dtype="float")

        ax2 = fig.add_subplot(2,2,1)
        plt.loglog(channels[:,0],channels[:,1],'o')
        plt.xlabel('Channel0')
        plt.ylabel('Channel1')
        plt.draw()
        
        plt.axes(ax2)
        plt.hold(True)
        lims[0:2,:] = plt.ginput(2)
        fbool = (self.filtvec(channels[:,0],lims[0:2,0]))&self.filtvec(channels[:,1],lims[0:2,1])
        plt.loglog(channels[fbool,0],channels[fbool,1],'or')
        plt.draw()
        print "indexes that pass filter: ", np.where(fbool==True)
                
        ax3 = fig.add_subplot(2,2,2)
        plt.loglog(channels[:,2],channels[:,3],'o')
        plt.xlabel('Channel2')
        plt.ylabel('Channel3')
        plt.draw()

        plt.axes(ax3)
        plt.hold(True)
        lims[2:4,:] = plt.ginput(2)
        fbool = (self.filtvec(channels[:,2],lims[2:4,0]))&self.filtvec(channels[:,3],lims[2:4,1])
        plt.loglog(channels[fbool,2],channels[fbool,3],'or')
        plt.draw()            
        print "indexes that pass filter: ", np.where(fbool==True)
        
        plt.draw()
        return lims
            
    def get_limits_corrfrac(self, channels, fig):
        """Get limits from graphical input
        @return  lims     - [[x,y],[x,y],[x,y],[x,y]] limit coordinates, one row for each channel 
        @param  channels
        @param  fig
        """
        lims = np.zeros((4,2),dtype="float")

        ax1 = fig.add_subplot(2,2,1)
        plt.loglog(channels[:,0],channels[:,1]/channels[:,0],'o')
        plt.xlabel('Channel0')
        plt.ylabel('Channel1/Channel0')
        plt.axvline(np.mean( channels[:,0] ))
        a1xlim = np.log10( plt.xlim() )

        a1N = fig.add_subplot(2,2,2)
        histv = np.logspace(a1xlim[0],a1xlim[1], np.round( channels[:,0].size / 50 ))
        N1plot, bin_edges1 = np.histogram(channels[:,0],bins=histv)
        plt.semilogx(bin_edges1[0:N1plot.size],N1plot,'k')
        plt.xlabel("Channel0")
        plt.ylabel('N')
        plt.draw()

        ax2 = fig.add_subplot(2,2,3)
        plt.loglog(channels[:,2],channels[:,3]/channels[:,2],'o')
        plt.xlabel('Channel2')
        plt.ylabel('Channel3/Channel2')
        plt.axvline(np.mean( channels[:,2] ))
        a2xlim = np.log10( plt.xlim() )

        a2N = fig.add_subplot(2,2,4)
        histv = np.logspace(a2xlim[0],a2xlim[1],np.round( channels[:,2].size / 50 ))
        N2plot, bin_edges2 = np.histogram( channels[:,2],bins=histv )
        plt.semilogx(bin_edges2[0:N2plot.size],N2plot,'k')
        plt.xlabel('Channel2')
        plt.ylabel('N')

        plt.axes(ax1)
        plt.hold(True)
        lims[0:2,:] = plt.ginput(2)
        fbool = (self.filtvec(channels[:,0],lims[0:2,0]))&self.filtvec(channels[:,1],lims[0:2,1])
        plt.loglog(channels[fbool,0],channels[fbool,1],'or')
        plt.draw()

        plt.axes(ax2)
        plt.hold(True)
        lims[2:4,:] = plt.ginput(2)
        fbool = (self.filtvec(channels[:,2],lims[2:4,0]))&self.filtvec(channels[:,3],lims[2:4,1])
        plt.loglog(channels[fbool,2],channels[fbool,3],'or')
        plt.draw()            
            
        plt.draw()
        return lims

    def getSTDMEANfrac_from_startpoint(self,x,y,x0):
        """Get fraction std/mean of y
        x = ch0
        y = ch1/ch0
        x0 = mean(ch0)
        iP are those indices for which x > x0
        iN are those indices for which x < x0
        """
        ind = np.argsort(x)
        xn = x[ind] #sorted according to x
        yn = y[ind] #sorted according to x

        frac = np.zeros(np.shape(yn))
        
        iP = (xn>x0) #array of True's and False's. 
        iN = (xn<x0) #array of True's and False's. 

        yP = yn[iP] # y, sorted according to x, and only for values where x > x0
        yN = yn[iN] # y, sorted according to x, and only for values where x < x0

        indx_xpos = np.nonzero(iP)[0] # indexes of x for which x>x0
        indx_xneg = np.nonzero(iN)[0] # indexes of x for which x<x0

        firstindexP = indx_xpos[0]  # first index of sorted x where x>x0
        lastindexN  = indx_xneg[-1] #  last index of sorted x where x<x0
        
        for n in range (0,len(yP)):
            frac[firstindexP+n] = np.std(yP[0:n])/np.mean(yP[0:n]);

            
        for n in range (0,len(yN)):
            frac[lastindexN-n] = np.std(yN[-(1+n):])/np.mean(yN[-(1+n):]);

        return xn,frac 


    def get_limits_automatic(self, channels, fig):
        """Get limits from graphical input
        @return  lims     - [[x,y],[x,y],[x,y],[x,y]] limit coordinates, one row for each channel 
        """
        lims = np.zeros((4,2),dtype="float")

        ax1 = fig.add_subplot(1,3,1)
        plt.loglog(channels[:,0],channels[:,1]/channels[:,0],'o')
        plt.axvline(np.mean( channels[:,0] )) # draw vertical line
        plt.xlabel('Channel0')
        plt.ylabel('Channel1/Channel0')
        
        ax2 = fig.add_subplot(1,3,2)
        plt.loglog(channels[:,2],channels[:,3]/channels[:,2],'o')
        plt.axvline(np.mean( channels[:,2] )) # draw vertical line
        plt.xlabel('Channel2')
        plt.ylabel('Channel3/Channel2')

        xn, frac = self.getSTDMEANfrac_from_startpoint(channels[:,0],
                                                       channels[:,1]/channels[:,0],
                                                       np.mean(channels[:,0]) )
            
        ax3 = fig.add_subplot(1,3,3)
        plt.plot(xn, frac,'o-')
        plt.draw()

        print "So where's the limit set?!"
        return lims
                

    def get_limits_channelhist(self, channels,fig):
        """Get limits from graphical input
        @return  lims     - [[x,y],[x,y],[x,y],[x,y]] limit coordinates, one row for each channel 
        """

        lims = np.zeros((4,2),dtype="float")

        plt.hold(True)
        for chNO in range (0, 4):
            tN, tx, patches = plt.hist(channels[:,chNO], len(channels[:,chNO])/100)
            plt.plot(tx[0:tN.size],tN,'.',label="Channel%d"%chNO)
            
        plt.legend(loc=2)
        plt.draw()

        print 'Select lower and upper limit of all 4 channels, reverse order in case channel is bad'
        for chNO in range (0, 4):
            tN, tx, patches = plt.hist(channels[:,chNO], len(channels[:,chNO])/100)
            tph = plt.plot(tx[0:tN.size],tN,'.',label="Channel%d"%chNO)
            print "expecting input"
            tlim = np.array(plt.ginput(2)) # returns a list of x,y coordinates
            tlim = tlim[:,0]               # get the x coordinates (column 0)
            if tlim[0] > tlim[1] :
                lims[chNO,:] = [0, 0]
            else :
                lims[chNO,:] = tlim
            del tlim
            print "limits for Ch", chNO, " = ", lims[chNO,:]
            plt.draw()

            
        print "draw"
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

        
