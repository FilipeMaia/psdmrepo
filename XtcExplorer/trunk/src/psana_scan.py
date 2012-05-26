#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $
#
# Description:
#  Pyana user analysis module psana_scan...
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
from utilities import ScanData


#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------

class psana_scan (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self,
                   controlpv = None,
                   input_epics = None,
                   input_scalars = None,
                   #plot_every_n = None,
                   fignum = "1" ) :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param controlpv        Name(s) of control PVs to use
                                if none given, use whatever we find in the event. 
        @param input_epics      Name(s) of other scalars to correlate in scan
        @param input_scalars    Name(s) of other scalars to correlate in scan
        #@param plot_every_n     Frequency for plotting. If n=0, no plots till the end
        @param fignum           Matplotlib figure number
        """
        logging.getLogger().setLevel(logging.INFO)
        opt = PyanaOptions()
        self.controlpv = opt.getOptStrings(controlpv)
        self.input_epics = opt.getOptStrings(input_epics)
        self.input_scalars = opt.getOptStrings(input_scalars)
        #self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.mpl_num = opt.getOptInteger(fignum)        

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

        print "m_fullName=", self.m_fullName
        print "m_className=", self.m_className

        # data counters
        self.n_runs =  0 # number of runs in this job             

        # store in a datastructure for ipython
        self.data_scan = {}
        for pv in self.controlpv :
            self.data_scan[pv] = ScanData( pv )
        
            
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
        self.n_shots = 0
        self.evts_scalars = {}
        print "Begin calibcycle ", self.n_ccls

        ctrl_config = env.getConfigByType("Psana::ControlData::Config", "")
        if not ctrl_config:
            print "ctrl_config not found"
            sys.exit(0)
            return
        print "ctrl_config=", ctrl_config

        # control.ConfigV1 element
        env.printAllKeys()
        print "xtc.TypeId.Type.Id_ControlConfig=", xtc.TypeId.Type.Id_ControlConfig
        ctrl_config = env.getConfig(xtc.TypeId.Type.Id_ControlConfig, "ProcInfo()")


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

        #ctrl_config = env.getConfig(xtc.TypeId.Type.Id_ControlConfig)
        #
        # Id_ControlConfig is found in control.ddl.h Psana::ControlData::ConfigV1
        #
        self.source = env.configSource("ProcInfo()")
        ctrl_config = env.getConfigByType("Psana::ControlData::Config", "")
        if not ctrl_config:
            return
        print "ctrl_config=", ctrl_config

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
        self.n_shots += 1
        print "INFO: psana_scan.event() called (%d)" % self.n_shots

        #if evt.get('skip_event'):
        #    return

        # Use environment object to access EPICS data
        if False: #for epv_name in self.input_epics :

            # at first event, make a list for each scalar, to store event data
            if epv_name not in self.evts_scalars.keys() :
                print epv_name
                self.evts_scalars[epv_name] = []

            # store the value
            epv = env.epicsStore().value(epv_name)
            if not epv:
                logging_warning('EPICS PV %s does not exist', epv_name)
            else :
                self.evts_scalars[epv_name].append(epv.value)

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

            elif scalar.find("FEEGasDetEnergy")>= 0 :
                fee_energy_array = evt.getFeeGasDet()
                if fee_energy_array:
                    energy= (fee_energy_array[2]+fee_energy_array[3])/2.0 
                    self.evts_scalars[scalar].append( energy)
                #else :
                #    self.evts_scalars[scalar].append(-99.0)

            elif scalar.find("PhaseCavity")>= 0 :
                pc = evt.getPhaseCavity()
                if pc:
                    val = (pc.fCharge1 - pc.fCharge2) / (pc.fFitTime1 - pc.fFitTime2)
                    self.evts_scalars[scalar].append( val )
                #else :
                #    self.evts_scalars[scalar].append(-99.0)

            elif ( scalar.find("IPM")>= 0 or scalar.find("DIO")>= 0 ):
                ipm = evt.get(xtc.TypeId.Type.Id_SharedIpimb, scalar )
                if ipm:
                    self.evts_scalars[scalar].append( ipm.ipmFexData.sum )
                #else :
                #    self.evts_scalars[scalar].append(-99.0)


    def endcalibcycle( self, evt, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        print "End calibcycle %d had %d events " % (self.n_ccls, self.n_shots)
        print "psana_scan.endcalibcycle() called" 
        
        self.ccls_nevts.append(self.n_shots)

        # process the chunk of events collected in this scan cycle
        for name, list in self.evts_scalars.iteritems() :
            arr = np.array(list)
            mean =  np.mean(arr)
            std = np.std(arr)

            if name not in self.ccls_scalars.keys() :
                self.ccls_scalars[name] = []

            self.ccls_scalars[name].append( np.array([mean,std]) )


    def endrun( self, evt, env ) :
        """This optional method is called if present at the end of the run.
        This is where we draw plots: One window showing several plots
        
        @param env    environment object
        """
        print "psana_scan.endrun() called" 
        print "End run %d had %d calibcycles " % (self.n_runs, self.n_ccls)


    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        print "End job had %d runs " % (self.n_runs)

        self.make_plots(fignum=self.fignum, suptitle="Scan (%d calib cycles, %d shots each)" \
                        %(self.n_ccls,self.n_shots))
        
        
        self.data = []
        for pv,data in self.data_scan.iteritems():
            self.data.append( data )
        evt.put( self.data, "data_scan")

    def make_plots(self, fignum=1, suptitle=""):

        nctrl = len(self.ccls_ctrl) 
        nsclr = len(self.ccls_scalars)
        nplots = nctrl + nctrl*nsclr
        ncols = 1
        nrows = 1
        if nplots == 2: ncols = 2
        if nplots == 3: ncols = 3
        if nplots == 4: ncols = 2; nrows = 2
        if nplots > 4:
            ncols = 3
            nrows = nplots / 3
            if nplots%3 > 0 : nrows += 1

        height=4.2
        if (nrows * height) > 14 : height = 14/nrows
        width=height*1.3

        print "Have %d variables to be plotted, layout = %d x %d, %.2f x %.2f" % \
              (nplots, nrows,ncols,(nrows*height),(ncols*width))
                
        
        # make figure
        fig = plt.figure(num=fignum, figsize=(width*ncols,height*nrows) )
        plt.clf()
        fig.subplots_adjust(wspace=0.4)
        fig.suptitle(suptitle)

        pos = 0
        for ctrl, values in self.ccls_ctrl.iteritems() : 

            # convert to numpy arrays
            ctrl_array = np.array(values)
            data_names = []
            data_array = []
        
            # store arrays (for iptyhon)
            self.data_scan[ctrl].scanvec = ctrl_array

            pos += 1
            ax1 = fig.add_subplot(nrows,ncols,pos)

            min,max = ctrl_array[0],ctrl_array[-1]
            nsteps = len(ctrl_array)

            plt.plot(ctrl_array,'bo:')

            plt.title('')
            plt.xlim(0-nsteps*0.1,nsteps+nsteps*0.1)
            plt.ylim(min-(max-min)/10,max+(max-min)/10)
            plt.xlabel("Scan step",horizontalalignment='left') # the other right
            plt.ylabel(ctrl,horizontalalignment='right')
            plt.draw()

            for sc_name, sc_list in self.ccls_scalars.iteritems() :
                
                pos += 1
                axn = fig.add_subplot(nrows,ncols,pos)

                mean_std_arr = np.array( sc_list ) 

                data_names.append( sc_name )
                data_array.append( mean_std_arr )

                plt.errorbar(ctrl_array,
                             mean_std_arr[:,0],
                             yerr=mean_std_arr[:,1],
                             marker='s')
                plt.title('')
                plt.xlim(min-((max-min)/10),max+((max-min)/10))
                plt.xlabel(ctrl,horizontalalignment='left') # the other right
                plt.ylabel(sc_name,horizontalalignment='right')
                plt.draw()

            # store more arrays (for iptyhon)
            self.data_scan[ctrl].arheader = data_names
            self.data_scan[ctrl].scandata = np.float_( data_array )


        plt.draw()

        
