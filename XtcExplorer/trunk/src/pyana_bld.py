#
# bld.py: plot beamline data
#
# 

import numpy as np
import logging
import matplotlib.pyplot as plt
from utilities import PyanaOptions 
from utilities import BldData
from pypdsdata import xtc


# analysis class declaration
class  pyana_bld ( object ) :
    
    def __init__ ( self,
                   do_ebeam        = "False",
                   do_gasdetector  = "False",
                   do_phasecavity  = "False",
                   do_ipimb        = "False",
                   plot_every_n    = None,
                   accumulate_n    = "0",
                   fignum          = "1" ):
        """
        Initialize data. Parameters:
        @param do_ebeam            Plot data from EBeam object
        @param do_gasdetector      Plot data from GasDetector
        @param do_phasecavity      Plot data from PhaseCavity
        @param do_ipimb            Plot data from Shared Ipimb
        @param plot_every_n        Plot after every N events
        @param accumulate_n        Accumulate all (0) or reset the array every n shots
        @param fignum              Matplotlib figure number
        """

        # parameters
        opt = PyanaOptions() # convert option string to appropriate type
        self.do_EBeam     = opt.getOptBoolean(do_ebeam)
        self.do_GasDet    = opt.getOptBoolean(do_gasdetector)
        self.do_PC        = opt.getOptBoolean(do_phasecavity)
        self.do_ipimb     = opt.getOptBoolean(do_ipimb)
        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.accumulate_n = opt.getOptInteger(accumulate_n)
        self.mpl_num      = opt.getOptInteger(fignum)

        # other
        self.n_shots = None
        self.accu_start = None

        # flags
        self.hadEB = False
        self.hadPC = False
        self.hadGD = False
        self.hadIPM = False

        # lists to fill numpy arrays
        self.initlists()


    def initlists(self):
        self.time = []

        self.EB_damages = []
        self.EB_energies = []
        self.EB_positions = []
        self.EB_angles = []
        self.EB_charge = []

        self.GD_energies = []

        self.PC_ftime1 = []
        self.PC_ftime2 = []
        self.PC_fchrg1 = []
        self.PC_fchrg2 = []

        self.IPM_fex_sum = []
        self.IPM_fex_channels = []
        self.IPM_raw_channels = []
        self.IPM_fex_position = []

    def resetlists(self):
        self.accu_start = self.n_shots
        del self.time[:]

        del self.EB_damages[:]
        del self.EB_energies[:]
        del self.EB_positions[:]
        del self.EB_angles[:]
        del self.EB_charge[:]

        del self.GD_energies[:]

        del self.PC_ftime1[:]
        del self.PC_ftime2[:]
        del self.PC_fchrg1[:]
        del self.PC_fchrg2[:]

        del self.IPM_fex_sum[:]
        del self.IPM_fex_channels[:]
        del self.IPM_raw_channels[:]
        del self.IPM_fex_position[:]


    def beginjob ( self, evt, env ) : 
        logging.info("pyana_bld.beginjob() called, process %d"%env.subprocess())

        self.n_shots = 0
        self.accu_start = 0

        self.data = {}
        if self.do_EBeam:  self.data["EBeam"]       = BldData("EBeam") 
        if self.do_GasDet: self.data["GasDetector"] = BldData("GasDetector")
        if self.do_PC :    self.data["PhaseCavity"] = BldData("PhaseCavity") 
        if self.do_ipimb : self.data["SharedIpimb"] = BldData("SharedIpimb") 

        self.doPlot = (env.subprocess()<1) and (self.plot_every_n != 0)

    def event ( self, evt, env ) :
        self.n_shots += 1

        do_plot = self.doPlot and (self.n_shots%self.plot_every_n)==0 

        # if a prior module has failed a filter...
        if evt.get('skip_event'):
            return

        self.time.append( evt.getTime().seconds() + 1.0e-9*evt.getTime().nanoseconds() )

        if self.do_EBeam :
            # EBeam object (of type bld.BldDataEBeam or bld.BldDataEBeamV0)
            ebeam = evt.getEBeam()
            self.hadEB = True
            if ebeam :
                beamDmgM = ebeam.uDamageMask
                beamChrg = ebeam.fEbeamCharge 
                beamEnrg = ebeam.fEbeamL3Energy
                beamPosX = ebeam.fEbeamLTUPosX
                beamPosY = ebeam.fEbeamLTUPosY
                beamAngX = ebeam.fEbeamLTUAngX
                beamAngY = ebeam.fEbeamLTUAngY
                #beamPkCr = ebeam.fEbeamPkCurrBC2

                self.EB_energies.append(beamEnrg)
                self.EB_positions.append( [beamPosX,beamPosY] )
                self.EB_angles.append( [beamAngX, beamAngY] )
                self.EB_charge.append( beamChrg )
            else : 
                if self.n_shots < 2 :
                    print "No EBeam object found in shot#%d" % self.n_shots
                if self.hadEB :
                    print "No EBeam object found in shot#%d" % self.n_shots
                    self.EB_energies.append( -9.0 )
                    self.EB_positions.append( [-9.0, -9.0] )
                    self.EB_angles.append( [0.0, 0.0] )
                    self.EB_charge.append( 0.0 )
                    
            
        if self.do_GasDet :
            # returns array of 4 numbers obtained from the bld.BldDataFEEGasDetEnergy object
            fee_energy_array = evt.getFeeGasDet()

            if fee_energy_array is None :
                if self.n_shots < 2 :
                    print "No Gas Detector data object found in shot#%d"%self.n_shots
                if self.hadGD:
                    print "No Gas Detector data object found in shot#%d" % self.n_shots
                    self.GD_energies.append( [0.0,0.0,0.0,0.0] )
            else :
                # fee_energy_array is a 4-number vector
                self.GD_energies.append( fee_energy_array )

            
        if self.do_PC:
            # PhaseCavity object (of type bld.BldDataPhaseCavity)
            pc = evt.getPhaseCavity()
            if pc :
                self.PC_ftime1.append( pc.fFitTime1 )
                self.PC_ftime2.append( pc.fFitTime2 )
                self.PC_fchrg1.append( pc.fCharge1 )
                self.PC_fchrg2.append( pc.fCharge2 ) 
            else :
                if self.n_shots < 2 :
                    print "No Phase Cavity data object found in shot#%d" % self.n_shots
                if self.hadPC :
                    print "No Phase Cavity data object found in shot#%d" % self.n_shots
                    self.PC_ftime1.append( -999.0 )
                    self.PC_ftime2.append( -999.0 )
                    self.PC_fchrg1.append( -999.0 )
                    self.PC_fchrg2.append( -999.0 ) 


        if self.do_ipimb : # BldDataIpimb / SharedIpimb
            ipm = evt.get(xtc.TypeId.Type.Id_SharedIpimb )
            if ipm :
                self.IPM_raw_channels.append( [ipm.ipimbData.channel0Volts(),
                                               ipm.ipimbData.channel1Volts(),
                                               ipm.ipimbData.channel2Volts(),
                                               ipm.ipimbData.channel3Volts()] )
                self.IPM_fex_channels.append( ipm.ipmFexData.channel )
                self.IPM_fex_sum.append( ipm.ipmFexData.sum )
                self.IPM_fex_position.append( [ipm.ipmFexData.xpos, ipm.ipmFexData.ypos] )
            else :
                if self.n_shots < 2 :
                    print "No BldDataIpimb data object found in shot#%d" % self.n_shots
                if self.hadIPM:
                    print "No BldDataIpimb data object found in shot#%d" % self.n_shots
                    self.IPM_raw_channels.append( [0.0,0.0,0.0,0.0] )
                    self.IPM_fex_channels.append( [0.0,0.0,0.0,0.0] )
                    self.IPM_fex_sum.append( 0.0 )
                    self.IPM_fex_position.append( [0.0,0.0] )
                    
        # ----------------- Plotting --------------------- 
        if do_plot :
            header = "shots %d-%d" % (self.accu_start, self.n_shots)
            self.make_plots(self.mpl_num, suptitle=header)

            # flag for pyana_plotter
            evt.put(True, 'show_event')

            data_bld = []
            for name,data in self.data.iteritems() :
                data_bld.append( data )
                    
            # give the list to the event object
            evt.put( data_bld, 'data_bld' )
        #else:
        #    print "bld doPlot is False ",
        #    print "subprocess < 1 ? ", (env.subprocess()<1)
        #    print "plot_every_n ? ", self.plot_every_n 
        #    print "modulus? ", (self.n_shots%self.plot_every_n)
            
        # --------- Reset -------------
        if self.accumulate_n!=0 and (self.n_shots%self.accumulate_n)==0 :
            self.resetlists()
                
                
    def endjob( self, evt, env ) :
        
        print "pyana_bld endjob has been reached, after processing %d events"%self.n_shots

        # ----------------- Plotting ---------------------
        if (env.subprocess()<1):
            header = "shots %d-%d" % (self.accu_start, self.n_shots)
            self.make_plots(self.mpl_num, suptitle=header)

            # flag for pyana_plotter
            evt.put(True, 'show_event')

            data_bld = []
            for name,data in self.data.iteritems() :
                data_bld.append( data )
                
            # give the list to the event object
            evt.put( data_bld, 'data_bld' )


    def make_plots(self, fignum = 1, suptitle = ""):

        if self.accu_start == self.n_shots :
            print "Can't do ", suptitle
        
        if self.do_EBeam :
            if len(self.EB_charge) > 0 :

                print "Making plot of array of length %d"%len(self.EB_charge)

                # numpy arrays
                xaxis = np.arange( self.accu_start, self.n_shots ) 
                charge = np.float_(self.EB_charge)
                energies = np.float_(self.EB_energies)
                positions = np.float_(self.EB_positions)
                angles = np.float_(self.EB_angles)

                # store for later (ipython)
                self.data["EBeam"].charge = charge
                self.data["EBeam"].energy = energies
                self.data["EBeam"].position = positions
                self.data["EBeam"].angle = angles
                
                # make figure                
                fig = plt.figure(num=(fignum+1), figsize=(8,8) )
                fig.clf()
                fig.subplots_adjust(wspace=0.4, hspace=0.4)
                fig.suptitle("BldInfo:EBeam data " + suptitle)

                ax1 = fig.add_subplot(221)
                if (np.size(xaxis) != np.size(energies) ):
                    print "event    ", self.n_shots
                    print "axis     ", np.size(xaxis), np.shape(xaxis)
                    print "energies ", np.size(energies), np.shape(energies)
                plt.plot(xaxis,energies)
                plt.title("Beam Energy")
                plt.xlabel('Datagram record',horizontalalignment='left') # the other right
                plt.ylabel('Beam Energy',horizontalalignment='right')
                
                ax2 = fig.add_subplot(222)
                plt.scatter(positions[:,0],angles[:,0])
                plt.title('Beam X')
                plt.xlabel('position X',horizontalalignment='left')
                plt.ylabel('angle X',horizontalalignment='left')

                ax3 = fig.add_subplot(223)
                plt.scatter(positions[:,1],angles[:,1])
                plt.title("Beam Y")
                plt.xlabel('position Y',horizontalalignment='left')
                plt.ylabel('angle Y',horizontalalignment='left')

                ax4 = fig.add_subplot(224)
                n, bins, patches = plt.hist(self.EB_charge, 100, normed=1, histtype='stepfilled')
                plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
                plt.title('Beam Charge')
                plt.xlabel('Beam Charge',horizontalalignment='left') # the other right
                
                plt.draw()

        if self.do_GasDet :

            if len(self.GD_energies) > 0 :

                # numpy array (4d)
                array = np.float_(self.GD_energies)

                # store for later (ipython)
                self.data["GasDetector"].energy = array

                # make figure
                fig = plt.figure(num=(fignum+2), figsize=(8,8) )
                fig.clf()
                fig.subplots_adjust(wspace=0.4, hspace=0.4)
                fig.suptitle("BldInfo:FEEGasDetEnergy data " + suptitle)

                ax1 = fig.add_subplot(221)
                n, bins, patches = plt.hist(array[:,0], 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'r', 'alpha', 0.75)
                plt.title('Energy 11')
                plt.xlabel('Energy E[0]',horizontalalignment='left')
                
                ax2 = fig.add_subplot(222)
                n, bins, patches = plt.hist(array[:,1], 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'g', 'alpha', 0.75)
                plt.title('Energy 12')
                plt.xlabel('Energy E[1]',horizontalalignment='left')
                
                ax3 = fig.add_subplot(223)
                n, bins, patches = plt.hist(array[:,2], 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'b', 'alpha', 0.75)
                plt.title('Energy 21')
                plt.xlabel('Energy E[2]',horizontalalignment='left')
                
                ax4 = fig.add_subplot(224)
                n, bins, patches = plt.hist(array[:,3], 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'm', 'alpha', 0.75)
                plt.title('Energy 22')
                plt.xlabel('Energy E[3]',horizontalalignment='left')
                
                plt.draw()
            

        if self.do_PC :

            if len(self.PC_ftime1) > 0 :

                # numpy arrays
                ftime1 = np.float_(self.PC_ftime1)
                ftime2 = np.float_(self.PC_ftime2)
                fchrg1 = np.float_(self.PC_fchrg1)
                fchrg2 = np.float_(self.PC_fchrg2)

                # store for later (ipython)
                self.data["PhaseCavity"].time = np.vstack( (ftime1, ftime2)).T
                self.data["PhaseCavity"].charge = np.vstack( (fchrg1, fchrg2)).T
                
                # make figure
                fig = plt.figure(num=(fignum+3), figsize=(12,8) )
                fig.clf()
                fig.subplots_adjust(wspace=0.4, hspace=0.4)
                fig.suptitle("BldInfo:PhaseCavity data " + suptitle)

                ax1 = fig.add_subplot(231)
                n, bins, patches = plt.hist(ftime1, 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'r', 'alpha', 0.75)
                plt.title('Time PC1')
                plt.xlabel('Time PC1',horizontalalignment='left')
                unit = bins[1] - bins[0]
                x1min, x1max = (bins[0]-unit), (bins[-1]+unit)
                plt.xlim(x1min,x1max)
             
                ax2 = fig.add_subplot(232)
                n, bins, patches = plt.hist(ftime2, 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'r', 'alpha', 0.75)
                plt.title('Time PC2')
                plt.xlabel('Time PC2',horizontalalignment='left')
                unit = bins[1] - bins[0]
                x2min, x2max = (bins[0]-unit), (bins[-1]+unit)
                plt.xlim(x2min,x2max)

                ax3 = fig.add_subplot(233)
                plt.scatter(ftime1,ftime2)
                plt.title("Time PC1 vs. Time PC2")
                plt.xlabel('Time PC1',horizontalalignment='left')
                plt.ylabel('Time PC2',horizontalalignment='left')
                plt.xlim(x1min,x1max)
                plt.ylim(x2min,x2max)
                
                ax4 = fig.add_subplot(234)
                n, bins, patches = plt.hist(fchrg1, 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'b', 'alpha', 0.75)
                plt.title('Charge PC1')
                plt.xlabel('Charge PC1',horizontalalignment='left')
             
                ax5 = fig.add_subplot(235)
                n, bins, patches = plt.hist(fchrg2, 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'b', 'alpha', 0.75)
                plt.title('Charge PC2')
                plt.xlabel('Charge PC2',horizontalalignment='left')
                
                ax6 = fig.add_subplot(236)
                plt.scatter(fchrg1,fchrg2)
                plt.title("Charge PC1 vs. Charge PC2")
                plt.xlabel('Charge PC1',horizontalalignment='left')
                plt.ylabel('Charge PC2',horizontalalignment='left')
                
                plt.draw()

            
        if self.do_ipimb :

            if len(self.IPM_fex_channels)>0 :

                # numpy arrays
                xaxis = np.arange( 0, len(self.IPM_fex_channels) ) 
                arrayCh = np.float_(self.IPM_fex_channels)
                arrayXY = np.float_(self.IPM_fex_position)               
                arraySm = np.float_(self.IPM_fex_sum)

                # store for later (ipython)
                self.data["SharedIpimb"].fex_channels = arrayCh
                self.data["SharedIpimb"].fex_position = arrayXY
                self.data["SharedIpimb"].fex_sum = arraySm

                # make figure
                fig = plt.figure(num=(fignum+4), figsize=(12,4) ) 
                fig.clf()
                fig.subplots_adjust(wspace=0.4, hspace=0.4)
                fig.suptitle("BldInfo:SharedIpimb data " + suptitle)
                
                ax1 = fig.add_subplot(1,3,1)
                plt.hist(arrayCh[:,0], 60, histtype='stepfilled', color='r', label='Ch0')
                plt.hist(arrayCh[:,1], 60, histtype='stepfilled', color='b', label='Ch1')
                plt.hist(arrayCh[:,2], 60, histtype='stepfilled', color='y', label='Ch2')
                plt.hist(arrayCh[:,3], 60, histtype='stepfilled', color='m', label='Ch3')
                plt.title("IPIMB Channels")
                plt.xlabel('Channels',horizontalalignment='left') # the other right
                leg = ax1.legend()
                
                ax2 = fig.add_subplot(1,3,2)
                plt.scatter(arrayXY[:,0],arrayXY[:,1])
                plt.title("Beam position")
                plt.xlabel('Beam position X',horizontalalignment='left')
                plt.ylabel('Beam position Y',horizontalalignment='left')
                
                ax3 = fig.add_subplot(1,3,3)
                plt.plot(xaxis,arraySm)
                plt.ylabel('Sum of channels',horizontalalignment='left') # the other right
                plt.xlabel('Shot number',horizontalalignment='left') # the other right
                plt.title("Sum vs. time")
            
                plt.draw()

