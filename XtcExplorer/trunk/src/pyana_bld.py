#
# bld.py: plot beamline data
#
# 

import numpy as np
import matplotlib.pyplot as plt
from utilities import PyanaOptions 
from pypdsdata import xtc


# analysis class declaration
class  pyana_bld ( object ) :
    
    def __init__ ( self,
                   do_ebeam        = "False",
                   do_gasdetector  = "False",
                   do_phasecavity  = "False",
                   do_ipimb        = "False",
                   plot_every_n    = None,
                   fignum          = "1" ):
        """
        Initialize data. Parameters:
        @param do_ebeam            Plot data from EBeam object
        @param do_gasdetector      Plot data from GasDetector
        @param do_phasecavity      Plot data from PhaseCavity
        @param do_ipimb            Plot data from Shared Ipimb
        @param plot_every_n        Plot after every N events
        @param fignum              Matplotlib figure number
        """

        # parameters
        opt = PyanaOptions() # convert option string to appropriate type
        self.do_EBeam     = opt.getOptBoolean(do_ebeam)
        self.do_GasDet    = opt.getOptBoolean(do_gasdetector)
        self.do_PC        = opt.getOptBoolean(do_phasecavity)
        self.do_ipimb     = opt.getOptBoolean(do_ipimb)
        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.mpl_num      = opt.getOptInteger(fignum)

        # other
        self.n_shots = None

        # lists to fill numpy arrays
        self.EB_time = []
        self.EB_energies = []
        self.EB_positions = []
        self.EB_angles = []
        self.EB_charge = []
        self.GD_energies = []
        self.PC_data = []
        self.ipm_sum = []
        self.ipm_channels = []
        self.ipm_position = []

    def beginjob ( self, evt, env ) : 
        self.n_shots = 0
        pass

    def event ( self, evt, env ) :

        self.n_shots += 1

        if self.do_EBeam :
            # EBeam object (of type bld.BldDataEBeam or bld.BldDataEBeamV0)
            ebeam = evt.getEBeam()
            if ebeam :
                beamChrg = ebeam.fEbeamCharge 
                beamEnrg = ebeam.fEbeamL3Energy
                beamPosX = ebeam.fEbeamLTUPosX
                beamPosY = ebeam.fEbeamLTUPosY
                beamAngX = ebeam.fEbeamLTUAngX
                beamAngY = ebeam.fEbeamLTUAngY
                #beamPkCr = ebeam.fEbeamPkCurrBC2

                self.EB_time.append( evt.getTime().seconds() + 1.0e-9*evt.getTime().nanoseconds() )
                self.EB_energies.append(beamEnrg)
                self.EB_positions.append( [beamPosX,beamPosY] )
                self.EB_angles.append( [beamAngX, beamAngY] )
                self.EB_charge.append( beamChrg )
            else :
                print "No EBeam object found"

            
        if self.do_GasDet :
            # returns array of 4 numbers obtained from the bld.BldDataFEEGasDetEnergy object
            fee_energy_array = evt.getFeeGasDet()

            if fee_energy_array is None :
                print "No Gas Detector dataobject found"
            else :
                # fee_energy_array is a 4-number vector
                self.GD_energies.append( fee_energy_array )

            
        if self.do_PC:
            # PhaseCavity object (of type bld.BldDataPhaseCavity)
            pc = evt.getPhaseCavity()
            if pc :
                pcFitTime1 = pc.fFitTime1
                pcFitTime2 = pc.fFitTime2
                pcCharge1 = pc.fCharge1
                pcCharge2 = pc.fCharge2
                self.PC_data.append( [pcFitTime1, pcFitTime2, pcCharge1, pcCharge2] )
            else :
                print "No Phase Cavity data object found"

        if self.do_ipimb : # BldDataIpimb / SharedIpimb
            ipm = evt.get(xtc.TypeId.Type.Id_SharedIpimb )
            if ipm :
                self.ipm_channels.append( ipm.ipmFexData.channel )
                self.ipm_sum.append( ipm.ipmFexData.sum )
                self.ipm_position.append( [ipm.ipmFexData.xpos, ipm.ipmFexData.ypos] )
            else :
                pass
            
        if self.plot_every_n != 0:
            if (self.n_shots%self.plot_every_n)==0 :
                #print "Shot#%d ... plotting " % self.n_shots
                self.make_plots(self.mpl_num, suptitle="Accumulated up to Shot#%d"%self.n_shots)
                                                
            
                
    def endjob( self, env ) :
        
        print "EndJob has been reached"

        self.make_plots(fignum=self.mpl_num)


    def make_plots(self, fignum = 1, suptitle = ""):
        
        if self.do_EBeam :

            fig = plt.figure(num=(fignum+1), figsize=(8,8) )
            fig.clf()
            fig.subplots_adjust(wspace=0.4, hspace=0.4)
            fig.suptitle("BldInfo:EBeam accumulated up to shot #%d"%self.n_shots)

            ax1 = fig.add_subplot(221)
            energies = np.float_(self.EB_energies)
            plt.plot(energies)
            plt.title("Beam Energy")
            plt.xlabel('Datagram record',horizontalalignment='left') # the other right
            plt.ylabel('Beam Energy',horizontalalignment='right')

            ax2 = fig.add_subplot(222)
            positions = np.float_(self.EB_positions)
            angles = np.float_(self.EB_angles)
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
            charge = np.float_(self.EB_charge)
            n, bins, patches = plt.hist(self.EB_charge, 100, normed=1, histtype='stepfilled')
            plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
            plt.title('Beam Charge')
            plt.xlabel('Beam Charge',horizontalalignment='left') # the other right

            plt.draw()

        if self.do_GasDet :

            fig = plt.figure(num=(fignum+2), figsize=(8,8) )
            fig.clf()
            fig.subplots_adjust(wspace=0.4, hspace=0.4)
            fig.suptitle("BldInfo:FEEGasDetEnergy data accumulated up to shot #%d"%self.n_shots)

            array = np.float_(self.GD_energies)

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

            if len(self.PC_data)==0:
                print "No PhaseCavity data found, skip plotting"
            else:
                fig = plt.figure(num=(fignum+3), figsize=(12,8) )
                fig.clf()
                fig.subplots_adjust(wspace=0.4, hspace=0.4)
                fig.suptitle("BldInfo:PhaseCavity data accumulated up to shot #%d"%self.n_shots)
            
                array = np.float_(self.PC_data)

                ax1 = fig.add_subplot(231)
                n, bins, patches = plt.hist(array[:,0], 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'r', 'alpha', 0.75)
                plt.title('Time PC1')
                plt.xlabel('Time PC1',horizontalalignment='left')
                unit = bins[1] - bins[0]
                x1min, x1max = (bins[0]-unit), (bins[-1]+unit)
                plt.xlim(x1min,x1max)
             
                ax2 = fig.add_subplot(232)
                n, bins, patches = plt.hist(array[:,1], 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'r', 'alpha', 0.75)
                plt.title('Time PC2')
                plt.xlabel('Time PC2',horizontalalignment='left')
                unit = bins[1] - bins[0]
                x2min, x2max = (bins[0]-unit), (bins[-1]+unit)
                plt.xlim(x2min,x2max)

                ax3 = fig.add_subplot(233)
                plt.scatter(array[:,0],array[:,1])
                plt.title("Time PC1 vs. Time PC2")
                plt.xlabel('Time PC1',horizontalalignment='left')
                plt.ylabel('Time PC2',horizontalalignment='left')
                plt.xlim(x1min,x1max)
                plt.ylim(x2min,x2max)
                
                ax4 = fig.add_subplot(234)
                n, bins, patches = plt.hist(array[:,2], 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'b', 'alpha', 0.75)
                plt.title('Charge PC1')
                plt.xlabel('Charge PC1',horizontalalignment='left')
             
                ax5 = fig.add_subplot(235)
                n, bins, patches = plt.hist(array[:,3], 60,histtype='stepfilled')
                plt.setp(patches,'facecolor', 'b', 'alpha', 0.75)
                plt.title('Charge PC2')
                plt.xlabel('Charge PC2',horizontalalignment='left')
                
                ax6 = fig.add_subplot(236)
                plt.scatter(array[:,2],array[:,3])
                plt.title("Charge PC1 vs. Charge PC2")
                plt.xlabel('Charge PC1',horizontalalignment='left')
                plt.ylabel('Charge PC2',horizontalalignment='left')
                
                plt.draw()
            
        if self.do_ipimb :
            if len(self.ipm_channels)==0 :
                print "No SharedIpimb data object found... skip plotting"
            else :
                fig = plt.figure(num=(fignum+4), figsize=(12,5) ) 
                fig.clf()
                fig.subplots_adjust(wspace=0.4, hspace=0.4)
                fig.suptitle("BldInfo:SharedIpimb data accumulated up to shot #%d"%self.n_shots)
                
                xaxis = np.arange( 0, len(self.ipm_channels) )
            
                ax1 = fig.add_subplot(1,3,1)
                array = np.float_(self.ipm_channels)
                plt.hist(array[:,0], 60, histtype='stepfilled', color='r', label='Ch0')
                plt.hist(array[:,1], 60, histtype='stepfilled', color='b', label='Ch1')
                plt.hist(array[:,2], 60, histtype='stepfilled', color='y', label='Ch2')
                plt.hist(array[:,3], 60, histtype='stepfilled', color='m', label='Ch3')
                plt.title("IPIMB Channels")
                plt.xlabel('Channels',horizontalalignment='left') # the other right
                leg = ax1.legend()
                
                ax2 = fig.add_subplot(1,3,2)
                array2 = np.float_(self.ipm_position)
                plt.scatter(array2[:,0],array2[:,1])
                plt.title("Beam position")
                plt.xlabel('Beam position X',horizontalalignment='left')
                plt.ylabel('Beam position Y',horizontalalignment='left')
                
                ax3 = fig.add_subplot(1,3,3)
                array = np.float_(self.ipm_sum)
                plt.plot(xaxis,array)
                plt.ylabel('Sum of channels',horizontalalignment='left') # the other right
                plt.xlabel('Shot number',horizontalalignment='left') # the other right
                plt.title("Sum vs. time")
            
                plt.draw()

