#
# bld.py: plot beamline data
#
# 

import numpy as np
import matplotlib.pyplot as plt
from utilities import PyanaOptions 

# analysis class declaration
class  pyana_bld ( object ) :
    
    def __init__ ( self,
                   do_ebeam="False",
                   do_gasdetector="False",
                   do_phasecavity="False",
                   plot_every_n = None,
                   fignum = "1" ) :
        """
        Initialize data. Parameters:
        @param do_ebeam            Plot data from EBeam object
        @param do_gasdetector      Plot data from GasDetector
        @param do_phasecavity      Plot data from PhaseCavity
        @param plot_every_n        Plot after every N events
        @param fignum              Matplotlib figure number
        """

        # parameters
        opt = PyanaOptions() # convert option string to appropriate type
        self.do_EBeam     = opt.getOptBoolean(do_ebeam)
        self.do_GasDet    = opt.getOptBoolean(do_gasdetector)
        self.do_PC        = opt.getOptBoolean(do_phasecavity)
        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.mpl_num      = opt.getOptInteger(fignum)

        # other
        self.shot_number = None

        # lists to fill numpy arrays
        #self.EB_time = []
        self.EB_energies = []
        self.EB_positions = []
        self.EB_angles = []
        self.EB_charge = []

        self.GD_energies = []

        self.PC_data = []
        
    def beginjob ( self, evt, env ) : 
        self.shot_number = 0
        pass

    def event ( self, evt, env ) :

        self.shot_number += 1

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

                #self.EB_time.append( evt.getTime().nanoseconds() + evt.getTime().nanoseconds() )
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
                print "No Gas Detector"
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
                print "No Phase Cavity object found"


        if self.plot_every_n is not None:
            if (self.shot_number%self.plot_every_n)==0 :
                print "Shot#%d ... plotting " % self.shot_number
                fignum = self.mpl_num*100
                self.make_plots(fignum, suptitle="Accumulated up to Shot#%d"%self.shot_number)
                                                
            
                
    def endjob( self, env ) :

        fignum = self.mpl_num*100
        self.make_plots(fignum, suptitle="Average of all (%d) events"%self.shot_number)


    def make_plots(self, fignum = 1, suptitle = ""):
        

        if self.do_EBeam :

            fig = plt.figure(num=(fignum+10), figsize=(8,8) )
            fig.clf()
            fig.suptitle(suptitle)

            ax1 = fig.add_subplot(221)
            array = np.float_(self.EB_energies)
            #time = np.float_(self.EB_time)
            #plt.plot(time, array)
            plt.plot(array)
            plt.title('Beam Energy')
            plt.xlabel('Datagram record',horizontalalignment='left') # the other right
            plt.ylabel('Beam Energy',horizontalalignment='right')

            array2 = np.float_(self.EB_positions)
            array3 = np.float_(self.EB_angles)

            ax2 = fig.add_subplot(222)
            plt.scatter(array2[:,0],array3[:,0])
            plt.title('Beam X')
            plt.xlabel('position X',horizontalalignment='left')
            plt.ylabel('angle X',horizontalalignment='left')

            ax3 = fig.add_subplot(223)
            plt.scatter(array2[:,1],array3[:,1])
            plt.title('Beam Y')
            plt.xlabel('position Y',horizontalalignment='left')
            plt.ylabel('angle Y',horizontalalignment='left')

            ax4 = fig.add_subplot(224)
            array4 = np.float_(self.EB_charge)
            n, bins, patches = plt.hist(array4, 100, normed=1, histtype='stepfilled')
            plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
            plt.title('Beam Charge')
            plt.xlabel('Beam Charge',horizontalalignment='left') # the other right

            plt.draw()


        if self.do_GasDet :

            fig = plt.figure(num=(fignum+20), figsize=(8,8) )
            fig.clf()
            fig.suptitle(suptitle)

            array = np.float_(self.GD_energies)

            ax1 = fig.add_subplot(221)
            n, bins, patches = plt.hist(array[:,0], 60,histtype='stepfilled')
            plt.setp(patches,'facecolor', 'r', 'alpha', 0.75)
            plt.title('Gas Detector E0')
            plt.xlabel('GasDetector Energy0',horizontalalignment='left')
            
            ax2 = fig.add_subplot(222)
            n, bins, patches = plt.hist(array[:,1], 60,histtype='stepfilled')
            plt.setp(patches,'facecolor', 'g', 'alpha', 0.75)
            plt.title('Gas Detector E1')
            plt.xlabel('GasDetector Energy1',horizontalalignment='left')

            ax3 = fig.add_subplot(223)
            n, bins, patches = plt.hist(array[:,2], 60,histtype='stepfilled')
            plt.setp(patches,'facecolor', 'b', 'alpha', 0.75)
            plt.title('Gas Detector E2')
            plt.xlabel('GasDetector Energy2',horizontalalignment='left')

            ax4 = fig.add_subplot(224)
            n, bins, patches = plt.hist(array[:,3], 60,histtype='stepfilled')
            plt.setp(patches,'facecolor', 'm', 'alpha', 0.75)
            plt.title('Gas Detector E3')
            plt.xlabel('GasDetector Energy3',horizontalalignment='left')

            plt.draw()


        if self.do_PC :

            fig = plt.figure(num=(fignum+30), figsize=(8,8) )
            fig.clf()
            fig.suptitle(suptitle)
            
            array = np.float_(self.PC_data)

            ax1 = fig.add_subplot(221)
            n, bins, patches = plt.hist(array[:,0], 60,histtype='stepfilled')
            plt.setp(patches,'facecolor', 'r', 'alpha', 0.75)
            plt.title('PhaseCavity FitTime1')
            plt.xlabel('PhaseCavity FitTime1',horizontalalignment='left')
            
            ax2 = fig.add_subplot(222)
            n, bins, patches = plt.hist(array[:,1], 60,histtype='stepfilled')
            plt.setp(patches,'facecolor', 'g', 'alpha', 0.75)
            plt.title('PhaseCavity FitTime2')
            plt.xlabel('PhaseCavity FitTime2',horizontalalignment='left')

            ax3 = fig.add_subplot(223)
            n, bins, patches = plt.hist(array[:,2], 60,histtype='stepfilled')
            plt.setp(patches,'facecolor', 'b', 'alpha', 0.75)
            plt.title('PhaseCavity FitCharge1')
            plt.xlabel('PhaseCavity FitCharge1',horizontalalignment='left')

            ax4 = fig.add_subplot(224)
            n, bins, patches = plt.hist(array[:,3], 60,histtype='stepfilled')
            plt.setp(patches,'facecolor', 'm', 'alpha', 0.75)
            plt.title('PhaseCavity FitChare2')
            plt.xlabel('PhaseCavity FitCharge2',horizontalalignment='left')

            plt.draw()
