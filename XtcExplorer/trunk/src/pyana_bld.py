#
# bld.py: plot beamline data
#
# 

import numpy as np
import logging

from pypdsdata import xtc

from utilities import PyanaOptions 
from utilities import BldData


# analysis class declaration
class  pyana_bld ( object ) :
    
    def __init__ ( self,
                   do_ebeam        = "False",
                   do_gasdetector  = "False",
                   do_phasecavity  = "False",
                   do_ipimb        = "False",
                   plot_every_n    = None,
                   accumulate_n    = "0",
                   fignum          = "1"):
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

        self.doPlot = (env.subprocess()<1) and (self.plot_every_n != 0)

    def event ( self, evt, env ) :
        self.n_shots += 1

        do_plot = self.doPlot and (self.n_shots%self.plot_every_n)==0 

#        # Evr (Event receiver data)
#        evrdata = evt.getEvrData("NoDetector-0|Evr-0")
#        ecodes = []
#        for i in range (evrdata.numFifoEvents()):
#            ecodes.append( evrdata.fifoEvent(i).EventCode )
#        print "Event codes in shot #%d : %s "%(self.n_shots, str(ecodes))

        if self.do_EBeam :
            # EBeam object (of type bld.BldDataEBeam or bld.BldDataEBeamV0)
            ebeam = evt.get(xtc.TypeId.Type.Id_EBeam);
            self.hadEB = True
            if ebeam :
                if env.fwkName() == "psana":
                    beamDmgM = ebeam.damageMask()
                    beamChrg = ebeam.ebeamCharge()
                    beamEnrg = ebeam.ebeamL3Energy()
                    beamPosX = ebeam.ebeamLTUPosX()
                    beamPosY = ebeam.ebeamLTUPosY()
                    beamAngX = ebeam.ebeamLTUAngX()
                    beamAngY = ebeam.ebeamLTUAngY()
                    #beamPkCr = ebeam.ebeamPkCurrBC2()
                else:
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
            fee_energy = evt.get(xtc.TypeId.Type.Id_FEEGasDetEnergy)
            if fee_energy is not None:
                if env.fwkName() == "psana":
                    fee_energy = [fee_energy.f_11_ENRC(),
                                        fee_energy.f_12_ENRC(),
                                        fee_energy.f_21_ENRC(),
                                        fee_energy.f_22_ENRC()]
                else:
                    fee_energy = [fee_energy.f_11_ENRC,
                                        fee_energy.f_12_ENRC,
                                        fee_energy.f_21_ENRC,
                                        fee_energy.f_22_ENRC]

            if fee_energy is None :
                if self.n_shots < 2 :
                    print "No Gas Detector data object found in shot#%d"%self.n_shots
                if self.hadGD:
                    print "No Gas Detector data object found in shot#%d" % self.n_shots
                    self.GD_energies.append( [0.0,0.0,0.0,0.0] )
            else :
                # fee_energy is a 4-number vector
                self.GD_energies.append( fee_energy )

            
        if self.do_PC:
            # PhaseCavity object (of type bld.BldDataPhaseCavity)
            pc = evt.get(xtc.TypeId.Type.Id_PhaseCavity);
            if pc :
                if env.fwkName() == "psana":
                    self.PC_ftime1.append( pc.fitTime1() )
                    self.PC_ftime2.append( pc.fitTime2() )
                    self.PC_fchrg1.append( pc.charge1() )
                    self.PC_fchrg2.append( pc.charge2() )
                else:
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
            ipm = evt.get(xtc.TypeId.Type.Id_SharedIpimb)
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
                    
            
        # only call plotter if this is the main thread
        if (env.subprocess()>0):
            return

        # ----------------- Plotting --------------------- 
        if do_plot :

            evt.put(True, 'show_event')

            data_bld = self.update_plot_data()
    
            evt.put(data_bld, 'data_bld')

            
        # --------- Reset -------------
        if self.accumulate_n!=0 and (self.n_shots%self.accumulate_n)==0 :
            self.resetlists()
                
                
    def endjob( self, evt, env ) :
        
        print "pyana_bld endjob has been reached, after processing %d events"%self.n_shots

        # ----------------- Plotting ---------------------
        if (env.subprocess()>0):
            return
        
        evt.put(True, 'show_event')
        
        data_bld = self.update_plot_data()
    
        evt.put(data_bld, 'data_bld')
        # flag for pyana_plotter
        evt.put(True, 'show_event')



    def update_plot_data(self):
        # convert lists to arrays and dict to a list:
        data_bld = []

        if 'EBeam' in self.data:
            self.data['EBeam'].shots = np.arange( self.accu_start, self.n_shots ) 
            self.data['EBeam'].energies = np.array(self.EB_energies)
            self.data['EBeam'].positions = np.array(self.EB_positions)
            self.data['EBeam'].angles = np.array(self.EB_angles)
            self.data['EBeam'].charge = np.array(self.EB_charge)
        if 'GasDetector' in self.data:
            self.data['GasDetector'].shots = np.arange( self.accu_start, self.n_shots ) 
            self.data['GasDetector'].energies = np.array( self.GD_energies )
        if 'PhaseCavity' in self.data:
            self.data['PhaseCavity'].shots = np.arange( self.accu_start, self.n_shots )
            self.data['PhaseCavity'].time = np.vstack( \
                ( np.array( self.PC_ftime1), np.array(self.PC_ftime2) )).T
            self.data['PhaseCavity'].charge = np.vstack( \
                ( np.array( self.PC_fchrg1), np.array(self.PC_fchrg2) )).T

        for name,data in self.data.iteritems() :
            #print "name, data = ", name, data            
            data_bld.append( data )
       
        return data_bld
