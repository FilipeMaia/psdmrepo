#--------------------------------------------------------------------------
# File and Version Information:

# Description:
#  Pyana user analysis module xppt_delayscan...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $

@author Ingrid Ofte
"""

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
import scipy.io
import h5py

from pypdsdata.xtc import TypeId


#---------------------
#  Class definition --
#---------------------
class xppt_delayscan (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self,
                   controlpv = "",
                   ipimb_sig = "",
                   ipimb_norm = "",
                   threshold = "",
                   outputfile = "point_scan_delay.npy" ):
        """Class constructor.
        The parameters to the constructor are passed from pyana configuration file.
        If parameters do not have default values  here then the must be defined in
        pyana.cfg. All parameters are passed as strings, convert to correct type before use.

        @param controlpv
        @param ipimb_sig
        @param ipimb_norm
        @param threshold
        """

        # parameters
        self.controlpv = controlpv
        self.ipimb_sig = ipimb_sig
        self.ipimb_norm = ipimb_norm
        self.threshold = None
        if threshold != "":
            self.threshold = float(threshold) # convert to floating point value (from string)
        self.outputfile = outputfile

        # other variables
        self.nevents = 0
        self.ncalib = 0

        # list for plotting
        self.h_ipm_rsig = [] # raw signal
        self.h_ipm_nsig = [] # normalized signal
        self.h_delaytime = [] # delaytime
        

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job. It should
        do a one-time initialization possible extracting values from event
        data (which is a Configure object) or environment.

        @param evt    event data object
        @param env    environment object
        """

        # Preferred way to log information is via logging package
        logging.info( "xppt_delayscan.beginjob() called" )

        # get configuration info here

        # control.ConfigV1 element
        ctrl_config = env.getConfig(TypeId.Type.Id_ControlConfig)

        # print some info to the screen:
        print "PV control config in this xtc file: "
        print "     uses duration? ", ctrl_config.uses_duration()
        print "     uses events? ", ctrl_config.uses_events()
        print "     duration: ", ctrl_config.duration()
        print "     events: ", ctrl_config.events()
        print "     size: ", ctrl_config.size()
        print 
        print "     npvControls: ", ctrl_config.npvControls()
        for ic in range (0, ctrl_config.npvControls() ):
            cpv = ctrl_config.pvControl(ic)            
            print "          name: ", cpv.name(), cpv.value()
        print
        print "     npvMonitors: ", ctrl_config.npvMonitors()
        for ic in range (0, ctrl_config.npvMonitors() ):
            cpv = ctrl_config.pvControl(ic)            
            print "          name: ", cpv.name(), cpv.value()
        

        # initialize stuff
        self.pv = []

    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning of the new run.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "xppt_delayscan.beginrun() called" )



    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "xppt_delayscan.begincalibcycle() called" )

        # if more than one calibcycle, initialize calib-cycle quantities (e.g. averages) here. 
        self.ncalib += 1


        # fetch it from the current env again
        ctrl_config = env.getConfig(TypeId.Type.Id_ControlConfig)

        for ic in range (0, ctrl_config.npvControls() ):
            cpv = ctrl_config.pvControl(ic)
            if cpv.name()==self.controlpv:
                # store the value
                self.pv.append( cpv.value() )
                #print "%s value at calibcycle #%d: %s" % (cpv.name(), self.ncalib, cpv.value())


    ###########################################################################################

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        
        self.nevents += 1

        # ----------------------------------------------
        # Retreive the Ipimb information for normalization and filtering
        ipmN_raw = evt.get(TypeId.Type.Id_IpimbData, self.ipimb_norm )
        ipmN_fex = evt.get(TypeId.Type.Id_IpmFex, self.ipimb_norm )

        try:
            # Here, we use the sum of FEX values for filter and normalization
            ipm_norm = ipmN_fex.sum

            if self.threshold is not None and ipm_norm < self.threshold:
                print "Failed ipimb threshold", ipm_norm, self.threshold
                return

        except:
            print "No %s found in shot#%d" %( self.ipimb_norm, self.nevent)
            return


        # -----------------------------------------------
        # Phase cavity
        pc = evt.getPhaseCavity()
        try:
            phasecav1 = pc.fFitTime1
            phasecav2 = pc.fFitTime2
            charge1 = pc.fCharge1
            charge2 = pc.fCharge2
        except:
            print "No phase cavity found in shot#", self.nevents
            return


        # -----------------------------------------------
        # Compute the delay time from the latest Control PV value and this events' phase cavity value
        delaytime = self.pv[self.ncalib-1] + phasecav1*1e3
        
        

        # -----------------------------------------------
        # Get the IPIMB used for signal
        ipmS_raw = evt.get(TypeId.Type.Id_IpimbData, self.ipimb_sig )
        ipmS_fex = evt.get(TypeId.Type.Id_IpmFex, self.ipimb_sig )

        try:
            # Here, we use channel 1 (counting from 0) as the signal
            ipm_sig = ipmS_fex.channel[1]

        except:
            print "No %s found in shot#%d" %( self.ipimb_sig, self.nevent)
            return


        self.h_ipm_rsig.append( ipm_sig )
        self.h_ipm_nsig.append( ipm_sig/ipm_norm )
        self.h_delaytime.append( delaytime )


            
            
    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        
        logging.info( "xppt_delayscan.endcalibcycle() called" )

    def endrun( self, env ) :
        """This optional method is called if present at the end of the run.
        
        @param env    environment object
        """
        
        logging.info( "xppt_delayscan.endrun() called" )

    def endjob( self, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        
        logging.info( "xppt_delayscan.endjob() called" )

        dt_min, dt_max, dt_entries = min(self.h_delaytime), max(self.h_delaytime), len(self.h_delaytime)
        rs_min, rs_max, rs_entries = min(self.h_ipm_rsig), max(self.h_ipm_rsig), len(self.h_ipm_rsig)
        ns_min, ns_max, ns_entries = min(self.h_ipm_nsig), max(self.h_ipm_nsig), len(self.h_ipm_nsig)


        print "End of job... "
        print "have collected: "
        print "       - delay time: %d entries, from %.2f to %.2f " % (dt_entries, dt_min, dt_max)
        print "       - raw signal: %d entries, from %.5f to %.5f " % (rs_entries, rs_min, rs_max)
        print "       - normalized: %d entries, from %.5f to %.5f " % (ns_entries, ns_min, ns_max)

        # ----------------------------------------------
        # Saving arrays

        print "Saving to file...", self.outputfile
        if self.outputfile.find(".mat")>=0:
            # ...................
            # save as matlab file
            scipy.io.savemat(self.outputfile,
                             mdict={'delaytime' : np.array(self.h_delaytime),
                                    'rawsignal' : np.array(self.h_ipm_rsig),
                                    'normsignal': np.array(self.h_ipm_nsig) }
                             )

                             
        elif self.outputfile.find(".h5")>=0:
            # ...................
            # save as HDF5 file
            ofile = h5py.File(self.outputfile,'w')
            group = ofile.create_group("MyGroup")
            group.create_dataset('delaytime',data=np.array(self.h_delaytime))
            group.create_dataset('rawsignal',data=np.array(self.h_ipm_rsig))
            group.create_dataset('normsignal',data=np.array(self.h_ipm_nsig))
            ofile.close()


        elif self.outputfile.find(".npy")>=0:
            # ...................
            # save as numpy binary file
            np.save(self.outputfile,
                    np.array( zip(self.h_delaytime,self.h_ipm_rsig,self.h_ipm_nsig) ))


        else :
            # ...................
            # Save as ascii file
            np.savetxt(self.outputfile,
                       np.array( zip(self.h_delaytime,self.h_ipm_rsig,self.h_ipm_nsig) ))


        # ---------------------------------------------------
        # Plotting

        print "Start plotting..."


        # equal-spaced bins
        delay_bins = np.linspace( min(self.h_delaytime), max(self.h_delaytime), 10+1)

        plt.figure()
        plt.title("")

        #plt.hist( self.h_delaytime, bins=delay_bins)
        #plt.plot( self.h_delaytime )
        plt.scatter( np.array(self.h_delaytime), np.array(self.h_ipm_nsig) )
        plt.show()

