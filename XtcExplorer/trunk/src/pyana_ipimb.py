#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#   Module pyana_ipimb
#
"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or
part of it, please give an appropriate acknowledgment.

@author Ingrid Ofte
"""

import numpy as np

from pypdsdata.xtc import TypeId
from utilities import PyanaOptions
from utilities import IpimbData

# analysis class declaration
class  pyana_ipimb ( object ) :
    
    def __init__ ( self,
                   source = None,
                   sources = None,
                   quantities = "fex:channels fex:sum",                   
                   plot_every_n = "0",
                   accumulate_n    = "0",
                   fignum = "1" ) :
        """
        @param sources           list of IPIMB addresses
        @param quantities        list of quantities to plot
        @param plot_every_n      Zero (don't plot until the end), or N (int, plot every N event)
        @param accumulate_n      Accumulate all (0) or reset the array every n shots
        @param fignum            matplotlib figure number
        """

        # initialize data
        opt = PyanaOptions()
        if sources is not None:
            self.sources = opt.getOptStrings(sources)
        elif source is not None:
            self.sources = opt.getOptStrings(source)
            
        print "pyana_ipimb, %d sources: " % len(self.sources)
        for source in self.sources :
            print "  ", source

        self.quantities = opt.getOptStrings(quantities)
        print "pyana_ipimb quantities to plot:"
        for var in self.quantities:
            print "  ", var
            
        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.accumulate_n = opt.getOptInteger(accumulate_n)
        self.mpl_num = opt.getOptInteger(fignum)

        # other
        self.n_shots = None
        self.accu_start = None

        # lists to fill numpy arrays
        print "Initializing lists..."
        self.initlists()
        print "done"



    def initlists(self):
        self.fex_sum = {}
        self.fex_channels = {}
        self.fex_position = {}
        self.raw_ch = {}
        self.raw_ch_volt = {}
        for source in self.sources :
            self.fex_sum[source] = list()
            self.fex_channels[source] = list()
            self.fex_position[source] = list()
            self.raw_ch[source] = list()
            self.raw_ch_volt[source] = list()

    def resetlists(self):
        self.accu_start = self.n_shots
        for source in self.sources :
            del self.fex_sum[source][:]
            del self.fex_channels[source][:]
            del self.fex_position[source][:]
            del self.raw_ch[source][:]
            del self.raw_ch_volt[source][:]


    def beginjob ( self, evt, env ) : 
        try:
            env.assert_psana()
            self.psana = True
        except:
            self.psana = False
        self.n_shots = 0
        self.accu_start = 0
        
        self.data = {}
        for source in self.sources :
            print source
            self.data[source] = IpimbData( source ) 

            # just for information:
            if self.psana:
                config = env.getConfig("Psana::Ipimb::Config", source)
            else:
                config = env.getConfig( TypeId.Type.Id_IpimbConfig , source )
            if config is not None:
                print "IPIMB %s configuration info: "%source
                print "   Trigger counter:     0x%lx" % config.triggerCounter()
                print "   serial ID:           0x%lx" %config.serialID()
                print "   Charge amp settings: 0x%x "% config.chargeAmpRange()
                print "   Acquisition window:  %ld ns" % config.resetLength()
                print "   Reset delay:         %d ns"% config.resetDelay()
                print "   Reference voltage:   %f V" % config.chargeAmpRefVoltage()
                print "   Diode bias voltage:  %f V" % config.diodeBias()
                print "   Sampling delay:      %ld ns" % config.trigDelay()
                print "   calibration range:   ", config.calibrationRange()
                print "   calibration voltage: ", config.calibrationVoltage()
                print "   status:  ", config.status()
                print "   errors:  ", config.errors()
                print "   calStrobeLength:     ", config.calStrobeLength()
                try: # These are only for ConfigV2
                    print "   trigger ps delay:    ", config.trigPsDelay()
                    print "   adc delay:           ", config.adcDelay()
                except:
                    pass

                # oppskrift fra Henrik (capacitor setting / gain):
                amprange = config.chargeAmpRange()

                capacitor = { 'V1' : ['1pF', '100pF', '10nF'],
                              'V2' : ["1pF", "4.7pF", "24pF", "120pF", "620pF", "3.3nF", "10nF", "expert"]
                              }

                # Convert chargeAmpRange (16 bits for ConfigV2, 8 bits for ConfigV1) to 4 integers
                version, nbits, mask = 'V2', 4, 0xf
                if str(type(config)).find("ConfigV1")>=0 :
                    version, nbits, mask = 'V1', 2, 0x3

                caval = []
                for i in range(4):
                    caval.append( ( amprange >> nbits*i) & mask )
                    
                self.data[source].gain_settings = [capacitor[version][i] for i in caval]
                print "Capacitor settings for %s diodes: %s" %\
                      (source, self.data[source].gain_settings)


    def event ( self, evt, env ) :

        self.n_shots+=1

        # IPM diagnostics, for saturation and low count filtering
        for source in self.sources :

            ipm_raw = None
            ipm_fex = None

            # -------------------------------------------
            # fetch the data object from the pyana event 
            # -------------------------------------------

            # try Shared IPIMB first
            if self.psana:
                ipm = evt.get("Psana::Bld::BldDataIpimb", source )
            else:
                ipm = evt.get(TypeId.Type.Id_SharedIpimb, source )
            if ipm is not None:
                ipm_raw = ipm.ipimbData
                ipm_fex = ipm.ipmFexData
            else: 
                # try to get the other data types for IPIMBs 
                if self.psana:
                    ipm_raw = evt.get("Psana::Ipimb::Data", source )
                    ipm_fex = evt.get("Psana::Lusi::IpmFex", source )
                else:
                    ipm_raw = evt.get(TypeId.Type.Id_IpimbData, source )
                    ipm_fex = evt.get(TypeId.Type.Id_IpmFex, source )

            # --------------------------------------------------------------
            # filter???
            # --------------------------------------------------------------
            

            # --------------------------------------------------------------
            # store arrays of the data that we want
            # --------------------------------------------------------------

            # ----- raw data -------
            if ipm_raw is not None: 
                self.raw_ch[source].append( [ipm_raw.channel0(),
                                             ipm_raw.channel1(),
                                             ipm_raw.channel2(),
                                             ipm_raw.channel3()] )
                self.raw_ch_volt[source].append( [ipm_raw.channel0Volts(),
                                                  ipm_raw.channel1Volts(),
                                                  ipm_raw.channel2Volts(),
                                                  ipm_raw.channel3Volts()] )
            else :
                #print "pyana_ipimb: no raw data from %s in event %d" % (source,self.n_shots)
                self.raw_ch[source].append( [-1,-1,-1,-1] )
                self.raw_ch_volt[source].append( [-1,-1,-1,-1] ) 


            # ------ fex data -------
            if ipm_fex is not None: 
                self.fex_sum[source].append( ipm_fex.sum )
                self.fex_channels[source].append( ipm_fex.channel )
                self.fex_position[source].append( [ipm_fex.xpos, ipm_fex.ypos] )
            else :
                #print "pyana_ipimb: no fex data from %s in event %d" % (source,self.n_shots)
                self.fex_sum[source].append( -1 )
                self.fex_channels[source].append( [-1,-1,-1,-1] ) 
                self.fex_position[source].append( [-1,-1] ) 
                
            
        # only call plotter if this is the main thread
        if (env.subprocess()>0):
            return


        # ----------------- Plotting ---------------------
        if self.plot_every_n != 0 and (self.n_shots%self.plot_every_n)==0 :

            # flag for pyana_plotter
            evt.put(True, 'show_event')

            data_ipimb = self.update_plot_data()

            # give the list to the event object
            #event_data_ipimb = evt.get('data_ipimb')
            #if event_data_ipimb is not None:
            #    event_data_ipimb.extend( data_ipimb )
            #    data_ipimb = event_data_ipimb
            evt.put( data_ipimb, 'data_ipimb')

        # --------- Reset -------------
        if self.accumulate_n!=0 and (self.n_shots%self.accumulate_n)==0 :
            self.resetlists()


    def endjob( self, evt, env ) :

        # only call plotter if this is the main thread
        if (env.subprocess()>0):
            return

        # flag for pyana_plotter
        evt.put(True, 'show_event')

        data_ipimb = self.update_plot_data()
        # give the list to the event object
        evt.put( data_ipimb, 'data_ipimb' )


    def update_plot_data(self):
            # convert lists to arrays and dict to a list:
            data_ipimb = []
            for source in self.sources :
                self.data[source].fex_sum = np.array(self.fex_sum[source])
                self.data[source].fex_channels = np.array(self.fex_channels[source])
                self.data[source].fex_position = np.array(self.fex_position[source])
                self.data[source].raw_channels = np.array(self.raw_ch[source])
                self.data[source].raw_voltages = np.array(self.raw_ch_volt[source])                
                data_ipimb.append( self.data[source] )
            return data_ipimb
