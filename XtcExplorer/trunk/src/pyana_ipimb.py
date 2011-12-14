#
# ipimb.py: plot beamline data
#
# 

import numpy as np
import matplotlib.pyplot as plt

from   pypdsdata import xtc
from utilities import PyanaOptions
from utilities import IpimbData
from utilities import Plotter

# analysis class declaration
class  pyana_ipimb ( object ) :
    
    def __init__ ( self,
                   source = None,
                   sources = None,
                   quantities = "fex:pos fex:sum fex:channels",
                   # "raw:channels raw:voltages"
                   # "fex:ch0 fex:ch1 fex:ch2 fex:ch3"
                   # "raw:ch0 raw:ch1 raw:ch2 raw:ch3"
                   # "raw:ch0volt raw:ch1volt raw:ch2volt raw:ch2volt"
                   plot_every_n = "0",
                   accumulate_n    = "0",
                   fignum = "1" ) :
        """
        @param sources           list of IPIMB addresses
        @param quantities         list of quantities to plot
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

        self.plotter = None
                
        
        # lists to fill numpy arrays
        self.initlists()



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
        self.n_shots = 0
        self.accu_start = 0
        
        self.data = {}
        for source in self.sources :
            self.data[source] = IpimbData( source ) 

            # just for information:
            config = env.getConfig( xtc.TypeId.Type.Id_IpimbConfig , source )
            if config is not None:
                print "IPIMB %s configuration info: "%source
                print "   Acquisition window (ns) ", config.resetLength()
                print "   Reset delay (ns) ", config.resetDelay()
                print "   Reference voltage ", config.chargeAmpRefVoltage()
                print "   Diode bias voltage ", config.diodeBias()
                print "   Sampling delay (ns) ", config.trigDelay()
                #print "   trigger counter ", config.triggerCounter()
                #print "   serial ID ", config.serialID()
                #print "   charge amp range ", config.chargeAmpRange()
                #print "   calibration range ", config.calibrationRange()
                #print "   calibration voltage ", config.calibrationVoltage()
                #print "   status ", config.status()
                #print "   errors ", config.errors()
                #print "   calStrobeLength ", config.calStrobeLength()
                #try: # These are only for ConfigV2
                #    print "   trigger ps delay ", config.trigPsDelay()
                #    print "   adc delay ", config.adcDelay()
                #except:
                #    pass


        self.plotter = Plotter()
        self.plotter.settings(7,7) # set default frame size
                
            
    def event ( self, evt, env ) :

        self.n_shots+=1

        if evt.get('skip_event') :
            return

        # IPM diagnostics, for saturation and low count filtering
        for source in self.sources :

            ipm_raw = None
            ipm_fex = None

            # -------------------------------------------
            # fetch the data object from the pyana event 
            # -------------------------------------------

            # try Shared IPIMB first
            ipm = evt.get(xtc.TypeId.Type.Id_SharedIpimb, source )
            if ipm is not None:
                ipm_raw = ipm.ipimbData
                ipm_fex = ipm.ipmFexData
            else: 
                # try to get the other data types for IPIMBs 
                ipm_raw = evt.get(xtc.TypeId.Type.Id_IpimbData, source )
                ipm_fex = evt.get(xtc.TypeId.Type.Id_IpmFex, source )

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

            header = "DetInfo:IPIMB data shots %d-%d" % (self.accu_start, self.n_shots)
            self.old_make_plots(title=header)

            # flag for pyana_plotter
            evt.put(True, 'show_event')

            # convert dict to a list:
            data_ipimb = []
            for source in self.sources :
                data_ipimb.append( self.data[source] )
            # give the list to the event object
            evt.put( data_ipimb, 'data_ipimb' )

                        
        # --------- Reset -------------
        if self.accumulate_n!=0 and (self.n_shots%self.accumulate_n)==0 :
            self.resetlists()


    def endjob( self, evt, env ) :

        # only call plotter if this is the main thread
        if (env.subprocess()>0):
            return

        # ----------------- Plotting ---------------------
        header = "DetInfo:IPIMB data shots %d-%d" % (self.accu_start, self.n_shots)
        self.old_make_plots(title=header)

        # flag for pyana_plotter
        evt.put(True, 'show_event')

        # convert dict to a list:
        data_ipimb = []
        for source in self.sources :
            data_ipimb.append( self.data[source] )
            # give the list to the event object
            evt.put( data_ipimb, 'data_ipimb' )

    def make_plots(self,title=""):

        for source in self.sources :

            xaxis = np.arange( self.accu_start, self.n_shots )

            # --------------------------------------
            if "fex:channels" in self.quantities:
                name = "fex:channel(%s)"%source
                title = "%s; %s; %s" % (source, "Event number", "Channel Fex")
                
                # turn the list into nx4 array
                array = np.float_(self.fex_channels[source])

                # turn nx4 array into list of channels, each a numpy array
                channels = [ np.float_(column) for column in array.T.tolist() ]

                # insert x-axis at the beginning of the list
                channels.insert(0,xaxis)

                # add frame to the canvas
                self.plotter.add_frame(name,title, tuple(channels) )


            # --------------------------------------
            # fills a histogram with the sum of channels
            if "fex:sum" in self.quantities:
                name = "fex:sum(%s)"%source
                title = "%s; %s; %s" % (source, "Event number", "Channel Sum")

                array = np.float_(self.fex_sum[source])

                # add frame to the canvas
                self.plotter.add_frame(name, title, (array,), plot_type="hist" )


        newmode = self.plotter.plot_all_frames(ordered=True)
        
    def old_make_plots(self, title = ""):

        if self.n_shots == self.accu_start: 
            print "Cannot do ", title
            return

        # -------- Begin: move this to beginJob
        """ This part should move to begin job, but I can't get
        it to update the plot in SlideShow mode when I don't recreate
        the figure each time. Therefore plotting is slow... 
        """
        ncols = len(self.quantities)
        nrows = len(self.sources)
        height=4.0
        width=height*1.2

        if nrows * height > 12 : height = 12.0/nrows
        if ncols * width > 22 : width = 22.0/ncols

        fig = plt.figure(num=self.mpl_num, figsize=(width*ncols,height*nrows) )
        fig.clf()
 
        fig.subplots_adjust(left=0.1,   right=0.9,
                            bottom=0.1,  top=0.8,
                            wspace=0.3,  hspace=0.3 )
        fig.suptitle(title)

        self.ax = []
        for i in range (0,ncols*nrows):
            self.ax.append( fig.add_subplot(nrows, ncols, i) )
        # -------- End: move this to beginJob

        
        
        i = 0
        for source in self.sources :

            xaxis = np.arange( self.accu_start, self.n_shots )

            array = np.float_(self.fex_sum[source])
            if "fex:sum" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array, 60, histtype='stepfilled', color='r', label='Fex Sum')
                plt.title(source)
                plt.xlabel('Sum of channels',horizontalalignment='left') # the other right
                i+=1
            self.data[source].fex_sum = array
            print "Checking the length of array: ", len(array)

            array = np.float_(self.fex_channels[source])
            if "fex:channels" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.plot(xaxis,array[:,0], label='Ch0')
                plt.plot(xaxis,array[:,1], label='Ch1')
                plt.plot(xaxis,array[:,2], label='Ch2')
                plt.plot(xaxis,array[:,3], label='Ch3')
                plt.title(source)
                plt.ylabel('Channel Fex',horizontalalignment='left') # the other right
                plt.xlabel('Shot number',horizontalalignment='left') # the other right
                leg = self.ax[i].legend()#('ch0','ch1','ch2','ch3'),'upper center')
                i+=1
            if "fex:ch0" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,0], 60, histtype='stepfilled', color='r', label='Ch0')
                plt.title(source)
                plt.xlabel('Ch0 Fex',horizontalalignment='left') # the other right
                i+=1
            if "fex:ch1" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,1], 60, histtype='stepfilled', color='r', label='Ch1')
                plt.title(source)
                plt.xlabel('Ch1 Fex',horizontalalignment='left') # the other right
                i+=1
            if "fex:ch2" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,2], 60, histtype='stepfilled', color='r', label='Ch2')
                plt.title(source)
                plt.xlabel('Ch2 Fex',horizontalalignment='left') # the other right
                i+=1
            if "fex:ch3" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,3], 60, histtype='stepfilled', color='r', label='Ch3')
                plt.title(source)
                plt.xlabel('Ch3 Fex',horizontalalignment='left') # the other right
                i+=1
            
            self.data[source].fex_channels = array

            array = np.float_(self.raw_ch[source])
            if "raw:channels" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.plot(xaxis,array[:,0], label='Ch0')
                plt.plot(xaxis,array[:,1], label='Ch1')
                plt.plot(xaxis,array[:,2], label='Ch2')
                plt.plot(xaxis,array[:,3], label='Ch3')
                plt.title(source)
                plt.ylabel('Channel Raw',horizontalalignment='left') # the other right
                plt.xlabel('Shot number',horizontalalignment='left') # the other right
                leg = self.ax[i].legend()#('ch0','ch1','ch2','ch3'),'upper center')
                i+=1
            if "raw:ch0" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,0], 60, histtype='stepfilled', color='r', label='Ch0')
                plt.title(source)
                plt.xlabel('Ch0 Raw',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch1" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,1], 60, histtype='stepfilled', color='r', label='Ch1')
                plt.title(source)
                plt.xlabel('Ch1 Raw',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch2" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,2], 60, histtype='stepfilled', color='r', label='Ch2')
                plt.title(source)
                plt.xlabel('Ch2 Raw',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch3" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,3], 60, histtype='stepfilled', color='r', label='Ch3')
                plt.title(source)
                plt.xlabel('Ch3 Raw',horizontalalignment='left') # the other right
                i+=1
            self.data[source].raw_channels = array

            array = np.float_(self.raw_ch_volt[source])
            if "raw:voltage" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.plot(xaxis,array[:,0], label='Ch0')
                plt.plot(xaxis,array[:,1], label='Ch1')
                plt.plot(xaxis,array[:,2], label='Ch2')
                plt.plot(xaxis,array[:,3], label='Ch3')
                plt.title(source)
                plt.ylabel('Channel Voltage',horizontalalignment='left') # the other right
                plt.xlabel('Shot number',horizontalalignment='left') # the other right
                leg = self.ax[i].legend()#('ch0','ch1','ch2','ch3'),'upper center')
                i+=1
            if "raw:ch0" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,0], 60, histtype='stepfilled', color='r', label='Ch0')
                plt.title(source)
                plt.xlabel('Ch0 Volt',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch1" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,1], 60, histtype='stepfilled', color='r', label='Ch1')
                plt.title(source)
                plt.xlabel('Ch1 Volt',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch2" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,2], 60, histtype='stepfilled', color='r', label='Ch2')
                plt.title(source)
                plt.xlabel('Ch2 Volt',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch3" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,3], 60, histtype='stepfilled', color='r', label='Ch3')
                plt.title(source)
                plt.xlabel('Ch3 Volt',horizontalalignment='left') # the other right
                i+=1
            self.data[source].raw_channels_volt = array

            if "fex:pos" in self.quantities:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                array2 = np.float_(self.fex_position[source])
                plt.scatter(array2[:,0],array2[:,1])
                plt.title(source)
                plt.xlabel('Beam position X',horizontalalignment='left')
                plt.ylabel('Beam position Y',horizontalalignment='left')
                i+=1
                self.data[source].fex_position = array2

        plt.draw()

                                            
