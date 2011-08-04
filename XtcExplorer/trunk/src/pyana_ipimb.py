#
# ipimb.py: plot beamline data
#
# 

import numpy as np
import matplotlib.pyplot as plt

from   pypdsdata import xtc
from utilities import PyanaOptions
from utilities import IpimbData



# analysis class declaration
class  pyana_ipimb ( object ) :
    
    def __init__ ( self,
                   sources = None,
                   variables = "fex:pos fex:sum fex:channels",
                   # "raw:channels raw:voltages"
                   # "fex:ch0 fex:ch1 fex:ch2 fex:ch3"
                   # "raw:ch0 raw:ch1 raw:ch2 raw:ch3"
                   # "raw:ch0volt raw:ch1volt raw:ch2volt raw:ch2volt"
                   plot_every_n = "0",
                   accumulate_n    = "0",
                   fignum = "1" ) :
        """
        @param sources           list of IPIMB addresses
        @param variables         list of variables to plot
        @param plot_every_n      Zero (don't plot until the end), or N (int, plot every N event)
        @param accumulate_n      Accumulate all (0) or reset the array every n shots
        @param fignum            matplotlib figure number
        """

        # initialize data
        opt = PyanaOptions()
        self.sources = opt.getOptStrings(sources)
        print "pyana_ipimb, %d sources: " % len(self.sources)
        for source in self.sources :
            print "  ", source

        self.variables = opt.getOptStrings(variables)
        print "pyana_ipimb variables to plot:"
        for var in self.variables:
            print "  ", var
            
        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.accumulate_n = opt.getOptInteger(accumulate_n)
        self.mpl_num = opt.getOptInteger(fignum)

        # other
        self.n_shots = None
        self.accu_start = None
        
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
            
    def event ( self, evt, env ) :

        self.n_shots+=1

        if evt.get('skip_event') :
            return

        # IPM diagnostics, for saturation and low count filtering
        for source in self.sources :

            # -------- TEST
            # BldDataIpimb / SharedIpimb
            ipm = evt.get(xtc.TypeId.Type.Id_SharedIpimb, source )
            #ipm = evt.getSharedIpimbValue("HFX-DG3-IMB-02")
            if ipm :
                self.raw_ch[source].append( [ipm.ipimbData.channel0(),
                                             ipm.ipimbData.channel1(),
                                             ipm.ipimbData.channel2(),
                                             ipm.ipimbData.channel3()] )

                self.raw_ch_volt[source].append( [ipm.ipimbData.channel0Volts(),
                                             ipm.ipimbData.channel1Volts(),
                                             ipm.ipimbData.channel2Volts(),
                                             ipm.ipimbData.channel3Volts()] )

                self.fex_sum[source].append( ipm.ipmFexData.sum )
                self.fex_channels[source].append( ipm.ipmFexData.channel )
                self.fex_position[source].append( [ipm.ipmFexData.xpos, ipm.ipmFexData.ypos] )
            else :

                # raw data
                ipmRaw = evt.get(xtc.TypeId.Type.Id_IpimbData, source )
                if ipmRaw :
                    ch = [ipmRaw.channel0(),
                          ipmRaw.channel1(),
                          ipmRaw.channel2(),
                          ipmRaw.channel3() ]
                
                    self.raw_ch[source].append(ch)
                
                    ch_volt = [ipmRaw.channel0Volts(),
                               ipmRaw.channel1Volts(),
                               ipmRaw.channel2Volts(),
                               ipmRaw.channel3Volts() ]
                    
                    self.raw_ch_volt[source].append( ch_volt )
            
                
                else :
                    print "pyana_ipimb: No IpimbData from %s found in event %d" % (source,self.n_shots)
                    self.raw_ch[source].append( [-1,-1,-1,-1] )
                    self.raw_ch_volt[source].append( [-1,-1,-1,-1] ) 

                # feature-extracted data
                ipmFex = evt.get(xtc.TypeId.Type.Id_IpmFex, source )

                if ipmFex :
                    self.fex_sum[source].append( ipmFex.sum )
                    self.fex_channels[source].append( ipmFex.channel )
                    self.fex_position[source].append( [ipmFex.xpos, ipmFex.ypos] )
                else :
                    print "pyana_ipimb: No IpmFex from %s found in event %d" % (source, self.n_shots)
                    self.fex_sum[source].append( -1 )
                    self.fex_channels[source].append( [-1,-1,-1,-1] ) 
                    self.fex_position[source].append( [-1,-1] ) 


        # ----------------- Plotting ---------------------
        if self.plot_every_n != 0 and (self.n_shots%self.plot_every_n)==0 :

            header = "DetInfo:IPIMB data shots %d-%d" % (self.accu_start, self.n_shots)
            self.make_plots(title=header)

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

        # ----------------- Plotting ---------------------
        header = "DetInfo:IPIMB data shots %d-%d" % (self.accu_start, self.n_shots)
        self.make_plots(title=header)

        # flag for pyana_plotter
        evt.put(True, 'show_event')

        # convert dict to a list:
        data_ipimb = []
        for source in self.sources :
            data_ipimb.append( self.data[source] )
            # give the list to the event object
            evt.put( data_ipimb, 'data_ipimb' )

        
    def make_plots(self, title = ""):

        if self.n_shots == self.accu_start: 
            print "Cannot do ", title
            return

        # -------- Begin: move this to beginJob
        """ This part should move to begin job, but I can't get
        it to update the plot in SlideShow mode when I don't recreate
        the figure each time. Therefore plotting is slow... 
        """
        ncols = len(self.variables)
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
            if "fex:sum" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array, 60, histtype='stepfilled', color='r', label='Fex Sum')
                plt.title(source)
                plt.xlabel('Sum of channels',horizontalalignment='left') # the other right
                i+=1
            self.data[source].fex_sum = array
 

            array = np.float_(self.fex_channels[source])
            if "fex:channels" in self.variables:
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
            if "fex:ch0" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,0], 60, histtype='stepfilled', color='r', label='Ch0')
                plt.title(source)
                plt.xlabel('Ch0 Fex',horizontalalignment='left') # the other right
                i+=1
            if "fex:ch1" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,1], 60, histtype='stepfilled', color='r', label='Ch1')
                plt.title(source)
                plt.xlabel('Ch1 Fex',horizontalalignment='left') # the other right
                i+=1
            if "fex:ch2" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,2], 60, histtype='stepfilled', color='r', label='Ch2')
                plt.title(source)
                plt.xlabel('Ch2 Fex',horizontalalignment='left') # the other right
                i+=1
            if "fex:ch3" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,3], 60, histtype='stepfilled', color='r', label='Ch3')
                plt.title(source)
                plt.xlabel('Ch3 Fex',horizontalalignment='left') # the other right
                i+=1
            
            self.data[source].fex_channels = array

            array = np.float_(self.raw_ch[source])
            if "raw:channels" in self.variables:
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
            if "raw:ch0" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,0], 60, histtype='stepfilled', color='r', label='Ch0')
                plt.title(source)
                plt.xlabel('Ch0 Raw',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch1" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,1], 60, histtype='stepfilled', color='r', label='Ch1')
                plt.title(source)
                plt.xlabel('Ch1 Raw',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch2" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,2], 60, histtype='stepfilled', color='r', label='Ch2')
                plt.title(source)
                plt.xlabel('Ch2 Raw',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch3" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,3], 60, histtype='stepfilled', color='r', label='Ch3')
                plt.title(source)
                plt.xlabel('Ch3 Raw',horizontalalignment='left') # the other right
                i+=1
            self.data[source].raw_channels = array

            array = np.float_(self.raw_ch_volt[source])
            if "raw:voltage" in self.variables:
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
            if "raw:ch0" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,0], 60, histtype='stepfilled', color='r', label='Ch0')
                plt.title(source)
                plt.xlabel('Ch0 Volt',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch1" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,1], 60, histtype='stepfilled', color='r', label='Ch1')
                plt.title(source)
                plt.xlabel('Ch1 Volt',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch2" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,2], 60, histtype='stepfilled', color='r', label='Ch2')
                plt.title(source)
                plt.xlabel('Ch2 Volt',horizontalalignment='left') # the other right
                i+=1
            if "raw:ch3" in self.variables:
                self.ax[i].clear()
                plt.axes(self.ax[i])
                plt.hist(array[:,3], 60, histtype='stepfilled', color='r', label='Ch3')
                plt.title(source)
                plt.xlabel('Ch3 Volt',horizontalalignment='left') # the other right
                i+=1
            self.data[source].raw_channels_volt = array

            if "fex:pos" in self.variables:
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

                                            
