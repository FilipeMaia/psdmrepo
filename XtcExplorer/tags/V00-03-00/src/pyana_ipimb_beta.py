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
class  pyana_ipimb_beta ( object ) :
    
    def __init__ ( self,
                   source = None,
                   quantities = "fex:pos fex:sum fex:channels",
                   ):
        """
        @param source             addresse of Ipimb
        @param quantities         list of quantities to plot
                                  fex:pos fex:sum fex:channels
                                  raw:channels raw:voltages
                                  fex:ch0 fex:ch1 fex:ch2 fex:ch3
                                  raw:ch0 raw:ch1 raw:ch2 raw:ch3
                                  raw:ch0volt raw:ch1volt raw:ch2volt raw:ch2volt
        """

        # initialize data
        opt = PyanaOptions()
        self.source = opt.getOptString(source)
            
        self.quantities = opt.getOptStrings(quantities)
        print "pyana_ipimb_beta quantities to plot:"
        for var in self.quantities:
            print "  ", var
            
        # other
        self.n_shots = None
        self.accu_start = None

        self.mydata = IpimbData(source)
        
        # lists to fill numpy arrays
        self.initlists()
        

    def initlists(self):
        if "fex:sum" in self.quantities:
            self.mydata.fex_sum = list()
        if "fex:channels" in self.quantities:
            self.mydata.fex_channels = list()
        if "fex:pos" in self.quantities:
            self.mydata.fex_position = list()
        if "raw:channels" in self.quantities:
            self.raw_channels = list()
        if "raw:voltages" in self.quantities:
            self.raw_voltages = list()


    def resetlists(self):
        self.accu_start = self.n_shots
        for item in dir(self.mydata):
            if item.find('__')>=0 : continue
            attr = getattr(self,item)
            if attr is not None:
                if type(attr)==np.ndarray:
                    del attr[:]

    def beginjob ( self, evt, env ) : 
        self.n_shots = 0
        self.accu_start = 0
        
    def event ( self, evt, env ) :

        self.n_shots+=1
        print self.n_shots

        if evt.get('skip_event') :
            return

        # IPM diagnostics, for saturation and low count filtering
        source = self.source

        ipmShr = evt.get(xtc.TypeId.Type.Id_SharedIpimb, source )
        ipmFex = evt.get(xtc.TypeId.Type.Id_IpmFex, source )
        ipmRaw = evt.get(xtc.TypeId.Type.Id_IpimbData, source )

        if ipmShr:
            # SharedIpimb's contain both raw and fex data
            if self.mydata.fex_sum is not None:
                self.mydata.fex_sum.append( ipmShr.ipmFexData.sum )
            if self.mydata.fex_position is not None:
                self.mydata.fex_position.append( [ipmShr.ipmFexData.xpos, ipmShr.ipmFexData.ypos] )
            if self.mydata.fex_channels is not None: 
                self.mydata.fex_channels.append( ipmShr.ipmFexData.channel )

            if self.mydata.raw_channels is not None:
                self.mydata.raw_channels.append( [ipmShr.ipimbData.channel0(),
                                                  ipmShr.ipimbData.channel1(),
                                                  ipmShr.ipimbData.channel2(),
                                                  ipmShr.ipimbData.channel3()] )
            if self.mydata.raw_voltages is not None:
                self.mydata.raw_voltages.append( [ipmShr.ipimbData.channel0Volts(),
                                                  ipmShr.ipimbData.channel1Volts(),
                                                  ipmShr.ipimbData.channel2Volts(),
                                                  ipmShr.ipimbData.channel3Volts()] )
        else :
            # feature-extracted data
            if ipmFex: 
                if self.mydata.fex_sum is not None:
                    self.mydata.fex_sum.append( ipmFex.sum )
                if self.mydata.fex_channels is not None:
                    self.mydata.fex_channels.append( ipmFex.channel )
                if self.mydata.fex_position is not None:
                    self.mydata.fex_position.append( [ipmFex.xpos, ipmFex.ypos] )
            else :
                print "No Ipimb fex data"

            # raw data
            if ipmRaw: 
                if self.mydata.raw_channels is not None:
                    self.mydata.raw_channels.append( [ipmRaw.channel0(),
                                                      ipmRaw.channel1(),
                                                      ipmRaw.channel2(),
                                                      ipmRaw.channel3() ] )            
                if self.mydata.raw_voltages is not None:
                    self.mydata.raw_voltages.append( [ipmRaw.channel0Volts(),
                                                      ipmRaw.channel1Volts(),
                                                      ipmRaw.channel2Volts(),
                                                      ipmRaw.channel3Volts() ] )
            else:
                print "No Ipimb raw data"



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

                                            
