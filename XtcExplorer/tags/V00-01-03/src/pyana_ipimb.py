#
# ipimb.py: plot beamline data
#
# 

import numpy as np
import matplotlib.pyplot as plt

from   pypdsdata import xtc
from utilities import PyanaOptions

# analysis class declaration
class  pyana_ipimb ( object ) :
    
    def __init__ ( self,
                   sources = None,
                   plot_every_n = "0",
                   fignum = "1" ) :
        """
        @param ipimb_addresses   list of IPIMB addresses
        @param plot_every_n      Zero (don't plot until the end), or N (int, plot every N event)
        @param fignum            matplotlib figure number
        """


        # initialize data
        opt = PyanaOptions()
        self.sources = opt.getOptStrings(sources)
        print "pyana_ipimb, %d sources: " % len(self.sources)
        for source in self.sources :
            print "  ", source

        self.mpl_num = opt.getOptInteger(fignum)
        self.plot_every_n = opt.getOptInteger(plot_every_n)

        self.n_shots = None

        self.fex_sum = {}
        self.fex_channels = {}
        self.fex_position = {}
        self.raw_channels = {}
        for source in self.sources :
            self.fex_sum[source] = list()
            self.fex_channels[source] = list()
            self.fex_position[source] = list()
            self.raw_channels[source] = list()


    def beginjob ( self, evt, env ) : 
        self.n_shots = 0

    def event ( self, evt, env ) :

        self.n_shots+=1

        # IPM diagnostics, for saturation and low count filtering
        for source in self.sources :

            # raw data
            ipmRaw = evt.get(xtc.TypeId.Type.Id_IpimbData, source )
            if ipmRaw :
                channelVoltages = []
                channelVoltages.append( ipmRaw.channel0Volts() )
                channelVoltages.append( ipmRaw.channel1Volts() )
                channelVoltages.append( ipmRaw.channel2Volts() )
                channelVoltages.append( ipmRaw.channel3Volts() )
                self.raw_channels[source].append( channelVoltages )
            else :
                print "pyana_ipimb: No IpimbData from %s found" % source

            # feature-extracted data
            ipmFex = evt.get(xtc.TypeId.Type.Id_IpmFex, source )

            if ipmFex :
                self.fex_sum[source].append( ipmFex.sum )
                self.fex_channels[source].append( ipmFex.channel )
                self.fex_position[source].append( [ipmFex.xpos, ipmFex.ypos] )
            else :
                print "pyana_ipimb: No IpmFex from %s found" % source


        if self.plot_every_n != 0: 
            if (self.n_shots%self.plot_every_n)==0 : 
                print "Shot#%d ... plotting " % self.n_shots
                self.make_plots(title="IPIMB info accumulated up to shot#%d"%self.n_shots)
                
    def endjob( self, env ) :

        self.make_plots(title="IPIMB info accumulated from all %d shots"%self.n_shots)


    def make_plots(self, title = ""):

        # -------- Begin: move this to beginJob
        """ This part should move to begin job, but I can't get
        it to update the plot in SlideShow mode when I don't recreate
        the figure each time. Therefore plotting is slow... 
        """
        ncols = 3
        nrows = len(self.sources)
        height=3.5
        if nrows * 3.5 > 12 : height = 12/nrows
        width=height*1.3

        fig = plt.figure(num=self.mpl_num, figsize=(width*ncols,height*nrows) )
        fig.clf()
        fig.subplots_adjust(wspace=0.45, hspace=0.45)
        fig.suptitle(title)

        self.ax = []
        for i in range (0, 3*len(self.sources)):
            self.ax.append( fig.add_subplot(nrows, ncols, i) )
        # -------- End: move this to beginJob

        
        
        i = 0
        for source in self.sources :

            xaxis = np.arange( 0, len(self.fex_channels[source]) )

            #ax1 = fig.add_subplot(nrows, ncols, i)
            #plt.axes(self.ax[i])
            self.ax[i].clear()
            plt.axes(self.ax[i])
            array = np.float_(self.fex_sum[source])
            #plt.hist(array, 60)
            plt.plot(xaxis,array)
            plt.title(source)
            plt.ylabel('Sum of channels',horizontalalignment='left') # the other right
            plt.xlabel('Shot number',horizontalalignment='left') # the other right
            i+=1
            #ax2 = fig.add_subplot(nrows, ncols, i)
            self.ax[i].clear()
            plt.axes(self.ax[i])
            array = np.float_(self.fex_channels[source])
            #plt.plot(xaxis, array[:,0],xaxis, array[:,1],xaxis, array[:,2],xaxis, array[:,3])
            plt.hist(array[:,0], 60, histtype='stepfilled', color='r', label='Ch0')
            plt.hist(array[:,1], 60, histtype='stepfilled', color='b', label='Ch1')
            plt.hist(array[:,2], 60, histtype='stepfilled', color='y', label='Ch2')
            plt.hist(array[:,3], 60, histtype='stepfilled', color='m', label='Ch3')
            plt.title(source)
            plt.xlabel('IPIMB Value',horizontalalignment='left') # the other right
            leg = self.ax[i].legend()#('ch0','ch1','ch2','ch3'),'upper center')
            i+=1
            #ax3 = fig.add_subplot(nrows, ncols, i)
            self.ax[i].clear()
            plt.axes(self.ax[i])
            array2 = np.float_(self.fex_position[source])
            plt.scatter(array2[:,0],array2[:,1])
            plt.title(source)
            plt.xlabel('Beam position X',horizontalalignment='left')
            plt.ylabel('Beam position Y',horizontalalignment='left')
            i+=1

        plt.draw()
