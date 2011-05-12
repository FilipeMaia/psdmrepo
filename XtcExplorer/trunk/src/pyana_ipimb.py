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
                   ipimb_addresses = None,
                   plot_every_n = "0",
                   fignum = "1" ) :
        """
        @param ipimb_addresses   list of IPIMB addresses
        @param plot_every_n      Zero (don't plot until the end), or N (int, plot every N event)
        @param fignum            matplotlib figure number
        """


        # initialize data
        opt = PyanaOptions()
        self.ipimb_addresses = opt.getOptStrings(ipimb_addresses)
        print "pyana_ipimb, %d sources: " % len(self.ipimb_addresses)
        for sources in self.ipimb_addresses :
            print "  ", sources

        self.mpl_num = opt.getOptInteger(fignum)
        self.plot_every_n = opt.getOptInteger(plot_every_n)

        self.n_shots = None

        self.fex_sum = {}
        self.fex_channels = {}
        self.fex_position = {}
        for addr in self.ipimb_addresses :
            self.fex_sum[addr] = list()
            self.fex_channels[addr] = list()
            self.fex_position[addr] = list()


    def beginjob ( self, evt, env ) : 
        self.n_shots = 0

    def event ( self, evt, env ) :

        self.n_shots+=1

        # IPM diagnostics, for saturation and low count filtering
        for addr in self.ipimb_addresses :

            # raw data
            ipmRaw = evt.get(xtc.TypeId.Type.Id_IpimbData, addr )
            if ipmRaw :
                pass
            else :
                print "No object of type %s found" % self.ipm_addr

            # feature-extracted data
            ipmFex = evt.get(xtc.TypeId.Type.Id_IpmFex, addr )

            if ipmFex :
                self.fex_sum[addr].append( ipmFex.sum )
                self.fex_channels[addr].append( ipmFex.channel )
                self.fex_position[addr].append( [ipmFex.xpos, ipmFex.ypos] )
            else :
                print "No object of type %s found" % self.ipm_addr


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
        nrows = len(self.ipimb_addresses)
        height=3.5
        if nrows * 3.5 > 12 : height = 12/nrows
        width=height*1.3

        fig = plt.figure(num=self.mpl_num, figsize=(width*ncols,height*nrows) )
        fig.clf()
        fig.subplots_adjust(wspace=0.45, hspace=0.45)
        fig.suptitle(title)

        self.ax = []
        for i in range (0, 3*len(self.ipimb_addresses)):
            self.ax.append( fig.add_subplot(nrows, ncols, i) )
        # -------- End: move this to beginJob

        
        
        i = 0
        for addr in self.ipimb_addresses :

            xaxis = np.arange( 0, len(self.fex_channels[addr]) )

            #ax1 = fig.add_subplot(nrows, ncols, i)
            #plt.axes(self.ax[i])
            self.ax[i].clear()
            plt.axes(self.ax[i])
            array = np.float_(self.fex_sum[addr])
            #plt.hist(array, 60)
            plt.plot(xaxis,array)
            plt.title(addr)
            plt.ylabel('Sum of channels',horizontalalignment='left') # the other right
            plt.xlabel('Shot number',horizontalalignment='left') # the other right
            i+=1
            #ax2 = fig.add_subplot(nrows, ncols, i)
            self.ax[i].clear()
            plt.axes(self.ax[i])
            array = np.float_(self.fex_channels[addr])
            #plt.plot(xaxis, array[:,0],xaxis, array[:,1],xaxis, array[:,2],xaxis, array[:,3])
            plt.hist(array[:,0], 60, histtype='stepfilled', color='r', label='Ch0')
            plt.hist(array[:,1], 60, histtype='stepfilled', color='b', label='Ch1')
            plt.hist(array[:,2], 60, histtype='stepfilled', color='y', label='Ch2')
            plt.hist(array[:,3], 60, histtype='stepfilled', color='m', label='Ch3')
            plt.title(addr)
            plt.xlabel('IPIMB Value',horizontalalignment='left') # the other right
            leg = self.ax[i].legend()#('ch0','ch1','ch2','ch3'),'upper center')
            i+=1
            #ax3 = fig.add_subplot(nrows, ncols, i)
            self.ax[i].clear()
            plt.axes(self.ax[i])
            array2 = np.float_(self.fex_position[addr])
            plt.scatter(array2[:,0],array2[:,1])
            plt.title(addr)
            plt.xlabel('Beam position X',horizontalalignment='left')
            plt.ylabel('Beam position Y',horizontalalignment='left')
            i+=1

        plt.draw()
