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
                   plot_every_n = None,
                   fignum = "1" ) :
        """
        @param ipimb_addresses   list of IPIMB addresses
        @param plot_every_n      None (don't plot until the end), or N (int, plot every N event)
        @param fignum            matplotlib figure number
        """


        # initialize data
        opt = PyanaOptions()
        self.ipimb_addresses = opt.getOptStrings(ipimb_addresses)
        print "pyana_ipimb, %d sources: " % len(self.ipimb_addresses)
        for sources in self.ipimb_addresses :
            print "  ", sources

        self.mpl_num = opt.getOptInteger(fignum)
        
        self.plot_every_n = None
        if plot_every_n is not None : self.plot_every_n = int(plot_every_n)

        self.shot_number = None

        self.fex_sum = {}
        self.fex_channels = {}
        self.fex_position = {}
        for addr in self.ipimb_addresses :
            self.fex_sum[addr] = list()
            self.fex_channels[addr] = list()
            self.fex_position[addr] = list()


    def beginjob ( self, evt, env ) : 
        self.shot_number = 0
        pass

    def event ( self, evt, env ) :

        self.shot_number+=1

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


        if self.plot_every_n is not None: 
            if (self.shot_number%self.plot_every_n)==0 : 
                print "Shot#%d ... plotting " % self.shot_number
                fignum = self.mpl_num*100
                self.make_plots(fignum, suptitle="Accumulated up to Shot#%d"%self.shot_number)
                
    def endjob( self, env ) :
        
        fignum = self.mpl_num*100
        self.make_plots(fignum, suptitle="Average of all events")

    def make_plots(self, fignum = 1, suptitle = ""):
        ncols = 3
        nrows = len(self.ipimb_addresses)
        print "Will now start to produce plots from %d IPMs in %d rows and %d columns" % \
              (len(self.ipimb_addresses), nrows, ncols)

        fig = plt.figure(num=fignum, figsize=(10*ncols/2,10*nrows/2) )
        fig.clf()
        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        fig.suptitle(suptitle)
        
        i = 0
        for addr in self.ipimb_addresses :

            i+=1
            ax1 = fig.add_subplot(nrows, ncols, i)
            array = np.float_(self.fex_sum[addr])
            print "plot 1) shape of array = ", np.shape(array)
            plt.hist(array, 60)
            plt.title(addr)
            plt.xlabel('Sum of channels',horizontalalignment='left') # the other right
            
            i+=1
            ax2 = fig.add_subplot(nrows, ncols, i)
            array = np.float_(self.fex_channels[addr])
            print "plot 2) shape of array = ", np.shape(array)
            xaxis = np.arange( 0, len(self.fex_channels[addr]) )
            plt.plot(xaxis, array[:,0],xaxis, array[:,1],xaxis, array[:,2],xaxis, array[:,3])
            plt.title(addr)
            plt.xlabel('Channels',horizontalalignment='left') # the other right
            leg = ax2.legend(('ch0','ch1','ch2','ch3'),'upper center')
            
            i+=1
            ax3 = fig.add_subplot(nrows, ncols, i)
            array2 = np.float_(self.fex_position[addr])
            print "plot 3) shape of array = ", np.shape(array2)

            plt.scatter(array2[:,0],array2[:,1])
            plt.title(addr)
            plt.xlabel('Beam position X',horizontalalignment='left')
            plt.ylabel('Beam position Y',horizontalalignment='left')

        plt.draw()
