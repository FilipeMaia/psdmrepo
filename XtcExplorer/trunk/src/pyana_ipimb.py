#
# ipimb.py: plot beamline data
#
# 

import numpy as np
import matplotlib.pyplot as plt

from   pypdsdata import xtc
from utilities import PyanaOptions

class IpimbData( object ):
    """Container to store data (as numpy arrays)
    for a given IPIMB
    """
    def __init__( self, name ):
        self.name = name 
        self.fex_sum = None
        self.fex_channels = None
        self.fex_position = None
        self.raw_channels = None

    def __str__( self ):
        """Printable description 
        (returned when doing print IpimbData)
        """
        itsme = "\nIpimbData: \n\t name = %s" % self.name
        if self.fex_sum is not None :
            itsme+="\n\t fex_sum = array of shape %s"%str(np.shape(self.fex_sum))
        if self.fex_channels is not None :
            itsme+="\n\t fex_channels = array of shape %s"%str(np.shape(self.fex_channels))
        if self.fex_position is not None :
            itsme+="\n\t fex_position = array of shape %s"%str(np.shape(self.fex_position))
        if self.raw_channels is not None :
            itsme+="\n\t raw_channels = array of shape %s"%str(np.shape(self.raw_channels))
        return itsme

    def __repr__( self ):
        """Short version"""
        itsme = "<IpimbData: %s>" % self.name
        return itsme



# analysis class declaration
class  pyana_ipimb ( object ) :
    
    def __init__ ( self,
                   sources = None,
                   plot_every_n = "0",
                   accumulate_n    = "0",
                   fignum = "1" ) :
        """
        @param ipimb_addresses   list of IPIMB addresses
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
        self.raw_channels = {}
        for source in self.sources :
            self.fex_sum[source] = list()
            self.fex_channels[source] = list()
            self.fex_position[source] = list()
            self.raw_channels[source] = list()

    def resetlists(self):
        self.accu_start = self.n_shots
        for source in self.sources :
            del self.fex_sum[source][:]
            del self.fex_channels[source][:]
            del self.fex_position[source][:]
            del self.raw_channels[source][:]


    def beginjob ( self, evt, env ) : 
        self.n_shots = 0
        self.accu_start = 0
        
        self.data = {}
        for source in self.sources :
            self.data[source] = IpimbData( source ) 
            
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


        # ----------------- Plotting ---------------------
        if self.plot_every_n != 0 and (self.n_shots%self.plot_every_n)==0 :

            header = "shots %d-%d" % (self.accu_start, self.n_shots)
            self.make_plots(title=header)

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
        header = "shots %d-%d" % (self.accu_start, self.n_shots)
        self.make_plots(title=header)

        # convert dict to a list:
        data_ipimb = []
        for source in self.sources :
            data_ipimb.append( self.data[source] )
        # give the list to the event object
        evt.put( data_ipimb, 'data_ipimb' )

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

            xaxis = np.arange( self.accu_start, self.n_shots )
            #xaxis = np.arange( 0, len(self.fex_channels[source]) )

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

            self.data[source].fex_sum = array

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

            self.data[source].fex_channels = array
            self.data[source].raw_channels = np.float_(self.raw_channels[source])

            #ax3 = fig.add_subplot(nrows, ncols, i)
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

                                            
