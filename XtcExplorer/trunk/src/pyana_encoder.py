#
# encoder.py: plot encoder data
#
# 

import numpy as np
import matplotlib.pyplot as plt

from   pypdsdata import xtc
from utilities import PyanaOptions
from utilities import EncoderData



# analysis class declaration
class  pyana_encoder ( object ) :
    
    def __init__ ( self,
                   sources = None,
                   plot_every_n = "0",
                   accumulate_n    = "0",
                   fignum = "1" ) :
        """
        @param sources           list of encoder source addresses
        @param plot_every_n      Zero (don't plot until the end), or N (int, plot every N event)
        @param accumulate_n      Accumulate all (0) or reset the array every n shots
        @param fignum            matplotlib figure number
        """

        # initialize data
        opt = PyanaOptions()
        self.sources = opt.getOptStrings(sources)
        print "pyana_encoder, %d sources: " % len(self.sources)
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
        self.values = {}
        self.counts = {}
        self.timestmps = {}
        for source in self.sources :
            self.values[source] = list()
            self.counts[source] = list()
            self.timestmps[source] = list()

    def resetlists(self):
        self.accu_start = self.n_shots
        for source in self.sources :
            del self.values[source][:]
            del self.counts[source][:]
            del self.timestmps[source][:]


    def beginjob ( self, evt, env ) : 
        self.n_shots = 0
        self.accu_start = 0
        
        self.data = {}
        for source in self.sources :
            self.data[source] = EncoderData( source ) 
            
    def event ( self, evt, env ) :

        self.n_shots+=1

        if evt.get('skip_event') :
            return

        # IPM diagnostics, for saturation and low count filtering
        for source in self.sources :

            encoder = evt.get(xtc.TypeId.Type.Id_EncoderData, source )
            if encoder :
                self.values[source].append( encoder.value() )
                self.counts[source].append( encoder._encoder_count )
                self.timestmps[source].append( encoder._33mhz_timestamp )
            else :
                print "pyana_encoder: No EncoderData from %s found" % source

        # ----------------- Plotting ---------------------
        if self.plot_every_n != 0 and (self.n_shots%self.plot_every_n)==0 :

            header = "Encoder data shots %d-%d" % (self.accu_start, self.n_shots)
            self.make_plots(title=header)

            # flag for pyana_plotter
            evt.put(True, 'show_event')
            
            # convert dict to a list:
            data_encoder = []
            for source in self.sources :
                data_encoder.append( self.data[source] )
            # give the list to the event object
            evt.put( data_encoder, 'data_encoder' )

                        
        # --------- Reset -------------
        if self.accumulate_n!=0 and (self.n_shots%self.accumulate_n)==0 :
            self.resetlists()


    def endjob( self, evt, env ) :

        # ----------------- Plotting ---------------------
        header = "Encoder data shots %d-%d" % (self.accu_start, self.n_shots)
        self.make_plots(title=header)

        # convert dict to a list:
        data_encoder = []
        for source in self.sources :
            data_encoder.append( self.data[source] )
        # give the list to the event object
        evt.put( data_encoder, 'data_encoder' )

    def make_plots(self, title = ""):

        # -------- Begin: move this to beginJob
        """ This part should move to begin job, but I can't get
        it to update the plot in SlideShow mode when I don't recreate
        the figure each time. Therefore plotting is slow... 
        """
        ncols = 3
        nrows = len(self.sources)
        height=4.0
        if nrows * 3.5 > 12 : height = 12/nrows
        width=height*1.2

        fig = plt.figure(num=self.mpl_num, figsize=(width*ncols,height*nrows) )
        fig.clf()
        fig.subplots_adjust(wspace=0.35, hspace=0.35, top=0.85)
        fig.suptitle(title)

        self.ax = []
        for i in range (0, ncols*len(self.sources)):
            self.ax.append( fig.add_subplot(nrows, ncols, i) )
        # -------- End: move this to beginJob

        
        
        i = 0
        for source in self.sources :

            xaxis = np.arange( self.accu_start, self.n_shots )
            nbinsx = xaxis.size
            ticks = [xaxis[0],xaxis[nbinsx/3],xaxis[2*nbinsx/3],xaxis[-1]] 

            self.ax[i].clear()
            plt.axes(self.ax[i])
            array = np.float_(self.values[source])

            plt.plot(xaxis,array)
            plt.title(source)
            plt.ylabel('Value',horizontalalignment='left') # the other right
            plt.xlabel('Shot number',horizontalalignment='left') # the other right
            self.ax[i].set_xlim( xaxis[0], xaxis[-1] )
            self.ax[i].set_xticks( ticks )
            i+=1
            self.data[source].values = array


            self.ax[i].clear()
            plt.axes(self.ax[i])
            array = np.float_(self.counts[source])

            plt.plot(xaxis,array)
            plt.title(source)
            plt.ylabel('Counts',horizontalalignment='left') # the other right
            plt.xlabel('Shot number',horizontalalignment='left') # the other right
            self.ax[i].set_xlim( xaxis[0], xaxis[-1] )
            self.ax[i].set_xticks( ticks )
            i+=1
            self.data[source].values = array

            self.ax[i].clear()
            plt.axes(self.ax[i])
            array = np.float_(self.timestmps[source])

            plt.plot(xaxis,array)
            plt.title(source)
            plt.ylabel('Timestamps',horizontalalignment='left') # the other right
            plt.xlabel('Shot number',horizontalalignment='left') # the other right
            self.ax[i].set_xlim( xaxis[0], xaxis[-1] )
            self.ax[i].set_xticks( ticks )
            i+=1
            self.data[source].values = array


        plt.draw()

                                            
