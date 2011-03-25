#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $
#
# Description:
#  Pyana user analysis module pyana_waveform...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $

@author Ingrid Ofte
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 1095 $"
# $Source$

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

from pypdsdata import xtc

from utilities import PyanaOptions


#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class pyana_waveform (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, 
                   sources = None,
                   plot_every_n = None,
                   fignum = "1" ) :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param plot_every_n
        @param fignum
        """

        opt = PyanaOptions()
        self.sources = opt.getOptStrings( sources )
        self.plot_every_n = opt.getOptInteger( plot_every_n )
        self.mpl_num = opt.getOptInteger( fignum )

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job.
        It will also be called in any Xtc Configure transition, if 
        configurations have changed since last it was run.

        @param evt    event data object
        @param env    environment object
        """

        # Preferred way to log information is via logging package
        logging.info( "pyana_waveform.beginjob() called " )

        # containers to store data from this job
        self.n_evts = 0
        self.data = {}
        self.datasize = {}
        self.edges = {}

        for source in self.sources:
            print source

            self.data[source] = []
            self.edges[source] = None

            #how many channels?
            cfg = env.getConfig(xtc.TypeId.Type.Id_AcqConfig, source )
            print "  # channels ", cfg.nbrChannels()
            print "  # samples  ", cfg.horiz().nbrSamples()
            print "  s interval ", cfg.horiz().sampInterval()

            
            self.datasize[source] = [cfg.nbrChannels(), cfg.horiz().nbrSamples() ]  
            
        
    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "pyana_waveform.beginrun() called ")

                
    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "pyana_waveform.begincalibcycle() called ")

            
    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "pyana_waveform.event() called ")
        self.n_evts+=1

        for source in self.sources:

            # get data from each channel
            nchannels, nsamples = self.datasize[source]

            wf_arrays = []
            ts_array = None
            for channel in range ( 0, nchannels ) :
                acqData = evt.getAcqValue( source, channel, env)

                m = None
                if acqData :
                    wf = acqData.waveform() 
                    ts = acqData.timestamps()
                    
                    # make sure dimensions are the same
                    if np.shape(wf) != np.shape(ts) :
                        dim = np.shape(wf)
                        ts = ts[0:dim[0]]

                    wf_arrays.append(wf)
                    ts_array = ts
                    
                    # constant threshold
                    baseline = -0.03
                    threshold = -0.06
                    edge = np.zeros(100,dtype=float)
                    self.fill_const_frac_hist(ts, wf, nsamples,
                                              baseline, threshold, edge )
                    if self.edges[source] is None :
                        self.edges[source] = edge
                    else :
                        self.edges[source] += edge
                     
            wf_arrays.insert(0,ts_array )
            
            self.data[source].append( np.array(wf_arrays) )
            

                             
            

                #if self.data[source] is None :
                #    self.data[source] = np.float_(wf)
                #    print len(self.data[source])
                #else :
                #    self.data[source]+=wf
                

    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        
        logging.info( "pyana_waveform.endcalibcycle() called" )

    def endrun( self, env ) :
        """This optional method is called if present at the end of the run.
        
        @param env    environment object
        """
        logging.info( "pyana_waveform.endrun() called" )

    def endjob( self, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        
        logging.info( "pyana_waveform.endjob() called" )

        fignum = self.mpl_num*100

        for source in self.sources :

            data = np.array(self.data[source])
            print "shape of data array: ", np.shape(data)

            self.make_plots(data, self.edges[source], suptitle=source)


    def make_plots(self, data, edges, fignum=1, suptitle = ""):
        
        fig = plt.figure()
        
        ax1 = fig.add_subplot(211)
        xs = np.mean(data, axis=0)
        
        nch,nsmp = np.shape(xs)
        timestamps = xs[0]
        for i in range (1, nch):
            plt.plot(timestamps,xs[i])            
        plt.title('average')
        plt.xlabel('Seconds')
        plt.ylabel('Volts')
            
        ax2 = fig.add_subplot(212)
        xt = edges
        print xt
        dim = np.shape(xt)
        for i in range (0, dim[0]):
            plt.plot(xt[i])
        plt.xlabel('Seconds')
        plt.ylabel('Volts')
        
        print "drawing..."
        plt.draw()


    def fill_const_frac(self, t, v, num_samples, baseline, threshold, edge, n, maxhits):
        """Find the boundaries where the pulse crosses the threshold
           copied from myana's main.cc
        """
        n = 0
        peak = 0.0
        crossed = False
        rising = threshold > baseline
        for k in range (0, num_samples):
            y = v[k]
            over = (( rising and y>threshold ) or
                    ( not rising and y<threshold ) )
            
            if (not crossed and over ):
                crossed = True
                start = k
                peak = y
            elif (crossed and not over ):
                #find the edge
                edge_v = 0.5*(peak+baseline)
                i = start
                if (rising): # leading edge +
                    while ( v[i] < edge_v ):
                        i += 1
                else :      # leading edge -
                    while ( v[i] > edge_v ):
                        i += 1
                if (i > 0) :
                    edge[n] = ((edge_v-v[i-1])*t[i] - (edge_v-v[i])*t[i-1])/(v[i]-v[i-1]);
                else :
                    edge[n] = t[0]
                n+=1
                if ( n >= maxhits ):
                    break
                crossed = False
            elif (( rising and y>peak ) or
                  (not rising and y<peak )) :
                peak = y
    

    def fill_const_frac_hist(self, t, v, num_samples, baseline, threshold, edge_array ): 
        n = 0
        self.fill_const_frac(t, v, num_samples, baseline, threshold, edge_array, n, 100 )
