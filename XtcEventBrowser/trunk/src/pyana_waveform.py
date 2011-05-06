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
import time

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
        self.ts = {} # time waveform
        self.wf = {} # voltage waveform
        self.wf2 = {} # waveform squared (for computation of RMS)
        self.cfg = {}

        for source in self.sources:
            print source

            self.ts[source] = None
            self.wf[source] = None
            self.wf2[source] = None

            cfg = env.getAcqConfig(self.source)
            self.cfg[source] = { 'nCh' : cfg.nbrChannels(),
                                 'nSamp' : cfg.horiz().nbrSamples(),
                                 'smpInt' : cfg.horiz().sampInterval() }

            #print "  # channels ", cfg.nbrChannels()
            #print "  # samples  ", cfg.horiz().nbrSamples()
            #print "  s interval ", cfg.horiz().sampInterval()
        

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

            print source , self.cfg[source]['nCh'] 
            for channel in range ( 0, self.cfg[source]['nCh'] ) :

                acqData = evt.getAcqValue( source, channel, env)
                if acqData :

                    if self.ts[source] is None:
                        self.ts[source] = acqData.timestamps()
                        
                    # average waveform
                    awf = acqData.waveform()

                    if self.wf[source] is None:
                        self.wf[source] = awf
                        self.wf2[source] = (awf*awf)
                    else :
                        self.wf[source] += awf
                        self.wf[source] += (awf*awf)
                    

        
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
            self.make_plots(data, self.data[source], suptitle=source)

            #print "shape of edge array: ", np.shape(self.edges[source])
            #print self.edges[source]
            #self.make_plots(data, self.edges[source], suptitle=source)


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

        plt.plot(xt)
        plt.xlabel('Bin')
        plt.ylabel('Edge')
        
        print "drawing..."
        plt.draw()

