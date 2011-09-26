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
from utilities import WaveformData


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
                   accumulate_n = "0",
                   fignum = "1" ) :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param sources         List of DetInfo addresses of Acqiris type
        @param plot_every_n    Frequency of plot updates
        @param accumulate_n    Accumulate all or reset the array every n shots
        @param fignum          Figure number for matplotlib
        """

        # initialize data
        opt = PyanaOptions()
        self.sources = opt.getOptStrings( sources )
        self.plot_every_n = opt.getOptInteger( plot_every_n )
        self.accumulate_n = opt.getOptInteger( accumulate_n )
        self.mpl_num = opt.getOptInteger( fignum )

        # other
        self.n_shots = None
        self.accu_start = None


    def initlists(self):
        # containers to store data from this job
        self.ctr = {} # source-specific event counter 
        self.ts = {} # time waveform
        self.wf = {} # voltage waveform
        self.wf2 = {} # waveform squared (for computation of RMS)
        for label in self.src_ch :
            self.ctr[label] = 0
            self.ts[label] = None
            self.wf[label] = None
            self.wf2[label] = None
                
    def resetlists(self):
        self.accu_start = self.n_shots
        for label in self.src_ch :
            self.ctr[label] = 0
            del self.wf[label] 
            del self.wf2[label]
            self.wf[label] = None
            self.wf2[label] = None

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

        self.n_shots = 0 # total
        self.accu_start = 0

        # list of channels and source (names)
        self.cfg = {} # configuration object
        self.src_ch = []

        self.data = {}
        for source in self.sources:
            print source

            cfg = env.getAcqConfig(source)
            self.cfg[source] = { 'nCh' : cfg.nbrChannels(),
                                 'nSamp' : cfg.horiz().nbrSamples(),
                                 'smpInt' : cfg.horiz().sampInterval() }
            
            nch = cfg.nbrChannels()
            for i in range (0, nch ):
                label =  "%s Ch%s" % (source,i) 
                self.src_ch.append(label)                    

                self.data[label] = WaveformData( label )
                
        # lists to fill numpy arrays
        self.initlists()



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
        self.n_shots+=1
        
        if evt.get('skip_event') :
            return

        for label in self.src_ch :

            parts = label.split(' Ch')
            source = parts[0]
            channel = int(parts[1])
            
            acqData = evt.getAcqValue( source, channel, env)
            if acqData :

                if self.ts[label] is None:
                    self.ts[label] = acqData.timestamps()
                        
                # average waveform
                awf = acqData.waveform()

                if self.wf[label] is None :
                    self.ctr[label] = 1
                    self.wf[label] = awf
                    self.wf2[label] = (awf*awf)
                else :
                    self.ctr[label] += 1
                    self.wf[label] += awf
                    self.wf[label] += (awf*awf)

                            
        if self.plot_every_n != 0 and (self.n_shots%self.plot_every_n)==0 :
            # flag for pyana_plotter
            evt.put(True, 'show_event')
            
            self.make_plots()

            data_waveform = []
            for label in self.src_ch :
                data_waveform.append( self.data[label] )
            evt.put( data_waveform, 'data_waveform' )
                
        if  ( self.accumulate_n != 0 and (self.n_shots%self.accumulate_n)==0 ):
            self.resetlists()

                    
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

    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        
        logging.info( "pyana_waveform.endjob() called" )

        self.make_plots()
        wfdata = []
        for label in self.src_ch :
            wfdata.append( self.data[label] )
        evt.put( wfdata, 'data_waveform' )

    def make_plots(self):

        nplots = len(self.src_ch)
        ncols = 1
        nrows = len(self.src_ch)
            
        height=4.0
        if (nrows * height) > 14 : height = 14/nrows
        width=height*4.3
        
        fig = plt.figure(num=self.mpl_num, figsize=(width*ncols,height*nrows) )
        fig.clf()
        fig.suptitle("Average waveform of shots %d-%d" % (self.accu_start,self.n_shots))

        if nplots > 1 :
            fig.subplots_adjust(wspace=0.45, hspace=0.45)

        pos = 1
        for source in self.src_ch :
            nev = self.ctr[source]
            ts_axis = self.ts[source]

            print "plotting %d events from source %s" % (nev,source)
            self.wf_avg = self.wf[source] / nev

#            # ... plotting with error bars is terribly slow.... enable at your own "risk"
#            self.wf2_avg = self.wf2[source] / nev
#            self.wf_rms = np.sqrt( self.wf2_avg - self.wf_avg*self.wf_avg ) / np.sqrt(nev)

            dim1 = np.shape(self.wf_avg)
#            dim2 = np.shape(self.wf_rms)
            dim3 = np.shape(ts_axis)

            if dim3 != dim1 :
                ts_axis = ts_axis[0:dim1[0]]

            ax = fig.add_subplot(nrows,ncols,pos)
            pos+=1 

            # scientific notation for time axis
            plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))

            unit = ts_axis[1]-ts_axis[0]
            plt.xlim(ts_axis[0]-unit,ts_axis[-1]+unit)
                
#           # ... plotting with error bars is terribly slow.... enable at your own "risk"
#           plt.errorbar(ts_axis, self.wf_avg,yerr=self.wf_rms, mew=0.0)
            plt.errorbar(ts_axis, self.wf_avg)
            plt.title(source)
            plt.xlabel("time   [s]")
            plt.ylabel("voltage   [V]")

            self.data[source].wf_voltage = self.wf_avg
            self.data[source].wf_time = ts_axis


        plt.draw()

        print "shot#%d: wf avg histogram plotted for %d sources" % (self.n_shots,nplots)
        

