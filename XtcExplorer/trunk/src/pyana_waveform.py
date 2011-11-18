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
import array

#-----------------------------
# Imports for other modules --
#-----------------------------
import numpy as np
import matplotlib.pyplot as plt

from pypdsdata import xtc

from utilities import PyanaOptions
from utilities import WaveformData
from utilities import Plotter


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
                   fignum = "1",
                   quantities = "average"):
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param sources         List of DetInfo addresses of Acqiris type
        @param plot_every_n    Frequency of plot updates
        @param accumulate_n    Accumulate all or reset the array every n shots
        @param fignum          Figure number for matplotlib
        @param quantities      string containing quantities to plot
        """

        # initialize data
        opt = PyanaOptions()
        self.sources = opt.getOptStrings( sources )
        self.quantities = opt.getOptString( quantities )

        self.plot_every_n = opt.getOptInteger( plot_every_n )
        self.accumulate_n = opt.getOptInteger( accumulate_n )
        self.mpl_num = opt.getOptInteger( fignum )

        # other
        self.n_shots = None
        self.accu_start = None

        self.do_plots = self.plot_every_n != 0
        self.plotter = None

        # containers to store data from this job
        self.ctr = {} # source-specific event counter 
        self.ts = {} # time waveform
        self.wf = {} # voltage waveform
        self.wf2 = {} # waveform squared (for computation of RMS)
        self.wf_stack = {}

                
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
        self.src_ch = []

        self.data = {}
        for source in self.sources:
            cfg = env.getAcqConfig(source)

            nch = cfg.nbrChannels()
            nsmp = cfg.horiz().nbrSamples()
            unit = cfg.horiz().sampInterval() * 1.0e9 # nano-seconds
            span = nsmp * unit
            print "%s has %d channels, wf window %.5f ns, %d samples"%(source,nch,span,nsmp)
            
            for i in range (0, nch ):
                label =  "%s Ch%s" % (source,i) 
                self.src_ch.append(label)                    

                self.data[label] = WaveformData( label )
                
                self.ctr[label] = 0
                self.ts[label] = None
                self.wf[label] = np.empty( nsmp, dtype='float64')
                self.wf2[label] = np.empty( nsmp, dtype='float64')

                self.wf_stack[label] = None
                if self.accumulate_n > 0 :
                    self.wf_stack[label] = np.empty( (self.accumulate_n, nsmp), dtype='float64')
                else :
                    self.wf_stack[label] = []
                    
                
        self.plotter = Plotter()
        self.plotter.settings(4,4) # set default frame size
        self.plotter.threshold = None

        #if self.threshold is not None:
        #    self.plotter.threshold = self.threshold.value
        #    self.plotter.vmin, self.plotter.vmax = self.plot_vmin, self.plot_vmax




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

        row = self.n_shots - self.accu_start -1

        for label in self.src_ch :

            parts = label.split(' Ch')
            source = parts[0]
            channel = int(parts[1])
            
            acqData = evt.getAcqValue( source, channel, env)
            if acqData :

                if self.ts[label] is None:
                    self.ts[label] = acqData.timestamps() * 1.0e9 # nano-seconds

                # a waveform
                awf = acqData.waveform() 
                
                self.ctr[label]+=1

                
                if self.quantities.find('stack')>=0 :
                    # fill image
                    try:
                        self.wf_stack[label][row] =  awf
                    except: 
                        if self.accumulate_n==0:
                            if self.n_shots < 10:
                                print "WARNING: Accumulating stack of waveforms ",
                                print "with accumulate_n set to 0. Faster if you ",
                                print "set accumulate_n to a number"
                            self.wf_stack[label].append( awf )


                if self.quantities.find('average')>=0:
                    # collect sum for average
                    if self.wf[label] is None : # first event
                        self.wf[label] = awf
                        self.wf2[label] = (awf*awf)
                    else :
                        self.wf[label] += awf
                        self.wf2[label] += (awf*awf)

                            
        if self.do_plots and (self.n_shots%self.plot_every_n)==0:

            # flag for pyana_plotter
            evt.put(True, 'show_event')
            
            self.make_plots()
                
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
        

    def make_plots(self):

        ## Plotting
        if self.quantities.find('average')>=0:
            for source in self.src_ch :            
                name = "wfavg(%s)"%source
                title = "Average, %s"%source
                nbins = self.wf[source].size
                contents = (self.ts[source][0:nbins],self.wf[source]/self.ctr[source])
                self.plotter.add_frame(name,title,contents, aspect='auto')

        if self.quantities.find('stack')>=0:
            for source in self.src_ch :            
                name = "wfstack(%s)"%source
                title = "Stack, %s"%source
                wf_image = self.wf_stack[source]
                if type(wf_image).__name__=='list' :
                    wf_image = np.float_(self.wf_stack[source])
                contents = (wf_image,) # a tuple
                self.plotter.add_frame(name,title,contents,aspect='equal')
                self.plotter.frames[name].axis_values = self.ts[source]

                
        self.plotter.plot_all_frames(ordered=True)

        suptitle = "Events %d-%d" % (self.accu_start,self.n_shots)
        plt.suptitle(suptitle)
        



