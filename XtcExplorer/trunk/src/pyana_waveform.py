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

        opt = PyanaOptions()
        self.sources = opt.getOptStrings( sources )
        self.plot_every_n = opt.getOptInteger( plot_every_n )
        self.accumulate_n = opt.getOptInteger( accumulate_n )
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
        self.n_shots = 0 # total
        self.ctr = {} # counter
        self.ts = {} # time waveform
        self.wf = {} # voltage waveform
        self.wf2 = {} # waveform squared (for computation of RMS)
        self.cfg = {} # configuration object

        # list of channels and source (names)
        self.src_ch = []
        
        for source in self.sources:
            print source

            cfg = env.getAcqConfig(source)
            self.cfg[source] = { 'nCh' : cfg.nbrChannels(),
                                 'nSamp' : cfg.horiz().nbrSamples(),
                                 'smpInt' : cfg.horiz().sampInterval() }

            for i in range (0, cfg.nbrChannels() ):
                label =  "%s Ch%s" % (source,i) 
                self.src_ch.append(label)
                    
                self.ctr[label] = None
                self.ts[label] = None
                self.wf[label] = None
                self.wf2[label] = None


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
        
        for source in self.sources:

            #print source , self.cfg[source]['nCh'] 
            for channel in range ( 0, self.cfg[source]['nCh'] ) :

                label =  "%s Ch%s" % (source,channel) 

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
            self.make_plots()

        if  ( self.accumulate_n != 0 and (self.n_shots%self.accumulate_n)==0 ):
            """ Reset arrays
            """
            for source in self.sources:
                for channel in range ( 0, self.cfg[source]['nCh'] ) :
                    label =  "%s Ch%s" % (source,channel) 
                    self.ctr[label] = 0
                    self.wf[label] = None
                    self.wf2[label] = None

                    
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
        self.make_plots()


    def make_plots(self):

        print "make_plots in shot#%d"%self.n_shots
        nplots = len(self.src_ch)

        ncols = 1
        nrows = 1
        if nplots == 2: ncols = 2
        if nplots == 3: ncols = 3
        if nplots == 4: ncols = 2; nrows = 2
        if nplots > 4:
            ncols = 3
            nrows = nplots / 3
            if nplots%3 > 0 : nrows += 1
            
        height=4.2
        if (nrows * height) > 14 : height = 14/nrows
        width=height*1.3
        
        fig = plt.figure(num=self.mpl_num, figsize=(width*ncols,height*nrows) )
        fig.clf()
        fig.suptitle("Average waveform of shots %d-%d" % ((self.n_shots-self.accumulate_n),self.n_shots))

        if nplots > 1 :
            fig.subplots_adjust(wspace=0.45, hspace=0.45)

        pos = 1
        for source in self.src_ch :
            print "plotting %d events from source %s" % (self.ctr[source],source)
            
            self.wf_avg = self.wf[source] / self.ctr[source]
            self.wf2_avg = self.wf2[source] / self.ctr[source]
            self.wf_rms = np.sqrt( self.wf2_avg - self.wf_avg*self.wf_avg ) / np.sqrt(self.ctr[source])

            dim1 = np.shape(self.wf_avg)
            dim2 = np.shape(self.wf_rms)
            dim3 = np.shape(self.ts[source])

            #print "Making plot for %s:" % source
            #print " wf shape ", dim1
            #print " rms shape ", dim2
            #print " time shape ", dim3

            ts_axis = self.ts[source]
            if dim3 != dim1 :
                ts_axis = self.ts[source][0:dim1[0]]

            ax = fig.add_subplot(nrows,ncols,pos)
            pos+=1 

            # scientific notation for time axis
            plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))

            unit = ts_axis[1]-ts_axis[0]
            plt.xlim(ts_axis[0]-unit,ts_axis[-1]+unit)
                
            plt.errorbar(ts_axis, self.wf_avg,yerr=self.wf_rms, mew=0.0)
            plt.title(source)
            plt.xlabel("time   [s]")
            plt.ylabel("voltage   [V]")


            plt.draw()

        print "wf avg histogram plotted for %d sources" % nplots
        

