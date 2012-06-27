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
import numpy as np
#-----------------------------
# Imports for other modules --
#-----------------------------
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
                   channels = None,
                   wf_window = None,
                   plot_every_n = None,
                   accumulate_n = "0",
                   fignum = "1",
                   quantities = "average"):
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param sources         List of DetInfo addresses of Acqiris type
        @param channels        List of channels (default: all)
        @param wf_window       Waveform window (array units)
        @param plot_every_n    Frequency of plot updates
        @param accumulate_n    Accumulate all or reset the array every n shots
        @param fignum          Figure number for matplotlib
        @param quantities      string containing quantities to plot
        """

        # initialize data
        opt = PyanaOptions()
        self.sources = opt.getOptStrings( sources )
        self.quantities = opt.getOptString( quantities )
        self.channels = opt.getOptIntegers( channels ) 

        self.wf_window = None
        if wf_window is not None:
            self.wf_window = [int(time) for time in wf_window.split('-')]

        self.plot_every_n = opt.getOptInteger( plot_every_n )
        self.accumulate_n = opt.getOptInteger( accumulate_n )
        self.mpl_num = opt.getOptInteger( fignum )

        # other
        self.n_shots = None
        self.accu_start = None

        self.do_plots = self.plot_every_n != 0

        # containers to store data from this job
        self.ctr = {} # source-specific event counter 
        self.ts = {} # time waveform
        self.wf = {} # voltage waveform (single event)
        self.wf_accum = {} # voltage waveform (accumulated)
        self.wf2_accum = {} # waveform squared (for computation of RMS) (accumulated)
        self.wf_stack = {}

                
    def resetlists(self):
        self.accu_start = self.n_shots
        for label in self.src_ch :
            self.ctr[label] = 0
            self.wf[label] = None
            del self.wf_accum[label] 
            del self.wf2_accum[label]
            self.wf_accum[label] = None
            self.wf2_accum[label] = None

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        #logging.getLogger().setLevel(logging.DEBUG)
        try:
            env.assert_psana()
            self.psana = True
        except:
            self.psana = False

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
            if self.psana:
                detsrc = source.split('|')[0].split('-')[0]
                if detsrc == "AmoGasdet":
                    detsrc = "AmoGD"
                cfg = env.configStore().get("Psana::Acqiris::ConfigV1", detsrc)
            else:
                cfg = env.getConfig(xtc.TypeId.Type.Id_AcqConfig, source)

            nch = cfg.nbrChannels() 
            nsmp = cfg.horiz().nbrSamples()
            unit = cfg.horiz().sampInterval() * 1.0e9 # nano-seconds
            span = nsmp * unit

            if self.wf_window is None:
                self.wf_window = [ 0, nsmp ]

            print "\n%s has %d channels... "%(source, nch)
            print "Window of %.5f ns, (%d samples, %d ns each)"%(span,nsmp,unit)
            
            for i in range (0, nch ):
                if self.channels is not None and i not in self.channels : continue
                label =  "%s Ch%s" % (source,i) 
                self.src_ch.append(label)                    

                self.data[label] = WaveformData( label )

                width = self.wf_window[1] - self.wf_window[0]
                self.ctr[label] = 0
                self.ts[label] = None
                self.wf[label] = np.empty( width, dtype='float64')
                self.wf_accum[label] = np.empty( width, dtype='float64')
                self.wf2_accum[label] = np.empty( width, dtype='float64')
                
                self.wf_stack[label] = None
                if self.accumulate_n > 0 :
                    self.wf_stack[label] = np.empty( (self.accumulate_n, width), dtype='float64')
                else :
                    self.wf_stack[label] = []
                    
                

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

            # label is e.g. 'AmoGasdet-0|Acqiris-0 Ch0' or 'Camp-0|Acqiris-0 Ch19'
            parts = label.split(' Ch')
            source = parts[0]       # e.g. 'AmoGasdet-0|Acqiris-0' or 'Camp-0|Acqiris-0'
            channel = int(parts[1]) # e.g. 0 or 19

            if self.psana:
                detsrc = source.split('|')[0].split('-')[0]
                if detsrc == "AmoGasdet":
                    detsrc = "AmoGD"
                acqData = evt.get("Psana::Acqiris::DataDesc", detsrc)
            else:
                acqData = evt.getAcqValue( source, channel, env) # pypdsdata.acqiris.DataDescV1

            if acqData :
                if self.ts[label] is None:
                    if self.psana:
                        print "$@%!$#**#@ ... if you see a crash here, it's probably Ingrid's fault"
                        elem = acqData.data(channel) # this is a DataDescElem
                        print "*** elem.nbrSamplesInSeg()=", elem.nbrSamplesInSeg() # e.g. 1000
                        print "*** elem.indexFirstPoint()=", elem.indexFirstPoint() # e.g. 0
                        print "*** elem.nbrSegments()=", elem.nbrSegments()         # e.g. 1
                        timestamp = elem.timestamp() # this is an ndarray with only one element per segment
                        print "*** elem.timestamp=", timestamp
                        print "*** len(elem.timestamp)=", len(timestamp)
                        print "*** elem.nbrSegments()=", elem.nbrSegments()
                        timestamps = [ timestamp[i].pos() * 1.0e9 for i in range(elem.nbrSegments()) ]  # nano-seconds
                        self.ts[label] = timestamps[self.wf_window[0]:self.wf_window[1]]
                        print "self.ts[", label, "] = ", self.ts[label]
                    else:
                        self.ts[label] = acqData.timestamps()[self.wf_window[0]:self.wf_window[1]] * 1.0e9
                        # nano-seconds

                # a waveform
                if self.psana:
                    elem = acqData.data(channel) # this is a DataDescElem
                    awf = elem.waveforms()[0][self.wf_window[0]:self.wf_window[1]]
                else:
                    awf = acqData.waveform()[self.wf_window[0]:self.wf_window[1]]
                
                self.ctr[label]+=1
                
                if self.quantities.find('single')>=0:
                    self.wf[label] = awf

                if self.quantities.find('average')>=0:
                    # collect sum for average
                    if self.wf_accum[label] is None : # first event
                        self.wf_accum[label] = awf
                        self.wf2_accum[label] = (awf*awf)
                    else :
                        self.wf_accum[label] += awf
                        self.wf2_accum[label] += (awf*awf)

                
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


                            
        # only call plotter if this is the main thread
        if (env.subprocess()>0):
            return

        newmode = None
        if self.do_plots and (self.n_shots%self.plot_every_n)==0:

            # flag for pyana_plotter
            evt.put(True, 'show_event')

            wfd = self.update_plot_data()
            evt.put(wfd, 'data_wf')
            
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


        newmode = None

        # flag for pyana_plotter
        evt.put(True, 'show_event')
        
        wfd = self.update_plot_data()
        evt.put(wfd, 'data_wf')
            



    def update_plot_data(self):

        # convert dict to a list:
        data_wf = []
        for label in self.src_ch :
            self.data[label].ts = self.ts[label]

            #self.data[label].wf = None
            #self.data[label].average = None
            #self.data[label].stack = None
            
            if self.quantities.find('single')>=0 :
                self.data[label].wf = self.wf[label]
                
            if self.quantities.find('average')>=0 :
                self.data[label].average = self.wf_accum[label]/self.ctr[label]
                self.data[label].counter = self.ctr[label]
                
            if self.quantities.find('stack')>=0 :
                self.data[label].stack = self.wf_stack[label]

            data_wf.append( self.data[label] )

        return data_wf
                                                            
