#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $
#
# Description:
#  Pyana user analysis module pyana_waveform_beta...
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
class pyana_waveform_beta (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, 
                   source = None,
                   filter = None, 
                   quantities = None ):
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param source          DetInfo address of Acqiris type
        @param filter          filter string
        @param quantities      string, list of quantities to plot: ch0:extent, ch1:extent...
                                                                    where extent=(start,stop)
                                                                    average, stack
        """

        # initialize data
        opt = PyanaOptions()
        self.source = opt.getOptString( source )
        self.filter = opt.getOptString( filter )
        self.quantities = opt.getOptStringsDict(quantities)

        # other
        self.n_shots = None

        self.mydata = WaveformData(self.source)
        

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
        logging.info( "pyana_waveform_beta.beginjob() called " )

        # reset the counter
        self.n_shots = 0

        # get the configuration object
        cfg = env.getAcqConfig(self.source)

        self.mydata.nCh = cfg.nbrChannels()
        self.mydata.nSmp = cfg.horiz().nbrSamples()
        self.mydata.smpInt = cfg.horiz().sampInterval() 

        self.mydata.channels = range(0,self.mydata.nCh)
        try:
            self.mydata.channels = self.quantities['channels']
        except:
            pass

        self.mydata.wf_time = []
        self.mydata.wf_voltages = []
        self.mydata.wf2_voltages = []
        for ch in xrange(self.mydata.nCh):
            self.mydata.wf_time.append(None)
            self.mydata.wf_voltages.append(None)
            self.mydata.wf2_voltages.append(None)
            

    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "pyana_waveform_beta.beginrun() called ")

                
    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "pyana_waveform_beta.begincalibcycle() called ")
            

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "pyana_waveform_beta.event() called ")
        self.n_shots+=1
        
        if evt.get('skip_event') :
            return

        # -------------------------------------------------
        # get event data
        # -------------------------------------------------
            
        for ch in xrange( self.mydata.nCh ):
            acqData = evt.getAcqValue( self.source, ch, env)

            if acqData :

                awf = acqData.waveform()

                if self.mydata.wf_time[ch] is None: 
                    self.mydata.wf_time[ch] = acqData.timestamps()
                    self.mydata.wf_voltages[ch] = awf
                    self.mydata.wf2_voltages[ch] = (awf*awf)
                else :
                    self.mydata.wf_voltages[ch] += awf
                    self.mydata.wf2_voltages[ch] += (awf*awf)
                        

        # -------------------------------------------------
        # make plots
        # -------------------------------------------------
        plot_data = evt.get('plot_data')
        if plot_data is None:
            plot_data = []
        plot_data.append( self.mydata )
        evt.put( plot_data, 'plot_data' )
        evt.put( True, 'show_event') 


                    
    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        
        logging.info( "pyana_waveform_beta.endcalibcycle() called" )

    def endrun( self, env ) :
        """This optional method is called if present at the end of the run.
        
        @param env    environment object
        """
        logging.info( "pyana_waveform_beta.endrun() called" )

    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """        
        logging.info( "pyana_waveform_beta.endjob() called" )



