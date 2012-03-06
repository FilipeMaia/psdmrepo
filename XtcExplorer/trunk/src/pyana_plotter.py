#--------------------------------------------------------------------------
# Description:
#  Pyana user analysis module pyana_plotter...
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

import numpy as np

#import matplotlib 
#matplotlib.use('Qt4Agg')

# alternative 1: pyplot (matlab-like)
import matplotlib.pyplot as plt

## alternative 2: object oriented matplotlib
#import matplotlib as mpl
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
#from matplotlib.figure import Figure

from IPython.Shell import IPShellEmbed

#-----------------------------
# Imports for other modules --
#-----------------------------
from utilities import PyanaOptions 
from displaytools import DataDisplay

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class pyana_plotter (object) :
    """Class whose instance will be used as a user analysis module. """

    #--------------------
    #  Class variables --
    #--------------------
    
    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self,
                   display_mode = "Interactive",
                   ipython         = "False"):
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param display_mode        Interactive (1) or SlideShow (2) or NoDisplay (0)
        @param ipython             Drop into ipython at the end of the job
        """
        self.n_shots = 0

        self.display_mode = None
        if display_mode == "NoDisplay" : self.display_mode = 0
        if display_mode == "Interactive" : self.display_mode = 1
        if display_mode == "SlideShow" :   self.display_mode = 2
        if self.display_mode is None:
            print "Unknown display mode %s, using NoDisplay (0)"%display_mode
            self.display_mode = 0

        opt = PyanaOptions() # convert option string to appropriate type        
        self.ipython      = opt.getOptBoolean(ipython)

        self.data_display = DataDisplay()

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job. It should
        do a one-time initialization possible extracting values from event
        data (which is a Configure object) or environment.

        @param evt    event data object
        @param env    environment object
        """
        self.starttime = time.time()

        # Preferred way to log information is via logging package
        logging.info( "pyana_plotter.beginjob() called with displaymode %d"%self.display_mode )

        if self.display_mode == 0 :
            plt.ioff()
        if self.display_mode == 1 :
            plt.ioff()
        if self.display_mode == 2 :
            plt.ion()


    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "pyana_plotter.beginrun() called" )

    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "pyana_plotter.begincalibcycle() called" )

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        self.n_shots += 1

        if evt.get('skip_event'):
            return

        # if any module changed the display mode, pick it up (we're last)
        evmode = evt.get('display_mode')
        if evmode is not None:

            if evmode != self.display_mode :
                self.display_mode = evmode
                print "pyana_plotter display mode changed: ", self.display_mode,
                
                if self.display_mode == 0:
                    plt.ioff()
                    print " (NoDisplay)" 
                if self.display_mode == 1:
                    plt.ioff()
                    print " (Interactive)" 
                if self.display_mode == 2:
                    plt.ion()
                    print " (SlideShow)" 
                print

            #print "pyana_plotter current display mode: ", self.display_mode

        # only call the plt.draw / plt.show is there's actually
        # something new to show, since it's slow. 
        show_event = evt.get('show_event')
        if show_event and env.subprocess()<1 :

            self.make_plots(evt)

    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        
        logging.info( "pyana_plotter.endcalibcycle() called" )

    def endrun( self, env ) :
        """This optional method is called if present at the end of the run.
        
        @param env    environment object
        """        
        logging.info( "pyana_plotter.endrun() called" )


    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        
        logging.info( "pyana_plotter.endjob() called" )
        endtime = time.time()
        duration = endtime - self.starttime
        #print "Start: %.3f, Stop: %.3f, Duration: %.4f" %(self.starttime,endtime,duration)
        print "\nTiming as measured by pyana_plotter endjob: %.4f s\n" %(duration)

        show_event = evt.get('show_event')
        if show_event and env.subprocess()<1 :

            self.make_plots(evt)

            if self.display_mode > 0 :
                print "Pyana will exit once you close all the MatPlotLib windows"            
                plt.ioff()
                plt.show()
                
        print "-------------------"
        print "'pyana' is done!   "
        print "-------------------"


    def make_plots(self,evt):
        print "pyana_plotter: Shot#%d, Displaymode: %d" % (self.n_shots,self.display_mode)
        self.data_display.event_number = self.n_shots

        #
        # get pointer to the data from each of the modules
        data_ipimb = evt.get('data_ipimb')
        if data_ipimb :
            self.data_display.show_ipimb(data_ipimb)

        data_image = evt.get('data_image')
        if data_image :
            newmode = self.data_display.show_image(data_image)                    
            if newmode is not None:
                self.display_mode = newmode
                    
        if self.ipython :
            plt.draw()
            self.launch_ipython(evt)

        if self.display_mode == 1:
            # Interactive
            plt.ioff()
            plt.show()

        elif self.display_mode == 2:
            # SlideShow
            plt.ion()
            plt.draw()            


    def launch_ipython(self, evt):
        """Launch an ipython session with access to data stored in the evt object
        """
        # get pointer to the data from each of the modules
        data_ipimb = evt.get('data_ipimb')
        if data_ipimb :  print "data_ipimb: ", data_ipimb
        else :           del data_ipimb
        
        data_bld = evt.get('data_bld')
        if data_bld :   print "data_bld: ", data_bld
        else :          del data_bld
        
        data_epics = evt.get('data_epics')
        if data_epics :   print "data_epics: ", data_epics
        else :          del data_epics
        
        data_scan = evt.get('data_scan')
        if data_scan :  print "data_scan: ", data_scan
        else :          del data_scan
        
        data_encoder = evt.get('data_encoder')
        if data_encoder :  print "data_encoder: ", data_encoder
        else :          del data_encoder
        
        data_waveform = evt.get('data_waveform')
        if data_waveform :  print "data_waveform: ", data_waveform
        else :          del data_waveform
        
        data_image = evt.get('data_image')
        if data_image:  print "data_image: ", data_image
        else:           del data_image
        
        data_cspad = evt.get('data_cspad')
        if data_cspad:  print "data_cspad: ", data_cspad
        else:           del data_cspad
        
        ipshell = IPShellEmbed(argv=['-pi1','In \\# >> ','-po','Out \\#: '], 
                               banner='--------- Dropping into iPython ---------',
                               exit_msg='--------- Leaving iPython -------------')
        
        ipshell("Called from endjob. \nTry 'whos' to see the workspace. " \
                "\nHit Ctrl-D to exit iPython and continue program.")
        
        
