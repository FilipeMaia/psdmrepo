#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $
#
# Description:
#  Pyana user analysis module pyana_plotter...
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

import matplotlib 
matplotlib.use('Qt4Agg')

# alternative 1: pyplot (matlab-like)
import matplotlib.pyplot as plt

# alternative 2: object oriented matplotlib
import matplotlib as mpl
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure


#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class ImagePlot(FigureCanvas) :
    def __init__(self, parent=None,
                 width=10, height=8, dpi=100, bgcolor=None, num=1 ):
        fig = Figure(figsize=(width,height),dpi=dpi,facecolor=bgcolor, edgecolor=bgcolor)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        fig.suptitle("ImagePlot Changed")
        self.axes = fig.add_subplot(111)
        #self.toolbar = NavigationToolbar(self,self)

    def draw_array(self,array):
        self.axesim = self.axes.imshow( array, origin='lower' )
        # axes image 
        self.show()


class pyana_plotter (object) :
    """Class whose instance will be used as a user analysis module. """

    #--------------------
    #  Class variables --
    #--------------------
    
    # usual convention is to prefix static variables with s_
    s_staticVariable = 0

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        """
        self.nevents = 0

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

        # Preferred way to log information is via logging package
        logging.info( "pyana_plotter.beginjob() called" )


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

        # being the first module to run, 
        # display all plots from previous event
        if self.nevents > 0 :
            print "pyana_plotter plotting event # ", self.nevents 
            plt.show()

        self.nevents += 1

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

    def endjob( self, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        
        logging.info( "pyana_plotter.endjob() called" )
