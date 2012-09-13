#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module image_plotter...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Ingrid Ofte
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging
import numpy as np

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
class image_plotter (object) :
    """Class whose instance will be used as a user analysis module. """

    #--------------------
    #  Class variables --
    #--------------------
    
    # usual convention is to prefix static variables with s_
    s_staticVariable = 0

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, images ) :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param images    list of (names of) images to be plotted
        """

        self.images = images.split()
        print "Preparing to plot images: ", self.images

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
        logging.info( "image_plotter.beginjob() called" )


    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "image_plotter.beginrun() called" )

    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "image_plotter.begincalibcycle() called" )

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        # get the image(s) from other modules
        for img_name in self.images:
            
            img = evt.get(img_name)


    def endcalibcycle( self, evt, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param evt    event data object
        @param env    environment object
        """
        
        logging.info( "image_plotter.endcalibcycle() called" )

    def endrun( self, evt, env ) :
        """This optional method is called if present at the end of the run.
        
        @param evt    event data object
        @param env    environment object
        """
        
        logging.info( "image_plotter.endrun() called" )

    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param evt    event data object
        @param env    environment object
        """
        
        logging.info( "image_plotter.endjob() called" )
