#--------------------------------------------------------------------------
# File and Version Information:
#
# Description:
#  Pyana user analysis module xppt_image_analysis...
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
from pypdsdata.xtc import TypeId

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#from matplotlib.widgets import RectangleSelector
import scipy.io
import scipy.ndimage
import h5py

#---------------------
#  Class definition --
#---------------------
class UpdatingRect(Rectangle):
    def __call__(self, ax):
        self.set_bounds(*ax.viewLim.bounds)
        ax.figure.canvas.draw_idle()
        

class xppt_image_analysis (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self,
                   source = "",
                   region = None, 
                   outputfile = None ) :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param source          address string of the detector source
        @param outputfile      name of output file
        """

        self.source = source
        self.outputfile = outputfile
        self.roi = region
        if self.roi is not None:
            self.roi = [ float(coordinate) for coordinate in region.strip('[()]').split(',') ]
            if len(self.roi) != 4:
                print "Wrong format of ROI!"
                exit(1)
                
        # initializations
        self.nevents = 0
        self.image_average = None

        # allows us to draw 'slideshow' mode
        plt.ion()

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
        logging.info( "xppt_image_analysis.beginjob() called" )

        config = env.getConfig(TypeId.Type.Id_CspadConfig, self.source)
        if not config:
            print '*** cspad config object is missing ***'
            return

        quads = range(4)
        sections = map(config.sections, quads)

        print "Cspad configuration"
        print "  N quadrants   : %d" % config.numQuads()
        print "  Quad mask     : %#x" % config.quadMask()
        print "  payloadSize   : %d" % config.payloadSize()
        print "  badAsicMask0  : %#x" % config.badAsicMask0()
        print "  badAsicMask1  : %#x" % config.badAsicMask1()
        print "  asicMask      : %#x" % config.asicMask()
        print "  numAsicsRead  : %d" % config.numAsicsRead()
        try:
            # older versions may not have all methods
            print "  roiMask       : [%s]" % ', '.join([hex(config.roiMask(q)) for q in quads])
            print "  numAsicsStored: %s" % str(map(config.numAsicsStored, quads))
        except:
            pass
        print "  sections      : %s" % str(map(config.sections, quads))



    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "xppt_image_analysis.beginrun() called" )

    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "xppt_image_analysis.begincalibcycle() called" )

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        self.nevents += 1

        elements = evt.getCsPadQuads(self.source, env)
        if not elements :
            print '*** cspad information is missing ***'
            return

        # elements is a list of elements, quadrants. Potentially there are 4 of them,
        # but in this particular example file, there's only one, which has only one single 2x1. 
        # Here, extract only this 2x1. 
        
        image = elements[0].data().reshape(185, 388)

        #if self.image_average is None:
        #    self.image_average = np.array(image) # need this 'conversion' in order to make a writable copy
        #else:
        #    self.image_average += image
        
        # ----------------------------------------------------------
        # Do some image analysis here

        # filter out hot / saturated pixels:
        im_hot_masked = np.ma.masked_greater_equal(image, 16383 )
        image = np.ma.filled( im_hot_masked, 0)

        # max value
        dims = np.shape(image)
        maxbin = image.argmax()
        maxvalue = image.ravel()[maxbin]
        maxbin_coord = np.unravel_index(maxbin,dims)
        print "Image max = %.2f at bin %d, or (x,y) = %s" % (maxvalue, maxbin, maxbin_coord)

        # center of mass of ROI 
        if self.roi is None: 
            self.roi = [ 0, image.shape[1], 0, image.shape[0] ]
            

        roi_array = image[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]]

        print "ROI   [x1, x2, y1, y2] = ", self.roi
        cms = scipy.ndimage.measurements.center_of_mass(roi_array)
        print "Center-of-mass of the ROI: (x, y) = (%.2f, %.2f)" % (self.roi[0]+cms[1],self.roi[2]+cms[0])
        

        # ----------------------------------------------------------
        # Plot individual events?
        fig = plt.figure(1,figsize=(16,5))
        axes1 = fig.add_subplot(121)
        axes2 = fig.add_subplot(122)

        axim1 = axes1.imshow(image)
        axes1.set_title("Full image")
        #plt.colorbar(axim)

        axim2 = axes2.imshow(roi_array, extent=(self.roi[0],self.roi[1],self.roi[3],self.roi[2]))
        axes2.set_title("Region of Interest")
        
        # rectangular ROI selector
        rect = UpdatingRect([0, 0], 0, 0, facecolor='None', edgecolor='red', picker=10)
        rect.set_bounds(*axes2.viewLim.bounds)
        axes1.add_patch(rect)
        
        # Connect for changing the view limits
        axes2.callbacks.connect('xlim_changed', rect)
        axes2.callbacks.connect('ylim_changed', rect)

        def onpick(event):
            xrange = axes2.get_xbound()
            yrange = axes2.get_ybound()
            self.roi = [ xrange[0], xrange[1], yrange[0], yrange[1]]

            print "Region of interest selected: ", self.roi

            roi_array = image[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]]
            cms = scipy.ndimage.measurements.center_of_mass(roi_array)

            print "Center-of-mass of the ROI: (x, y) = (%.2f, %.2f)" % (self.roi[0]+cms[1],self.roi[2]+cms[0])

        fig.canvas.mpl_connect('pick_event', onpick)
        
        plt.draw()

        return

        # ----------------------------------------------------------
        # this is how you should do it if you have more than one 2x1: 
        for element in elements:
            
            print "  Quadrant %d" % element.quad()
            print "    virtual_channel: %s" % element.virtual_channel()
            print "    lane: %s" % element.lane()
            print "    tid: %s" % element.tid()
            print "    acq_count: %s" % element.acq_count()
            print "    op_code: %s" % element.op_code()
            print "    seq_count: %s" % element.seq_count()
            print "    ticks: %s" % element.ticks()
            print "    fiducials: %s" % element.fiducials()
            print "    frame_type: %s" % element.frame_type()
            print "    sb_temp: %s" % map(element.sb_temp, range(4))

            # image data as 3-dimentional array
            data = element.data()
            print "    Data shape: %s" % str(data.shape)

            



    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        
        logging.info( "xppt_image_analysis.endcalibcycle() called" )

    def endrun( self, env ) :
        """This optional method is called if present at the end of the run.
        
        @param env    environment object
        """
        
        logging.info( "xppt_image_analysis.endrun() called" )

    def endjob( self, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        
        logging.info( "xppt_image_analysis.endjob() called" )

        # to make sure the window stays open 
        print "Done! Pyana will exit once you close the image window!"
        plt.ioff()
        plt.show()
