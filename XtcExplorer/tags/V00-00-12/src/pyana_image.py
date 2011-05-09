#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#   Module pyana_image
#   pyana module with intensity threshold, plotting with matplotlib, allow rescale color plot
#
#   Example xtc file: /reg/d/psdm/sxr/sxrcom10/xtc/e29-r0603-s00-c00.xtc 
#
#   To run: pyana -m mypkg.pyana_image <filename>
#
"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or
part of it, please give an appropriate acknowledgment.
@author Ingrid Ofte
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import time

import numpy as np
import matplotlib 
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc


#---------------------
#  Class definition --
#---------------------
class  pyana_image ( object ) :

    # initialize
    def __init__ ( self,
                   image_source=None,
                   good_range="0--999999",
                   dark_range="-999999--0",
                   draw_each_event = False):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param image_source     address string of Detector-Id|Device-ID
        @param good_range       threshold values selecting images of interest
        @param dark_range       threshold values selecting dark images
        @param draw_each_event  bool
        """

        self.img_addr = image_source
        print "Using image_source = ", self.img_addr

        tli = str(good_range).split("--")[0]
        thi = str(good_range).split("--")[1]
        tld = str(dark_range).split("--")[0]
        thd = str(dark_range).split("--")[1]

        self.thr_low_image = float( tli )
        self.thr_high_image = float( thi )
        self.thr_low_dark = float( tld )
        self.thr_high_dark = float( thd )
        print "Using good_range = %s " % good_range
        print "  (thresholds =  %d (low) and %d (high) " % (self.thr_low_image, self.thr_high_image)
        print "Using dark_range = %s " % dark_range
        print "  (thresholds =  %d (low) and %d (high) " % (self.thr_low_dark, self.thr_high_dark)

        self.draw_each_event = draw_each_event
        print "Using draw_each_event = ", draw_each_event

        # sum up all image data (above threshold) and all dark data (below threshold)
        self.img_data = None
        self.dark_data = None

        # these will be plotted too
        self.lolimits = []
        self.hilimits = []

        # to keep track
        self.n_events = 0
        self.n_img = 0
        self.n_dark = 0


    # start of job
    def beginjob ( self, evt, env ) : 
        # ideally we should open a figure canvas here.
        pass


    # process event/shot data
    def event ( self, evt, env ) :

        # this one counts every event
        self.n_events+=1
        # print a progress report
        if (self.n_events%1000)==0 :
            print "Event ", self.n_events
        

        # get the requested pnCCD image
        frame = evt.getFrameValue(self.img_addr)
        if frame :

            # image is a numpy array (pixels)
            image = np.float_(frame.data())
            if self.n_events<2 : print "Image dimensions: ", np.shape(image)

            # collect min and max intensity of this image
            self.lolimits.append( image.min() )
            self.hilimits.append( image.max() )

            # select good images
            isGood = False
            if ( image.max() > self.thr_low_image) and (image.max() < self.thr_high_image) :
                isGood = True
                
                # add this image to the sum
                self.n_img+=1
                if self.img_data is None :
                    self.img_data = np.float_(image)
                else :
                    self.img_data += image

            # select dark image
            isDark = False
            if ( image.max() > self.thr_low_dark ) and ( image.max() < self.thr_high_dark ) :
                isDark = True

                self.n_dark+=1
                if self.dark_data is None :
                    self.dark_data = np.float_(image)
                else :
                    self.dark_data += image

            # Draw this event. Background subtracted if possible.
            if self.draw_each_event and isGood :
                if self.n_dark > 0 :
                    av_dark_img = self.dark_data/self.n_dark
                    subimage = image - av_dark_img 
                    title = "Event %d, background subtracted (avg of %d dark images)" % \
                            ( self.n_events, self.n_dark )
                    self.drawframe( subimage, title )
                else :
                    title = "Event %d " % self.n_events
                    self.drawframe( image, title )

                


    # after last event has been processed. 
    def endjob( self, env ) :

        print "Done processing       ", self.n_events, " events"
        print "Range defining images: %f (lower) - %f (upper)" % (self.thr_low_image, self.thr_high_image)
        print "Range defining darks: %f (lower) - %f (upper)" %  (self.thr_low_dark, self.thr_high_dark)
        print "# Signal images = ", self.n_img
        print "# Dark images = ", self.n_dark
        
        if self.img_data is None :
            print "No image data found from source ", self.img_addr
            return

        # plot the minimums and maximums
        xaxis = np.arange(self.n_events)
        plt.plot( xaxis, np.array(self.lolimits), "gv", xaxis, np.array(self.hilimits), "r^" )
        plt.title("Maxim (^)and lower (v) limits")
        #plt.plot( np.array(self.lolimits))
        #plt.plot( np.array(self.hilimits))

        # plot the average image
        av_good_img = self.img_data/self.n_img
        av_bkgsubtracted = av_good_img 
        self.drawframe( av_good_img, "Average of images above threshold")

        if self.n_dark>0 :
            av_dark_img = self.dark_data/self.n_dark
            av_bkgsubtracted -= av_dark_img 
            self.drawframe( av_dark_img, "Average of images below threshold" )
            self.drawframe( av_bkgsubtracted, "Average background subtracted")

        plt.show()



    # -------------------------------------------------------------------
    # Additional functions
        
    def drawframe( self, frameimage, title="" ):

        # plot image frame

        plt.ion()
        fig = plt.figure( 1 )
        cid1 = fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid2 = fig.canvas.mpl_connect('key_press_event', self.onpress)

        self.canvas = fig.add_subplot(111)
        self.canvas.set_title(title)
        # canvas is the main "Axes" object

        self.axes = plt.imshow( frameimage, origin='lower' )
        # axes is the are where the image is plotted
        
        self.colb = plt.colorbar(self.axes,pad=0.01)
        # colb is the colorbar object

        self.orglims = self.axes.get_clim()
        # min and max values in the axes are
        print "Original value limits: ", self.orglims

        print """
        To change the color scale, click on the color bar:
          - left-click sets the lower limit
          - right-click sets higher limit
          - middle-click resets to original
        """
        plt.draw() 
        plt.draw() 

        #plt.show() # starts the GUI main loop
                   # you need to kill window to proceed... 
                   # (this shouldn't be done for every event!)



                               
    # define what to do if we click on the plot
    def onclick(self, event) :

        # can we open a dialogue box here?
        print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y,
        print ' xdata=',event.xdata,' ydata=', event.ydata

        if event.inaxes :
            lims = self.axes.get_clim()
            
            colmin = lims[0]
            colmax = lims[1]
            range = colmax - colmin
            value = colmin + event.ydata * range
            #print colmin, colmax, range, value
            
            # left button
            if event.button is 1 :
                if value > colmin and value < colmax :
                    colmin = value
                    print "new mininum: ", colmin
                else :
                    print "min has not been changed (click inside the color bar to change the range)"
                        
            # middle button
            elif event.button is 2 :
                colmin, colmax = self.orglims
                print "reset"
                    
            # right button
            elif event.button is 3 :
                if value > colmin and value < colmax :
                    colmax = value
                    print "new maximum: ", colmax
                else :
                    print "max has not been changed (click inside the color bar to change the range)"

            plt.clim(colmin,colmax)
            plt.draw() # redraw the current figure




    # define what to do if a button is pressed
    def onpress(self, event) :

        if event.key not in ('t', 'l'): return
        if event.key=='t' : self.set_threshold()
        if event.key=='l' : self.add_savelist()
        

    def set_threshold(self) :
        print " open a dialog to change the threshold to a new value"
        pass


    def add_savelist(self) :
        print "Schedule this image array for saving to binary file"
        pass

    
