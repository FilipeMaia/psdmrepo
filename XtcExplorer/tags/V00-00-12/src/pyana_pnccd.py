#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#   Module pyana_misc
#   pyana module with intensity threshold, plotting with matplotlib, allow rescale color plot
#
#   Example xtc file: /reg/d/psdm/sxr/sxrcom10/xtc/e29-r0603-s00-c00.xtc 
#
#   To run: pyana -m mypkg.pyana_misc <filename>
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
class  pyana_pnccd ( object ) :

    # initialize
    def __init__ ( self, image_source="Camp-0|pnCCD-1", threshold="3200" ):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param image_source  address string of Detector-Id|Device-ID
        @param threshold     threshold value (image intensity max > threshold)
        """

        self.img_addr = image_source
        print "Using image source ", self.img_addr

        self.threshold = float(threshold)
        print "Applying threshold ", self.threshold

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
        frame = evt.getPnCcdValue(self.img_addr, env)
        if frame :

            # image is a numpy array (1024x1024 if pnCCD)
            image = np.float_(frame.data())

            # collect min and max intensity of this image
            self.lolimits.append( image.min() )
            self.hilimits.append( image.max() )


            # only plot if intensity is above threshold 
            #print "Image max = %f, threshold = %f " %(image.max(), self.threshold)
            if image.max() > self.threshold :

                # add this image to the sum
                self.n_img+=1
                if self.img_data is None :
                    self.img_data = np.float_(image)
                else :
                    self.img_data += image

                # Draw this event. Background subtracted if possible.
                if self.n_dark > 0 :
                    av_dark_img = self.dark_data/self.n_dark
                    subimage = image - av_dark_img 
                    title = "Event %d, background subtracted (avg of %d dark images)" % \
                          ( self.n_events, self.n_dark )
                    #self.drawframe( subimage, title )
                else :
                    title = "Event %d " % self.n_events
                    #self.drawframe( image, title )


            # if not, collect as dark
            else :

                # additional criteria for dark image can go here
                
                self.n_dark+=1
                if self.dark_data is None :
                    self.dark_data = np.float_(image)
                else :
                    self.dark_data += image


    # after last event has been processed. 
    def endjob( self, env ) :

        print "Done processing       ", self.n_events, " events"
        print "# Dark images (events with max intensity below threshold) = ", self.n_dark
        print "# Signal images (events with max intensity above threshold) = ", self.n_img
        print "Threshold = ", self.threshold
        
        if self.img_data is None :
            print "No image data found from source ", self.img_addr
            return

        # plot the minimums and maximums
        xaxis = np.arange(self.n_events)
        plt.plot( xaxis, np.array(self.lolimits), "g", xaxis, np.array(self.hilimits), "r" )
        #plt.plot( np.array(self.lolimits))
        #plt.plot( np.array(self.hilimits))

        # plot the average image
        av_good_img = self.img_data/self.n_img
        av_dark_img = self.dark_data/self.n_dark
        #baseline = np.average( av_dark_img )
        #av_bkgsubtracted = av_good_img - av_dark_img + baseline
        av_bkgsubtracted = av_good_img - av_dark_img 

        #print "Baseline (average intensity of dark image) = ", baseline

        self.drawframe( av_good_img, "Average of good images")
        self.drawframe( av_dark_img, "Average of dark images" )
        self.drawframe( av_bkgsubtracted, "Average background subtracted")
        #plt.show()



    # -------------------------------------------------------------------
    # Additional functions
        
    def drawframe( self, frameimage, title="" ):

        # plot image frame

        fig = plt.figure()
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
        #plt.draw() 

        plt.show() # starts the GUI main loop
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

    
