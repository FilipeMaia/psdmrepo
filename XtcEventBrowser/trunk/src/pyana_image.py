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

import scipy.ndimage.interpolation as interpol
import numpy as np
import matplotlib.pyplot as plt
import h5py

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc
#from xbplotter import Plotter
from mpl_toolkits.axes_grid1 import AxesGrid

#---------------------
#  Class definition --
#---------------------
class  pyana_image ( object ) :

    # initialize
    def __init__ ( self,
                   image_addresses = None,
                   image_nicknames = None,
                   image_rotations = None,
                   image_shifts = None,
                   image_scales = None,
                   output_file = None,
                   good_range="0--999999",
                   dark_range="-999999--0",
                   draw_each_event = False):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param image_addresses   (list) address string of Detector-Id|Device-ID
        @param image_rotations  (list) rotation, in degrees, to be applied to image(s)
        @param image_shifts     (list) shift, in (npixX,npixY), to be applied to image(s)
        @param image_scales     (list) scale factor to be applied to images
        @param output_file      filename (If collecting: write to this file)
        @param good_range       threshold values selecting images of interest
        @param dark_range       threshold values selecting dark images
        @param draw_each_event  bool
        """
        if image_addresses is None :
            print "Error! You've called pyana_image without specifying an image address"
            
        self.image_addresses = image_addresses.split(" ")
        nsources = len(self.image_addresses)
        print "pyana_image, %d sources: " % nsources
        for sources in self.image_addresses :
            print "  ", sources

        self.image_nicknames = []
        if image_nicknames is None:
            for i in range (0, len(self.image_addresses) ):
                self.image_nicknames.append( "Im%d"%(i+1) )
        else :
            self.image_nicknames = image_nicknames.split(" ")

        
        self.image_rotations = None
        if image_rotations is not None:
            if image_rotations == "" or image_rotations == "None" :
                self.image_rotations = None
            else :    
                self.image_rotations = {}
                list_of_rotations = image_rotations.split(" ")
                if len(list_of_rotations) != nsources: print "Plz provide rotation angles for *all* images!"
                i = 0
                for source in self.image_addresses :
                    self.image_rotations[source] = float( list_of_rotations[i] )
                    i+=1
                
            
        self.image_shifts = None
        if image_shifts is not None:
            if image_shifts == "" or image_shifts == "None" :
                self.image_shifts = None
            else :
                self.image_shifts = {}
                list_of_shifts =  image_shifts.split(" ") 
                if len(list_of_shifts) != nsources: print "Plz provide shift amount for *all* images!"
                i = 0
                for source in self.image_addresses :
                    shift = list_of_shifts[i].lstrip("(").rstrip(")").split(",")
                    self.image_shifts[source] = (int(shift[0]), int(shift[1]))
                    i+=1

        self.image_scales = None
        if image_scales is not None:
            if image_scales == "" or image_scales == "None" :
                self.image_scales = None
            else :
                self.image_scales = {}
                list_of_scales = image_scales.split(" ")            
                if len(list_of_scales) != nsources: print "Plz provide scale factors for *all* images!"
                i = 0
                for sources in self.image_adresses :
                    self.image_scales[source] = float( list_of_scales[i] )
                    i+=1

        self.output_file = output_file
        if output_file == "" or output_file == "None" :
            self.output_file = None
        print "Using output_file: ", self.output_file

                
        # ranges
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


    # start of job
    def beginjob ( self, evt, env ) : 

        #self.plotter = Plotter()

        # to keep track
        self.n_events = 0

        # averages
        self.image_data = {}
        self.dark_data = {}
        self.lolimits = {}
        self.hilimits = {}
        self.fignum = {}
        self.n_img = {}
        self.n_dark = {}
        for addr in self.image_addresses :
            self.image_data[addr] = None
            self.dark_data[addr] = None

            # these will be plotted too
            self.lolimits[addr] = []
            self.hilimits[addr] = []

            self.fignum[addr] = 200 + self.image_addresses.index(addr)

            self.n_img[addr] = 0
            self.n_dark[addr] = 0



    # process event/shot data
    def event ( self, evt, env ) :
        print "pyana_image event"

        # this one counts every event
        self.n_events+=1
        # print a progress report
        if (self.n_events%1000)==0 :
            print "Event ", self.n_events


        # for each event, append images to be plotted to this list
        event_display_images = []

        fignum = 101
        axspos = 0

        # get the requested images
        for addr in self.image_addresses :
        
            frame = evt.getFrameValue(addr)
            if not frame :
                print "No frame from ", addr
                continue

            # image is a numpy array (pixels)
            image = np.float_(frame.data())

            # collect min and max intensity of this image
            self.lolimits[addr].append( image.min() )
            self.hilimits[addr].append( image.max() )

            # select good (signal) images
            isGood = False
            if ( image.max() > self.thr_low_image) and (image.max() < self.thr_high_image) :
                isGood = True
                
            # select dark images
            isDark = False
            if ( image.max() > self.thr_low_dark ) and ( image.max() < self.thr_high_dark ) :
                isDark = True

            # sanity check
            if isGood and isDark :
                print "WARNING! This image has been selected both as signal AND dark! "


            # Apply shift, rotation, scaling of this image if needed:
            if self.image_rotations is not None:
                rotatedimage = interpol.rotate( image, self.image_rotations[addr], reshape=False )                
                print "rot: shape of old image = ", np.shape(image)
                print "rot: shape of new image = ", np.shape(rotatedimage)
                image = rotatedimage

            if self.image_shifts is not None:
                #implement this
                shiftx, shifty = self.image_shifts[addr]
                shiftedimage = np.roll(image,shiftx,0)
                shiftedimage = np.roll(image,shifty,1)
                print "shift: shape of old image = ", np.shape(image)
                print "shift: shape of new image = ", np.shape(shiftedimage)
                image = shiftedimage
                
            if self.image_scales is not None:
                #implement this!
                pass


            # add this image to the sum (for average)
            if isGood :
                self.n_img[addr]+=1
                if self.image_data[addr] is None :
                    self.image_data[addr] = np.float_(image)
                else :
                    self.image_data[addr] += image

            elif isDark :                
                self.n_dark[addr]+=1
                if self.dark_data[addr] is None :
                    self.dark_data[addr] = np.float_(image)
                else :
                    self.dark_data[addr] += image


            # images for plotting
            #if isGood :
            #    if self.subtract_bkg and self.n_dark[addr] > 0 :
            #        # subtract average of darks so far collected
            #        image_bkgsub = image - self.dark_data[addr]/self.n_dark[addr]
            #        event_display_images.append( (addr, image_bkgsub) )
            #    else :
            event_display_images.append( (addr, image) )
            

        # Draw images from this event
        for i in range ( 0, len(event_display_images) ):

            ad1,im1 = event_display_images[i]
            ad2,im2 = event_display_images[i-1]
            lb1 = self.image_nicknames[i]
            lb2 = self.image_nicknames[i-1]
            event_display_images.append( ("Diff %s-%s"%(lb1,lb2), im1-im2) )

            F = np.fft.fftn(im1-im2)
            event_display_images.append( ("FFT %s-%s"%(lb1,lb2), np.log(np.abs(np.fft.fftshift(F))**2) ) )


        nplots = len(event_display_images)
        ncol = 3
        if nplots<3 : ncol = nplots
        nrow = int( nplots/ncol)
        fig = plt.figure(101,(4*ncol,4*nrow))
        fig.clf()
        fig.suptitle("Event#%d"%self.n_events)

        #grid = AxesGrid( fig, 111,
        #                 nrows_ncols = (2,3),
        #                 axes_pad = 0.6,
        #                 label_mode = 1,
        #                 share_all = True,
        #                 cbar_mode = "each",
        #                 cbar_location = "right",
        #                 cbar_size = "7%",
        #                 cbar_pad = "3%",
        #                 )


        pos = 0
        self.caxes = [] # list of references to colorbar Axes
        self.axims = [] # list of references to image Axes
        for ad, im in sorted(event_display_images) :
            pos += 1
            
            # Axes
            ax = fig.add_subplot(nrow,ncol,pos)
            indx = event_display_images.index((ad,im))
            nickn = ""
            if indx < len(self.image_addresses):
                nickn = self.image_nicknames[indx]
                nickn+=": "
            ax.set_title( "%s%s" % (nickn,ad) )

            # AxesImage
            axim = plt.imshow( im, origin='lower' )
            self.axims.append( axim )
        
            cbar = plt.colorbar(axim,pad=0.02,shrink=0.78) 
            self.caxes.append( cbar.ax )
            
            self.orglims = axim.get_clim()
            # min and max values in the axes are


            #grid[pos-1].set_title("%s"%(ad))
            #axesim = grid[pos-1].imshow( im )
            #grid.cbar_axes[pos-1].colorbar(axesim)


            #self.plotter.gridplot( image, title, fignum, (2,ncols,axspos) )
        for cax in self.caxes :
            print cax
        for axim in self.axims :
            print axim

        #self.plotter.gridplot(self.image_diff,"Difference", fignum, (2,ncols,axspos+1) )
        #self.plotter.suptitle(fignum,"Event#%d"%self.n_events)
        #plt.draw()
                    
        plt.draw()


        # save the average data image (numpy array)
        # binary file .npy format
        if self.output_file is not None :

            for ad, im in event_display_images :
                fname = self.output_file.split('.')
                label = ad.replace("|","_")
                label = label.replace(" ","")
                filnamn = ''.join( (fname[0],"%s_ev%d."%(label,self.n_events),fname[-1]) )
                print filnamn
                
                if ".npy" in self.output_file :
                    np.save(filnamn, im)
                    print "saving to ", filnamn
                elif ".txt" in self.output_file :
                    np.savetxt(filnamn, im) 
                    print "saving to ", filnamn
                else :
                    print "Output file does not have the expected file extension (.txt or .npy): ", fname[-1]
                    print "I'm not sure what file format to save it as. Please correct."
                    print "I'm not saving this event... "
        
                        

    # after last event has been processed. 
    def endjob( self, env ) :

        print "Done processing       ", self.n_events, " events"
        print "Range defining images: %f (lower) - %f (upper)" % (self.thr_low_image, self.thr_high_image)
        print "Range defining darks: %f (lower) - %f (upper)" %  (self.thr_low_dark, self.thr_high_dark)
        
        for addr in self.image_addresses:
            print "# Signal images from %s = %d "% (addr, self.n_img[addr])
            print "# Dark images from %s = %d" % (addr, self.n_dark[addr])


            # plot the minimums and maximums
            xaxis = np.arange(self.n_events)
            plt.plot( xaxis, np.array(self.lolimits[addr]), "gv", xaxis, np.array(self.hilimits[addr]), "r^" )
            plt.title("Maxim (^)and lower (v) limits")
            #plt.plot( np.array(self.lolimits))
            #plt.plot( np.array(self.hilimits))

            # plot the average image
            av_good_img = self.image_data[addr]/self.n_img[addr]
            self.plotter.drawframe( av_good_img, "%s: Average of images above threshold"%addr,
                                    100+self.fignum[addr])

            if self.n_dark[addr]>0 :
                av_dark_img = self.dark_data[addr]/self.n_dark[addr]
                av_bkgsubtracted = av_good_img - av_dark_img 
                self.plotter.drawframe( av_dark_img, "%s: Average of images below threshold"%addr,
                                        200+self.fignum[addr] )
                self.plotter.drawframe( av_bkgsubtracted, "%s: Average background subtracted"%add,
                                        300+self.fignum[addr])

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



