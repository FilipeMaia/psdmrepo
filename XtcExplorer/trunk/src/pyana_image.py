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

from utilities import Plotter
from utilities import PyanaOptions
from utilities import ImageData

#---------------------
#  Class definition --
#---------------------
class  pyana_image ( object ) :

    # initialize
    def __init__ ( self,
                   sources = None,
                   good_range = "0,999999",
                   dark_range = "-999999,0",
                   image_rotations = None,
                   image_shifts = None,
                   image_scales = None,
                   image_nicknames = None,
                   image_manipulations = None, 
                   output_file = None,
                   n_hdf5 = None ,
                   plot_every_n = None,
                   accumulate_n = None,
                   fignum = "1" ):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param sources          (list) address string of Detector-Id|Device-ID
        @param plot_every_n     Frequency for plotting. If n=0, no plots till the end
        @param accumulate_n     Not implemented yet
        @param fignum           Matplotlib figure number
        @param good_range       threshold values selecting images of interest
        @param dark_range       threshold values selecting dark images
        @param image_rotations  (list) rotation, in degrees, to be applied to image(s)
        @param image_shifts     (list) shift, in (npixX,npixY), to be applied to image(s)
        @param image_scales     (list) scale factor to be applied to images
        @param image_nicknames  (list) nicknames for plot titles
        @param output_file      filename (If collecting: write to this file)
        @param n_hdf5           if output file is hdf5, combine n events in each output file. 
        """

        opt = PyanaOptions() # convert option string to appropriate type
        self.plot_every_n  =  opt.getOptInteger(plot_every_n)
        self.mpl_num = opt.getOptInteger(fignum)

        self.sources = opt.getOptStrings(sources)
        print sources, self.sources
        nsources = len(self.sources)
        print "pyana_image, %d sources: " % nsources
        for sources in self.sources :
            print "  ", sources

        self.image_nicknames = []
        if image_nicknames is None:
            for i in range (0, len(self.sources) ):
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
                for source in self.sources :
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
                for source in self.sources :
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

        self.image_manipulations = None
        if image_manipulations is not None:
            if image_manipulations == "" or image_manipulations == "None" :
                self.image_manipulations = None
            else :    
                self.image_manipulations = image_manipulations


        self.output_file = output_file
        if output_file == "" or output_file == "None" :
            self.output_file = None
        print "Using output_file: ", self.output_file

                
        # ranges
        tli = str(good_range).split(",")[0]
        thi = str(good_range).split(",")[1]
        tld = str(dark_range).split(",")[0]
        thd = str(dark_range).split(",")[1]

        self.thr_low_image = float( tli )
        self.thr_high_image = float( thi )
        self.thr_low_dark = float( tld )
        self.thr_high_dark = float( thd )
        print "Using good_range = %s " % good_range
        print "  (thresholds =  %d (low) and %d (high) " % (self.thr_low_image, self.thr_high_image)
        print "Using dark_range = %s " % dark_range
        print "  (thresholds =  %d (low) and %d (high) " % (self.thr_low_dark, self.thr_high_dark)

        self.n_hdf5 = None
        if n_hdf5 is not None :
            if n_hdf5 == "" or n_hdf5 == "None" :
                self.n_hdf5 = None
            else :
                self.n_hdf5 = int(n_hdf5)

        # to keep track
        self.n_shots = None

        # averages
        self.image_data = {}
        self.dark_data = {}
        self.minmax = {}
        self.fignum = {}
        self.n_img = {}
        self.n_dark = {}
        for addr in self.sources :
            self.image_data[addr] = None
            self.dark_data[addr] = None

            # these will be plotted too
            self.minmax[addr] = []

            self.fignum[addr] = self.mpl_num*10 + 100*self.sources.index(addr)

            self.n_img[addr] = 0
            self.n_dark[addr] = 0

        # output file
        self.hdf5file = None
        if self.output_file is not None :
            if ".hdf5" in self.output_file  and self.n_hdf5 is None:
                print "opening %s for writing" % self.output_file
                self.hdf5file = h5py.File(self.output_file, 'w')

        self.plotter = Plotter()


    def beginjob ( self, evt, env ) : 

        self.n_shots = 0
        self.n_accum = 0

        self.data = {}
        for source in self.sources:
            self.data[source] = ImageData(source)


    # process event/shot data
    def event ( self, evt, env ) :

        # this one counts every event
        self.n_shots+=1

        if evt.get('skip_event') :
            return

        # new hdf5-file every N events
        if self.output_file is not None :
            if ".hdf5" in self.output_file and self.n_hdf5 is not None:
                if (self.n_shots%self.n_hdf5)==1 :
                    start = self.n_shots # this event
                    stop = self.n_shots+self.n_hdf5-1
                    self.sub_output_file = self.output_file.replace('.hdf5',"_%d-%d.hdf5"%(start,stop) )
                    print "opening %s for writing" % self.sub_output_file
                    self.hdf5file = h5py.File(self.sub_output_file, 'w')

        # for each event, collect a list of images to be plotted 
        event_display_images = []

        # get the requested images
        for addr in self.sources :

            frame = None
            if addr.find("Princeton")>0 :
                frame = evt.getPrincetonValue(addr, env)
            if addr.find("pnCCD")>0 :
                frame = evt.getPnCcdValue(addr, env)
            else :
                frame = evt.getFrameValue(addr)

            if not frame :
                print "No frame from ", addr
                continue

            # image is a numpy array (pixels)
            image = np.float_(frame.data())

            # check that it has dimensions as expected from a camera image
            dim = np.shape( image )
            if len( dim )!= 2 :
                print "Unexpected dimensions of image array from %s: %s" % (addr,dim)

            # collect min and max intensities of images from this detector
            self.minmax[addr].append( [image.min(), image.max()] )

            # ---------------------------------------------------------------------------------------
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


            # ---------------------------------------------------------------------------------------
            # Apply shift, rotation, scaling of this image if needed:
            if self.image_rotations is not None:
                rotatedimage = interpol.rotate( image, self.image_rotations[addr], reshape=False )                
                image = rotatedimage

            if self.image_shifts is not None:
                #implement this
                shiftx, shifty = self.image_shifts[addr]
                shiftedimage = np.roll(image,shiftx,0)
                shiftedimage = np.roll(image,shifty,1)
                image = shiftedimage
                
            if self.image_scales is not None:
                #implement this!
                pass


            # ---------------------------------------------------------------------------------------
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


            # ---------------------------------------------------------------------------------------
            # Here's where we add the raw (or subtracted) image to the list for plotting
            event_display_images.append( (addr, image) )

            # This is for use by ipython
            self.data[addr].image   = image
            if self.n_img[addr] > 0 :
                self.data[addr].average = self.image_data[addr]/self.n_img[addr]
            if self.n_dark[addr] > 0 :
                self.data[addr].dark    = self.dark_data[addr]/self.n_dark[addr]
            
            
        if len(event_display_images)==0 :
            return
        
        if self.image_manipulations is not None: 
            for i in range ( 0, len(event_display_images) ):

                if "Diff" in self.image_manipulations :
                    ad1,im1 = event_display_images[i]
                    ad2,im2 = event_display_images[i-1]
                    lb1 = self.image_nicknames[i]
                    lb2 = self.image_nicknames[i-1]
                    event_display_images.append( ("Diff %s-%s"%(lb1,lb2), im1-im2) )

                if "FFT" in self.image_manipulations :
                    F = np.fft.fftn(im1-im2)
                    event_display_images.append( \
                        ("FFT %s-%s"%(lb1,lb2), np.log(np.abs(np.fft.fftshift(F))**2) ) )


        # -----------------------------------
        # Draw images from this event
        # -----------------------------------
        if self.plot_every_n != 0 and (self.n_shots%self.plot_every_n)==0 :
            newmode = self.plotter.draw_figurelist(self.mpl_num,
                                                   event_display_images,
                                                   title="Cameras shot#%d"%self.n_shots,
                                                   showProj=True)
            if newmode is not None:
                # propagate new display mode to the evt object 
                evt.put(newmode,'display_mode')
                # reset
                self.plotter.display_mode = None

            # convert dict to a list:
            data_image = []
            for source in self.sources :
                data_image.append( self.data[source] )
                # give the list to the event object
                evt.put( data_image, 'data_image' )
                                                            
            
        # -----------------------------------
        # Saving to file
        # -----------------------------------
        if self.hdf5file is not None :
            # save this event as a group in hdf5 file:
            group = self.hdf5file.create_group("Event%d" % self.n_shots)
        
        # save the average data image (numpy array)
        # binary file .npy format
        if self.output_file is not None :

            for ad, im in event_display_images :

                fname = self.output_file.split('.')
                label = ad.replace("|","_")
                label = label.replace(" ","")
                filnamn = ''.join( (fname[0],"%s_ev%d."%(label,self.n_shots),fname[-1]) )

                # HDF5
                if self.hdf5file is not None :
                    # save each image as a dataset in this event group
                    dset = group.create_dataset("%s"%ad,data=im)

                # Numpy array
                elif ".npy" in self.output_file :
                    np.save(filnamn, im)
                    print "saving to ", filnamn
                elif ".txt" in self.output_file :
                    np.savetxt(filnamn, im) 
                    print "saving to ", filnamn
                        
                else :
                    print "Output file does not have the expected file extension: ", fname[-1]
                    print "Expected hdf5, txt or npy. Please correct."
                    print "I'm not saving this event... "
        

    # after last event has been processed. 
    def endjob( self, evt, env ) :

        if self.hdf5file is not None :
            self.hdf5file.close()

        print "Done processing       ", self.n_shots, " events"
        print "Range defining images: %f (lower) - %f (upper)" % (self.thr_low_image, self.thr_high_image)
        print "Range defining darks: %f (lower) - %f (upper)" %  (self.thr_low_dark, self.thr_high_dark)
        
        for addr in self.sources:
            print "# Signal images from %s = %d "% (addr, self.n_img[addr])
            print "# Dark images from %s = %d" % (addr, self.n_dark[addr])

            # plot the average image
            av_good_img = self.image_data[addr]/self.n_img[addr]
            self.plotter.draw_figure( av_good_img, "%s: Average of images above threshold"%addr,
                                    fignum=self.fignum[addr])

            if self.n_dark[addr]>0 :
                av_dark_img = self.dark_data[addr]/self.n_dark[addr]
                av_bkgsubtracted = av_good_img - av_dark_img 
                self.plotter.draw_figure( av_dark_img, "%s: Average of images below threshold"%addr,
                                        fignum=self.fignum[addr]+1 )
                self.self.draw_figure( av_bkgsubtracted, "%s: Average background subtracted"%addr,
                                     fignum=self.fignum[addr]+2)

        plt.draw()
        

        # convert dict to a list:
        data_image = []
        for source in self.sources :
            data_image.append( self.data[source] )
            # give the list to the event object
            evt.put( data_image, 'data_image' )
            

