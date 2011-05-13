#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#   Module pyana_cspad
#   pyana module with intensity threshold, plotting with matplotlib, allow rescale color plot
#
#   Example xtc file: /reg/d/psdm/sxr/sxrcom10/xtc/e29-r0603-s00-c00.xtc 
#
#   To run: pyana -m mypkg.pyana_cspad <filename>
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

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc
from cspad import CsPad

from utilities import Plotter
from utilities import Threshold
from utilities import PyanaOptions

#---------------------
#  Class definition --
#---------------------
class  pyana_cspad ( object ) :

    #--------------------
    #  Class variables --
    #--------------------
    
    #----------------
    #  Constructor --
    #----------------
            
    # initialize
    def __init__ ( self,
                   img_sources = None,
                   dark_img_file = None,
                   out_img_file = None,
                   plot_vrange = None,
                   threshold = None,
                   thr_area = None,
                   plot_every_n = None,
                   fignum = "1" ):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param img_sources      string, Address of Detector-Id|Device-ID
        @param dark_img_file    filename, Dark image file to be loaded, if any
        @param out_img_file      filename (If collecting: write to this file)
        @param plot_vrange      range=vmin,vmax of values for plotting (pixel intensity)
        @param threshold        lower threshold for image intensity in threshold area of the plot
        @param thr_area         range=xmin,xmax,ymin,ymax defining threshold area
        @param plot_every_n     int, Draw plot for every N event? (if None or 0, don't plot till end) 
        @param fignum           int, Matplotlib figure number
        """

        # initializations from argument list
        opt = PyanaOptions()

        self.img_sources = opt.getOptString(img_sources)
        print "Using img_sources = ", self.img_sources

        self.plot_every_n = opt.getOptInteger(plot_every_n)
        print "Using plot_every_n = ", self.plot_every_n

        self.mpl_num = opt.getOptInteger(fignum)
        print "Using matplotlib fignum ", self.mpl_num

        self.darkfile = opt.getOptString(dark_img_file)
        print "Using dark image file: ", self.darkfile
                
        self.out_img_file = opt.getOptString(out_img_file)
        print "Using out_img_file: ", self.out_img_file

        self.plot_vmin = None
        self.plot_vmax = None
        if plot_vrange is not None and plot_vrange is not "" : 
            self.plot_vmin = float(plot_vrange.split(",")[0])
            self.plot_vmax = float(plot_vrange.split(",")[1])
            print "Using plot_vrange = %f,%f"%(self.plot_vmin,self.plot_vmax)

        self.plotter = Plotter()
        #if self.plot_every_n > 0 : self.plotter.display_mode = 1 # interactive 

        self.threshold = Threshold()
        self.threshold.minvalue = opt.getOptFloat(threshold)
        if self.threshold.minvalue :
            tarea = opt.getOptString(thr_area)
            if tarea is not None: 
                tarea = np.array([0.,0.,0.,0.])
                for i in range (4) :
                    tarea[i] = float(thr_area.split(",")[i])                    
                self.threshold.area = tarea
            print "Using threshold value ", self.threshold.minvalue
            print "Using threshold area ", self.threshold.area
        else :
            del self.threshold
            self.threshold = None
            
        # pass this Threshold to the Plotter
        self.plotter.threshold = self.threshold

        # ----
        # initializations of other class variables
        # ----

        # to keep track
        self.n_shots = 0
        self.n_img = 0

        # accumulate image data above and below threshold
        self.sum_good_images = None
        self.sum_dark_images = None

        # load dark image from file
        self.dark_image = None
        if self.darkfile is None :
            print "No dark-image file provided. The images will not be background subtracted."
        else :
            print "Loading dark image from ", self.darkfile
            try: 
                self.dark_image = np.load(self.darkfile)
            except IOError:
                print "No dark file found, will compute darks on the fly"
                print "(will only work if a reasonable threshold is used)"
        
    # this method is called at an xtc Configure transition
    def beginjob ( self, evt, env ) : 

        config = env.getConfig(xtc.TypeId.Type.Id_CspadConfig, self.img_sources )
        if not config:
            print '*** cspad config object is missing ***'
            return
        
        quads = range(4)
        
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
        
        self.cspad = CsPad(config)


    # process event/shot data
    def event ( self, evt, env ) :

        self.images = []
        self.ititle = []

        # this one counts every event
        self.n_shots+=1

        # print a progress report
        if (self.n_shots%1000)==0 :
            print "Event ", self.n_shots
        
        quads = evt.getCsPadQuads(self.img_sources, env)
        if not quads :
            print '*** cspad information is missing ***'
            return
        
        # dump information about quadrants
        #print "Number of quadrants: %d" % len(quads)
        qimages = np.zeros((4, self.cspad.npix_quad, self.cspad.npix_quad ), dtype="uint16")

        for q in quads:
            
            #print "  Quadrant %d" % q.quad()
            #print "    virtual_channel: %s" % q.virtual_channel()
            #print "    lane: %s" % q.lane()
            #print "    tid: %s" % q.tid()
            #print "    acq_count: %s" % q.acq_count()
            #print "    op_code: %s" % q.op_code()
            #print "    seq_count: %s" % q.seq_count()
            #print "    ticks: %s" % q.ticks()
            #print "    fiducials: %s" % q.fiducials()
            #print "    frame_type: %s" % q.frame_type()
            #print "    sb_temp: %s" % map(q.sb_temp, range(4))
            
            # image data as 3-dimentional array
            data = q.data()
            #print "min and max of original array for quad#%d: %d, %d" %(q.quad(),np.min(data),np.max(data))
            
            qimage = self.cspad.CsPadElement(data, q.quad())
            qimages[q.quad()] = qimage


        # need to do this a better way:
        h1 = np.hstack( (qimages[0], qimages[1]) )
        h2 = np.hstack( (qimages[3], qimages[2]) )
        cspad_image = np.vstack( (h1, h2) )
        self.vmax = np.max(cspad_image)
        self.vmin = np.min(cspad_image)

        # subtract background if provided from a file
        if self.dark_image is not None: 
            cspad_image = cspad_image - self.dark_image 

        # apply threshold filter
        if self.threshold is not None:            
            #topval = np.max(cspad_image) # value of maximum bin
            topval = np.sum(cspad_image)/np.size(cspad_image) # average value of bins
            if self.threshold.area is not None:
                subset = cspad_image[self.threshold.area[0]:self.threshold.area[1],   # x1:x2
                                     self.threshold.area[2]:self.threshold.area[3]]   # y1:y2                
                #topval = np.max(subset) # value of maximum bin
                topval = np.sum(subset)/np.size(cspad_image) # average value of bins

            if topval < self.threshold.minvalue :
                print "skipping event #%d %.2f < %.2f " % \
                      (self.n_shots, topval, float(self.threshold.minvalue))

                # collect the rejected shots before returning to the next event
                if self.sum_dark_images is None :
                    self.sum_dark_images = np.float_(cspad_image)
                else :
                    self.sum_dark_images += cspad_image
                return
            else :
                print "accepting event #%d, vmax = %.2f > %.2f " % \
                      (self.n_shots, topval, float(self.threshold.minvalue))

        # -----
        # Passed the threshold filter. Add this to the sum
        # -----
        self.n_img+=1
        if self.sum_good_images is None :
            self.sum_good_images = np.float_(cspad_image)
        else :
            self.sum_good_images += cspad_image

        # -----
        # Draw this event.
        # -----
        if self.plot_every_n > 0 :
            title = "CsPad shot # %d" % self.n_shots
            if self.dark_image is not None:
                title = title + " (background subtracted) "
            
            if (self.n_shots%self.plot_every_n)==0 :
                print "plotting CsPad data for shot # ", self.n_shots
                self.plotter.drawframe(cspad_image,title, fignum=self.mpl_num, showProj = True)

        # check if plotter has changed its display mode. If so, tell the event
        switchmode = self.plotter.display_mode
        if switchmode is not None :
            evt.put(switchmode,'display_mode')
            if switchmode == 0 : self.plot_every_n = 0
            
    # after last event has been processed. 
    def endjob( self, env ) :

        print "Done processing       ", self.n_shots, " events"        
        
        if self.img_data is None :
            print "No image data found from source ", self.img_sources
            return

        # plot the average image
        average_image = self.img_data/self.n_img 
        print "the average intensity of average image ", np.mean(average_image)
        print "the highest intensity of average image ", np.max(average_image)
        print "the lowest intensity of average image ", np.min(average_image)

        self.plotter.drawframe(average_image,"Average of %d events" % self.n_img, fignum=self.mpl_num )        

        # save the average data image (numpy array)
        # binary file .npy format
        if self.out_img_file is not None :
            if ".npy" in self.out_img_file :
                print "saving to ",  self.out_img_file
                np.save(self.out_img_file, average_image)
            else :
                print "outputfile file does not have the required .npy ending..."
                svar = raw_input("Do you want to provide an alternative file name? ")
                if svar == "" :
                    print "Nothing saved"
                else :
                    if ".npy" not in svar:
                        print "I still don't like your file name, saving anyway..."
                    print "saving to ",  svar
                    np.save(svar, average_image)
                
