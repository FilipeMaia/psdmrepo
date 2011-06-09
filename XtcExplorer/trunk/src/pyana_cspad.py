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
from utilities import CsPadData

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
                   accumulate_n = "0",
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
        @param accumulate_n     Accumulate all (0) or reset the array every n shots
        @param fignum           int, Matplotlib figure number
        """

        # initializations from argument list
        opt = PyanaOptions()

        self.img_source = opt.getOptString(img_sources)
        print "Using img_sources = ", self.img_source

        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.accumulate_n = opt.getOptInteger(accumulate_n)
        self.mpl_num = opt.getOptInteger(fignum)

        self.darkfile = opt.getOptString(dark_img_file)
        print "Using dark image file: ", self.darkfile
                
        self.out_img_file = opt.getOptString(out_img_file)
        print "Using outputfile: ", self.out_img_file

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
            

        # Set up the plotter's frame by hand, since
        # we need to also tell it about thresholds
        self.plotter.add_frame(self.img_source)
        self.plotter.frame[self.img_source].threshold = self.threshold

        # ----
        # initializations of other class variables
        # ----

        # to keep track
        self.n_shots = 0
        self.accu_start = 0
        self.n_good = 0
        self.n_dark = 0

        # accumulate image data 
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

        config = env.getConfig(xtc.TypeId.Type.Id_CspadConfig, self.img_source )
        if not config:
            print '*** cspad config object is missing ***'
            return
        
        quads = range(4)

        print 
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
        print
        
        self.cspad = CsPad(config)

        self.data = CsPadData(self.img_source)

    # process event/shot data
    def event ( self, evt, env ) :

        self.images = []
        self.ititle = []

        # this one counts every event
        self.n_shots+=1

        # print a progress report
        if (self.n_shots%1000)==0 :
            print "Event ", self.n_shots
        
        quads = evt.getCsPadQuads(self.img_source, env)
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

        # mask out hot pixels (16383)
        cspad_image_masked = np.ma.masked_greater_equal( cspad_image,16383 )
        cspad_image = np.ma.filled(cspad_image_masked, 0)
        
        self.vmax = np.max(cspad_image)
        self.vmin = np.min(cspad_image)

        # subtract background if provided from a file
        if self.dark_image is not None: 
            cspad_image = cspad_image - self.dark_image 

        # threshold filter
        roi = None
        if self.threshold is not None:            

            # first, assume we're considering the whole image
            roi = cspad_image

            if self.threshold.area is not None:
                # or set it to the selected threshold area
                roi = cspad_image[self.threshold.area[2]:self.threshold.area[3],   # rows (y-range)
                                  self.threshold.area[0]:self.threshold.area[1]]   # columns (x-range)

            
            dims = np.shape(roi)
            maxbin = roi.argmax()
            maxvalue = roi.ravel()[maxbin] 
            maxbin_coord = np.unravel_index(maxbin,dims)
            print "CsPad: Max value of ROI %s is %d, in bin %d == %s"%(dims,maxvalue,maxbin,maxbin_coord)
            

            if maxvalue < self.threshold.minvalue :
                print "skipping event #%d %.2f < %.2f " % \
                      (self.n_shots, maxvalue, float(self.threshold.minvalue))

                # collect the rejected shots before returning to the next event
                self.n_dark+=1
                if self.sum_dark_images is None :
                    self.sum_dark_images = np.float_(cspad_image)
                else :
                    self.sum_dark_images += cspad_image
                return
            else :
                print "accepting event #%d, vmax = %.2f > %.2f " % \
                      (self.n_shots, maxvalue, float(self.threshold.minvalue))

        # -----
        # Passed the threshold filter. Add this to the sum
        # -----
        self.n_good+=1
        if self.sum_good_images is None :
            self.sum_good_images = np.float_(cspad_image)
        else :
            self.sum_good_images += cspad_image

        # -----
        # Draw this event.
        # -----
        if self.plot_every_n > 0 and (self.n_shots%self.plot_every_n)==0 :
            title = "%s shot # %d" % (self.img_source,self.n_shots)
            if self.dark_image is not None:
                title = title + " (background subtracted) "
            
            
            #newmode = self.plotter.draw_figure(cspad_image,title, fignum=self.mpl_num, showProj=True)
            #if roi is not None:
            #    self.plotter.draw_figure(roi,title="Threshold region",fignum=self.mpl_num+1,
            #                             showProj=False,extent=self.threshold.area)
            newmode = None

            if roi is not None: 
                event_display_images = []
                event_display_images.append( (title, cspad_image ) )
                event_display_images.append( ("Region of Interest", roi ) )
                
                newmode = self.plotter.draw_figurelist(self.mpl_num,
                                                       event_display_images,
                                                       title="CsPad shot#%d"%self.n_shots,
                                                       showProj=True)

            else :
                # Just one plot
                newmode = self.plotter.draw_figure(cspad_image,title, fignum=self.mpl_num, showProj=True)
                
            if newmode is not None:
                # propagate new display mode to the evt object 
                evt.put(newmode,'display_mode')
                # reset
                self.plotter.display_mode = None
                
            
            
            # data for iPython
            self.data.image = cspad_image
            self.data.average = self.sum_good_images/self.n_shots
            self.data.dark = self.dark_image


            
    # after last event has been processed. 
    def endjob( self, evt, env ) :

        print "Done processing       ", self.n_shots, " events"        
        
        if self.sum_good_images is None :
            print "No good images collected!", self.img_source
        else :
            # plot the average image
            average_image = self.sum_good_images/self.n_good 
            print "the average intensity of average image ", np.mean(average_image)
            print "the highest intensity of average image ", np.max(average_image)
            print "the lowest intensity of average image ", np.min(average_image)
            
            self.plotter.draw_figure(average_image,"Average of %d events" % self.n_good, fignum=self.mpl_num )
            print "******* image data in endjob "
            print self.data.image

            evt.put( self.data, 'data_cspad')

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
                
