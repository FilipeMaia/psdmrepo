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
                   pedestal_file = None,
                   out_avg_file = None,
                   out_shot_file = None,
                   plot_vrange = None,
                   threshold = None,
                   plot_every_n = None,
                   accumulate_n = "0",
                   max_save = "100",
                   fignum = "1" ):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param img_sources      string, Address of Detector-Id|Device-ID
        @param dark_img_file    name (base) of dark image file (numpy array) to be loaded, if any
        @param pedestal_file    full path/name of pedestal file (same as translator is using).
                                Alternative! to dark image file. 
        @param out_avg_file     name (base) of output file for average image (numpy array)
        @param out_shot_file    name (base) of numpy array file for selected single-shot events
        @param plot_vrange      range (format vmin:vmax) of values for plotting (pixel intensity)
        @param threshold        threshold intensity and threshold area (xlow:xhigh,ylow:yhigh)
        @param plot_every_n     int, Draw plot for every N event? (if None or 0, don't plot till end) 
        @param accumulate_n     Accumulate all (0) or reset the array every n shots
        @param max_save         Maximum single-shot images to save
        @param fignum           int, Matplotlib figure number
        """

        # initializations from argument list
        opt = PyanaOptions()

        self.img_source = opt.getOptString(img_sources)
        print "Using img_sources = ", self.img_source

        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.accumulate_n = opt.getOptInteger(accumulate_n)

        self.max_save = opt.getOptInteger(max_save)
        self.mpl_num = opt.getOptInteger(fignum)

        self.darkfile = opt.getOptString(dark_img_file)
        if self.darkfile is not None: print "Input dark image file: ", self.darkfile

        self.pedestalfile = opt.getOptString(pedestal_file)
        if self.pedestalfile is not None: print "Using pedestal file: ", self.pedestalfile

        if self.darkfile is not None and self.pedestalfile is not None:
            print "... cannot use both! user-supplied dark image will be used. Pedestals will be ignored"
            self.pedestalfile = None
                
        self.out_avg_file = opt.getOptString(out_avg_file)
        if self.out_avg_file is not None: print "Output average image file: ", self.out_avg_file

        self.out_shot_file = opt.getOptString(out_shot_file)
        if self.out_shot_file is not None: print "Output shot image file: ", self.out_shot_file

        self.plot_vmin = None
        self.plot_vmax = None
        if plot_vrange is not None and plot_vrange is not "" : 
            self.plot_vmin = float(plot_vrange.strip("()").split(":")[0])
            self.plot_vmax = float(plot_vrange.strip("()").split(":")[1])
            print "Using plot_vrange = %.2f,%.2f"%(self.plot_vmin,self.plot_vmax)

        #if self.plot_every_n > 0 : self.plotter.display_mode = 1 # interactive 

        threshold_string = opt.getOptStrings(threshold)
        # format: 'value (xlow:xhigh,ylow:yhigh)', only value is required

        self.threshold = None
        if len(threshold_string)>0:
            self.threshold = Threshold()
            self.threshold.value = opt.getOptFloat(threshold_string[0])
            print "Using threshold value ", self.threshold.value
        if len(threshold_string)>1:
            self.threshold.area = np.array([0.,0.,0.,0.])            

            intervals = threshold_string[1].strip('[()]').split(',')
            xrange = intervals[0].split(":")
            yrange = intervals[1].split(":")
            self.threshold.area[0] = float(xrange[0])
            self.threshold.area[1] = float(xrange[1])
            self.threshold.area[2] = float(yrange[0])
            self.threshold.area[3] = float(yrange[1])

            print "Using threshold area ", self.threshold.area
            

        self.plotter = Plotter()
        self.plotter.settings(7,7)
        self.plotter.threshold = None
        if self.threshold is not None:
            self.plotter.threshold = self.threshold.value            
        self.plotter.vmin, self.plotter.vmax = self.plot_vmin, self.plot_vmax

        # ----
        # initializations of other class variables
        # ----

        # to keep track
        self.n_shots = 0
        self.accu_start = 0
        self.n_good = 0
        self.n_dark = 0
        self.n_saved = 0

        # accumulate image data 
        self.sum_good_images = None
        self.sum_dark_images = None

        # test
        self.sum_array = None

        self.dark_image = None

                
    # this method is called at an xtc Configure transition
    def beginjob ( self, evt, env ) : 

        config = env.getConfig(xtc.TypeId.Type.Id_CspadConfig, self.img_source)
        if not config:
            print '*** cspad config object is missing ***'
            return
                
        quads = range(4)
        sections = map(config.sections, quads)

        self.cspad = CsPad(sections)
        self.data = CsPadData(self.img_source)

        if self.pedestalfile is not None: 
            self.cspad.set_pedestals( self.pedestalfile )
            print "cspad images will be pedestal subtracted. Using pedestals file ", self.pedestalfile
        else :
            # load dark image from file
            try: 
                self.dark_image = np.load(self.darkfile)
                print "Dark Image %s loaded from %s" %(str(self.dark.image.shape), self.darkfile)
                print "Darks will be subtracted from displayed images"
            except:
                print "No dark image loaded"
                pass


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
        print "  sections      : %s" % str(sections)
        print


    # process event/shot data
    def event ( self, evt, env ) :

        if  ( self.accumulate_n != 0 and (self.n_good%self.accumulate_n)==0 ):
            self.sum_good_images = None
            self.sum_array = None
            self.n_good = 0

        if  ( self.accumulate_n != 0 and (self.n_dark%self.accumulate_n)==0 ):
            self.sum_dark_images = None
            self.n_dark = 0
            
        # this one counts every event
        self.n_shots+=1

        if evt.get('skip_event'):
            return


        self.images = []
        self.ititle = []

        # full Cspad
        cspad_image = None

        quads = evt.getCsPadQuads(self.img_source, env)
        if quads is not None:         
            cspad_image = self.cspad.get_detector_image(quads)
        else :
            # mini Cspad (only 2x2)
            quads = evt.get(xtc.TypeId.Type.Id_Cspad2x2Element, self.img_source)
            if quads is not None:         
                cspad_image = self.cspad.get_mini_image(quads)
            
        if not quads:
            print '*** cspad information is missing ***'
            return

        # mask out hot pixels (16383)
        cspad_image_masked = np.ma.masked_greater_equal( cspad_image,16383 )
        cspad_image = np.ma.filled(cspad_image_masked, 0)
        
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
            #print "CsPad: Max value of ROI %s is %d, in bin %d == %s"%(dims,maxvalue,maxbin,maxbin_coord)

            #minbin = roi.argmin()
            #minvalue = roi.ravel()[minbin] 
            #minbin_coord = np.unravel_index(minbin,dims)
            

            print "pyana_cspad: shot#%d "%self.n_shots ,
            if maxvalue < self.threshold.value :
                print " skipped (%.2f < %.2f) " % (maxvalue, float(self.threshold.value))
                
                # collect the rejected shots before returning to the next event
                self.n_dark+=1
                if self.sum_dark_images is None :
                    self.sum_dark_images = np.float_(cspad_image)
                else :
                    self.sum_dark_images += cspad_image

                evt.put(True,'skip_event') # tell downstream modules to skip this event
                return
            else :
                print "%d accepting event #%d, vmax = %.0f > %.0f ,hitrate: %.4f" % \
                      (env.subprocess(), self.n_shots, maxvalue,
                       float(self.threshold.value),float(self.n_good)/float(self.n_shots))
        # -----
        # Passed the threshold filter. Add this to the sum
        # -----
        self.n_good+=1
        if self.sum_good_images is None :
            self.sum_good_images = np.float_(cspad_image)
            self.sum_array = np.float_( self.cspad.pixel_array )
        else :
            self.sum_good_images += cspad_image
            self.sum_array += self.cspad.pixel_array
            
        # test

        # -----
        # Draw this event.
        # -----
        if self.plot_every_n > 0 and (self.n_shots%self.plot_every_n)==0 :

            # flag for pyana_plotter
            evt.put(True, 'show_event')
            
            title = "%s shot#%d"%(self.img_source,self.n_shots)

            # keep a list of images 
            event_display_images = []
            event_display_images.append( ("%s shot#%d"%(self.img_source,self.n_shots), cspad_image ) )

            if roi is not None:
                extent=self.threshold.area                
                event_display_images.append( ("Region of Interest", roi, extent ) )
                
            if self.dark_image is not None :
                title += " (bkg. subtr.)"
                event_display_images.append( ("Dark image", self.dark_image ) )
            elif self.n_dark > 1 :
                dark = self.sum_dark_images/self.n_dark
                event_display_images.append( ("%s (bkg. subtr.)"%self.img_source, cspad_image-dark ) )
                event_display_images.append( ("Dark (average of %d shots)"%self.n_dark, dark ) )
            else :
                event_display_images.append( ("%s average of %d shots"%(self.img_source,self.n_good), \
                                              self.sum_good_images/self.n_good) )

            ## Just one plot
            # newmode = self.plotter.draw_figure(cspad_image,title, fignum=self.mpl_num, showProj=True)
            newmode = self.plotter.draw_figurelist(self.mpl_num,
                                                   event_display_images,
                                                   title="",
                                                   showProj=True)
            
            if newmode is not None:
                # propagate new display mode to the evt object 
                evt.put(newmode,'display_mode')
                # reset
                self.plotter.display_mode = None
                
            
            
            # data for iPython
            self.data.image = cspad_image
            self.data.average = self.sum_good_images/self.n_good
            self.data.dark = self.dark_image

            # save this shot image (numpy array)
            # binary file .npy format
            if (self.out_shot_file is not None) and (self.n_saved < self.max_save):
                filename = self.out_shot_file
                if ".npy" not in filename :
                    filename += ".npy"
                    
                parts = filename.split('.')
                filename = "".join(parts[0:-1]) + "_shot%d."%self.n_shots + parts[-1]

                print "Saving this shot to file ", filename
                np.save(filename, cspad_image)
                self.n_saved += 1
                

            
    # after last event has been processed. 
    def endjob( self, evt, env ) :

        print "Done processing       ", self.n_shots, " events"        
        
        title = self.img_source

        # keep a list of images 
        event_display_images = []

        average_image = None
        if self.n_good > 0 :
            label = "Average of %d shots"%self.n_good

            average_image = self.sum_good_images/self.n_good 
            event_display_images.append( (label, average_image ) )

            average_array = self.sum_array / self.n_good
            np.savetxt("testfile.txt", average_array.reshape((4*8*185),388))
            np.save("testfile.npy", average_array.reshape((4*8*185),388))

        rejected_image = None
        if self.n_dark > 0 :
            label = "Average of %d dark/rejected shots"%self.n_dark

            rejected_image = self.sum_dark_images/self.n_dark
            event_display_images.append( (label, rejected_image ) )
            
        if self.dark_image is not None :
            label = "Dark image from input file"
            event_display_images.append( ("Dark image from file", self.dark_image ) )

            
        ## Just one plot
        # newmode = self.plotter.draw_figure(cspad_image,title, fignum=self.mpl_num, showProj=True)

        if len(event_display_images) == 0:
            print "No images to display from ", self.img_source
            return
                
        
        evt.put( self.data, 'data_cspad')

        # save the average data image (numpy array)
        # binary file .npy format
        if self.out_avg_file is not None :
            filename1 = self.out_avg_file
            filename2 = self.out_avg_file
            if ".npy" not in filename1 :
                filename1 += ".npy"

            parts = filename1.split('.')
            filename1 = "".join(parts[0:-1]) + "_lumi." + parts[-1]
            filename2 = "".join(parts[0:-1]) + "_dark." + parts[-1]

            if average_image is not None:
                print "Saving average of good shots to file ", filename1
                np.save(filename1, average_image)

            if rejected_image is not None: 
                print "Saving average of dark shots to file ", filename2
                np.save(filename2, rejected_image)


        self.plotter.draw_figurelist(self.mpl_num+1,
                                     event_display_images,
                                     title=title,
                                     showProj=True)
            
