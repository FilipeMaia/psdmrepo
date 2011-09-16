#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#   Module pyana_image_beta
#   pyana module with intensity threshold, plotting with matplotlib, allow rescale color plot
#
#   Example xtc file: /reg/d/psdm/sxr/sxrcom10/xtc/e29-r0603-s00-c00.xtc 
#
#   To run: pyana -m mypkg.pyana_image_beta <filename>
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
from utilities import ImageData


#---------------------
#  Class definition --
#---------------------
class  pyana_image_beta ( object ) :

    #--------------------
    #  Class variables --
    #--------------------
    
    #----------------
    #  Constructor --
    #----------------
            
    # initialize
    def __init__ ( self,
                   source = None,
                   quantities = None,
                   dark_img_file = None,
                   pedestal_file = None,
                   out_avg_file = None,
                   out_shot_file = None,
                   threshold = None ):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param source           string, Address of Detector-Id|Device-ID
        @param quantities       string, list of quantities to plot: image spectrum projX projY projR
        @param dark_img_file    name (base) of dark image file (numpy array) to be loaded, if any
        @param pedestal_file    full path/name of pedestal file (same as translator is using). Alternative! to dark image file. 
        @param out_avg_file     name (base) of output file for average image (numpy array)
        @param out_shot_file    name (base) of numpy array file for selected single-shot events
        @param threshold        threshold intensity and threshold area (xlow:xhigh,ylow:yhigh)
        @param plot_every_n     int, Draw plot for every N event? (if None or 0, don't plot till end) 
        @param accumulate_n     Accumulate all (0) or reset the array every n shots
        """

        # initializations from argument list
        opt = PyanaOptions()

        self.source = opt.getOptString(source)
        self.quantities = opt.getOptStringsDict(quantities)
        self.darkfile = opt.getOptString(dark_img_file)
        self.pedestalfile = opt.getOptString(pedestal_file)
        self.out_avg_file = opt.getOptString(out_avg_file)
        self.out_shot_file = opt.getOptString(out_shot_file)
        threshold_string = opt.getOptStrings(threshold)

        if self.source is None:
            print "WARNING, not input address has been given. Exiting..."
            return

        #if (self.quantities is None) or (len( self.quantities ) == 0) :
        #    print "WARNING, no quantities for plotting. Exiting..."
        #    return

        
        
        self.threshold = None
        if len(threshold_string)>0:
            self.threshold = Threshold()
            self.threshold.value = opt.getOptFloat(threshold_string[0])

        if len(threshold_string)>1:
            self.threshold.area = np.array([0.,0.,0.,0.])            

            intervals = threshold_string[1].strip('()').split(',')
            xrange = intervals[0].split(":")
            yrange = intervals[1].split(":")
            self.threshold.area[0] = float(xrange[0])
            self.threshold.area[1] = float(xrange[1])
            self.threshold.area[2] = float(yrange[0])
            self.threshold.area[3] = float(yrange[1])

            

#        # Set up the plotter's frame by hand, since
#        # we need to also tell it about thresholds, and vlimits
#        self.plotter.add_frame(self.source)
#        self.plotter.frame[self.source].threshold = self.threshold
#        self.plotter.frame[self.source].vmin = self.plot_vmin
#        self.plotter.frame[self.source].vmax = self.plot_vmax

        # ----
        # initializations of other class variables
        # ----
        # to keep track
        self.n_shots = 0
        self.accu_start = 0
        self.n_good = 0
        self.n_dark = 0

        # other pointers
        self.cspad = None
        self.mydata = None

        self.radii = None
        self.thetas = None
        self.r_indices = None
        self.th_indices = None

        # Dictionary mapping a function to each quantity plot option
        self.funcdict_bookplot = { 'image'    : self.book_image_plot,
                                   'roi'      : self.book_roi_plot,
                                   'spectrum' : self.book_spectrum_plot,
                                   'projX'    : self.book_projX_plot,
                                   'projY'    : self.book_projY_plot,
                                   'projR'    : self.book_projR_plot }
        
        # Dictionary mapping the source to the right get function
        self.funcdict_getimage = { 'Cspad' :     self.get_cspad_image,
                                   'Cspad2x2':   self.get_cspad2x2_image,
                                   'Princeton' : self.get_princeton_image,
                                   'pnCCD' :     self.get_pnccd_image,
                                   'Fccd' :      self.get_fccd_image, 
                                   'TM6740' :    self.get_other_image }
        
        
        # accumulate image data 
        self.sum_good_images = None
        self.sum_dark_images = None

        # load dark image from file
        self.dark_image = None
        try: 
            self.dark_image = np.load(self.darkfile)
            print "Dark Image %s loaded from %s" %(str(self.dark.image.shape), self.darkfile)
            print "Darks will be subtracted from displayed images"
        except:
            print "No dark image loaded"
            pass

        ## load dark from pedestal file:
        #try:
        #    # pedestals txt file is (4*8*185)=5920 (lines) x 388 (columns)
        #    array = np.loadtxt(self.pedestalfile)
        #    self.pedestals = np.reshape(array, (4,8,185,388) )
        #    print "Pedestals has been loaded from %s"%self.pedestalfile
        #    print "Pedestals will be subtracted from displayed images"
        #except:
        #    print "No pedestals loaded"
        #    pass
         

    # this method is called at an xtc Configure transition
    def beginjob ( self, evt, env ) : 

        self.mydata = ImageData(self.source)

        if self.source.find("Cspad")>0 :
            config = env.getConfig(xtc.TypeId.Type.Id_CspadConfig, self.source)
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
            self.cspad.set_pedestals( self.pedestalfile )
            


    # process event/shot data
    def event ( self, evt, env ) :

        # this one counts every event
        self.n_shots+=1

        if evt.get('skip_event'):
            print "Told to skip this event..."
            return


        #################################
        ## Get data from XTC file      ##
        #################################

        # call the relevant function to get the image (faster than if-else clauses)
        label = self.source.split('|')[1].split('-')[0]

        the_image = self.funcdict_getimage[label]([evt, env])
        if the_image is None:
            print "No image from ", label
            return


        ##################################################
        # subtract background if provided from a file
        ##################################################
        if self.dark_image is not None: 
            the_image = the_image - self.dark_image 


        ##################
        # apply a filter #
        ##################
        #passed = self.apply_filter():
        #if not passed:
        #    return
            
        # -----
        # Passed the threshold filter. Add this to the sum
        # -----
        self.n_good+=1
        if self.sum_good_images is None :
            self.sum_good_images = np.float_(the_image)
        else :
            self.sum_good_images += the_image

        # -----
        # Draw this event.
        # -----

        #masked_im = np.ma.masked_array(the_image, mask=(the_image==0) )

        # compute radial coordinates, in case it'll be needed
        if self.radii is None: 
            self.compute_polarcoordinates(the_image.shape)

        for quantity in self.quantities :
            option = self.quantities[quantity]
            print "pyana_image_beta.py: Plotting ", quantity, option
            self.funcdict_bookplot[quantity](the_image,option)

        # add mydata to event's plot_data 
        plot_data = evt.get('plot_data')
        if plot_data is None:
            plot_data = []
        plot_data.append( self.mydata ) 
        evt.put( plot_data ,'plot_data' )
        evt.put( True, 'show_event')
            
    # after last event has been processed. 
    def endjob( self, evt, env ) :

        print "Done processing       ", self.n_shots, " events"        
        
        title = self.source

        # keep a list of images 
        event_display_images = []

        average_image = None
        if self.n_good > 0 :
            label = "Average of %d shots"%self.n_good

            average_image = self.sum_good_images/self.n_good 
            event_display_images.append( (label, average_image ) )

        rejected_image = None
        if self.n_dark > 0 :
            label = "Average of %d dark/rejected shots"%self.n_dark

            rejected_image = self.sum_dark_images/self.n_dark
            event_display_images.append( (label, rejected_image ) )
            
        if self.dark_image is not None :
            label = "Dark image from input file"
            event_display_images.append( ("Dark image from file", self.dark_image ) )

            
        ## Just one plot
        # newmode = self.plotter.draw_figure(the_image,title, fignum=self.mpl_num, showProj=True)

        if len(event_display_images) == 0:
            print "No images to display from ", self.source
            return
                
        #evt.put( self.data, 'data_cspad')

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


#        for title,image in event_display_images: 
#            self.plotter.add_frame(title)
#            self.plotter.frame[title].threshold = self.threshold
#            self.plotter.frame[title].vmin = self.plot_vmin
#            self.plotter.frame[title].vmax = self.plot_vmax

        # try something new: 
        list_for_plotting = evt.get( 'event_display_images')
        if list_for_plotting :
            list_for_plotting.extend( event_display_images )
        else :
            evt.put(event_display_images, 'event_display_images')
            
            
                    

    # -----------------
    # Other functions
    # -----------------
    def apply_filter(self):
        roi = None
        if self.threshold is not None:            

            # first, assume we're considering the whole image
            roi = the_image

            if self.threshold.area is not None:
                # or set it to the selected threshold area
                roi = the_image[self.threshold.area[2]:self.threshold.area[3],   # rows (y-range)
                                  self.threshold.area[0]:self.threshold.area[1]]   # columns (x-range)

            
            dims = np.shape(roi)

            maxbin = roi.argmax()
            maxvalue = roi.ravel()[maxbin] 
            maxbin_coord = np.unravel_index(maxbin,dims)
            #print "CsPad: Max value of ROI %s is %d, in bin %d == %s"%(dims,maxvalue,maxbin,maxbin_coord)

            #minbin = roi.argmin()
            #minvalue = roi.ravel()[minbin] 
            #minbin_coord = np.unravel_index(minbin,dims)
            

            print "pyana_image_beta: shot#%d "%self.n_shots ,
            if maxvalue < self.threshold.value :
                print " skipped (%.2f < %.2f) " % (maxvalue, float(self.threshold.value))
                
                # collect the rejected shots before returning to the next event
                self.n_dark+=1
                if self.sum_dark_images is None :
                    self.sum_dark_images = np.float_(the_image)
                else :
                    self.sum_dark_images += the_image

                evt.put(True,'skip_event') # tell downstream modules to skip this event
                return
            else :
                print "accepted (%.2f > %.2f) " % (maxvalue, float(self.threshold.value))
        
    def compute_polarcoordinates(self,size,nbins=100,origin=None):
        nx, ny = size
        if origin is None:
            origin = (nx/2, ny/2)
                            
        xx,yy = np.meshgrid( (np.arange(nx)-origin[0]), (np.arange(ny)-origin[1]) )
        # 2D array of x-coordintes, and 2D array of y-coordinates of each pixel
        
        rr = np.sqrt(xx**2 + yy**2)
        # 2D array of radial coordinate of each pixel
        self.radii = rr.ravel()
        
        ttheta = np.arctan2(yy,xx)
        # 2D array of angular coordinate of each pixel
        self.thetas = ttheta.ravel()

        # binned:
        self.rbins = np.linspace( self.radii.min(), self.radii.max(), nbins+1)
        self.rbins[-1]+=1
        self.r_indices = np.digitize(self.radii,self.rbins)

        self.tbins = np.linspace( self.thetas.min(), self.thetas.max(), nbins+1)
        self.tbins[-1]+=1
        self.th_indices = np.digitize(self.thetas,self.tbins)

    
    def book_image_plot(self,image,options=None):
        self.mydata.image = image

    def book_roi_plot(self,image=None,options=None):
        self.mydata.roi = None
        if options is not None:
            self.mydata.roi = map(int,options) # a list
            
    def book_spectrum_plot(self,image,options=None):
        flat = image.ravel()
        compressed = flat[ flat!=0 ]
        self.mydata.spectrum = compressed

        if options is None:
            return

        print options
        # or should we bin it? 
        nbins = None
        range = None
        if options is not None:

            min   = compressed.min()
            if options[0] != '' : min = int(options[0])

            max   = compressed.max()
            if options[1] != '' : max = int( options[1] )

            nbins = len(compressed)
            if options[2] != '' : nbins = int( options[2] )

            range = (min,max)
            
        hist, bins = np.histogram( compressed, range=range, bins=nbins )
        self.mydata.spectrum = np.vstack((bins[:-1],hist))

    def book_projX_plot(self,image,options=None): 
        # projX (horizontal axis, columns)
        # for each column, average of elements
        self.mydata.projX = np.ma.average(image,0).data

    def book_projY_plot(self,image,options=None): 
        # projY (vertical axis, rows)
        # for each row, average of elements
        self.mydata.projY = np.ma.average(image,1).data

    def book_projR_plot(self,image,options=None):
        # Make a histogram of Intensity binned by radial coordinate 
        image_flat = image.ravel()

        intensities = [] 
        for i in xrange(1, len(self.rbins)):
            intensities.append(image_flat[self.r_indices==i] )
            
        mean_intensity = map( np.ma.mean, intensities )

        self.mydata.projR = np.array(mean_intensity)
        self.mydata.binsR =self.rbins
 
    def book_projTheta_plot(self,image,options=None):
        image_flat = image.ravel()

        intensities = [] 
        for i in xrange(1, len(self.tbins)):
            intensities.append(image_flat[self.th_indices==i] )
            
        mean_intensity = map( np.ma.mean, intensities )

        self.mydata.projTheta = mean_intensity
        self.mydata.binsTheta =self.tbins


    def get_cspad2x2_image(self, arg):
        """small Cspad2x2
        """
        evt = arg[0]
        env = arg[1]

        if self.cspad is None:
            print "get_cspad_image is returning in event ", self.n_shots
            return

        image = None
        quads = evt.get(xtc.TypeId.Type.Id_Cspad2x2Element, self.source)
        if quads is not None: 
            image = self.cspad.CsPad2x2Image(quads)
        else: 
            print '*** cspad information is missing ***'
            return

        return image

    def get_cspad_image(self, arg):
        """ full Cspad
        """
        evt = arg[0]
        env = arg[1]

        if self.cspad is None:
            print "get_cspad_image is returning in event ", self.n_shots
            return

        print "Trying to get Full CsPad from ", self.source
        quads = evt.getCsPadQuads(self.source, env)
        image = None
        if quads is not None:         
            image = self.cspad.CsPadImage(quads)
        else :
            print '*** cspad information is missing ***'
            print self.source
            return
        
        return image


    def get_princeton_image(self, arg):
        evt = arg[0]
        env = arg[1]
        try: 
            frame = evt.getPrincetonValue(self.source, env)
            return frame.data()
        except: 
            print "No frame from ", self.source, " in shot#", self.n_shots
            return
        
    def get_pnccd_image(self,arg):
        evt = arg[0]
        env = arg[1]
        try: 
            frame = evt.getPnCcdValue(self.source, env)
            return frame.data()
        except: 
            print "No frame from ", self.source, " in shot#", self.n_shots

    def get_fccd_image(self,arg):
        evt = arg[0]
        try: 
            frame = evt.getFrameValue(self.source)
            image = frame.data()
            # convert to 16-bit integer
            image.dtype = np.uint16
            return image
        except: 
            print "No frame from ", self.source, " in shot#", self.n_shots
            return 
            
    def get_other_image(self, arg):
        evt = arg[0]
        try: 
            frame = evt.getFrameValue(self.source)
            return frame.data()
        except: 
            print "No frame from ", self.source, " in shot#", self.n_shots
            return 
