#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#   Module pyana_image
#   pyana module with intensity threshold, plotting with matplotlib, allow rescale color plot
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
import matplotlib.pyplot as plt
#import h5py

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc

#-----------------------------
# Imports for local modules --
#-----------------------------
from cspad     import CsPad
from utilities import Plotter
from utilities import Threshold
from utilities import PyanaOptions
from utilities import ImageData
import algorithms as alg

#---------------------
#  Class definition --
#---------------------
class  pyana_image ( object ) :

    # initialize
    def __init__ ( self,
                   sources = None,       
                   inputdark = None,
                   threshold = None,
                   algorithms = None,
                   # plotting options:
                   plot_every_n = None,
                   accumulate_n = None,
                   plot_vrange = None,                   
                   show_projections = None,
                   # output options
                   outputfile = None,
                   n_hdf5 = None ,
                   max_save = "0",
                   fignum = "1" ):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param sources           address string of Detector-Id|Device-ID
        @param inputdark         name (base) of dark image file (numpy array) to be loaded, if any
        @param algorithms        (list of) algorithms to be applied to the image before plotting
        @param threshold         value (xmin:xmax,ymin:xmax) type

        @param plot_every_n      Frequency for plotting. If n=0, no plots till the end
        @param accumulate_n      Not implemented yet
        @param plot_vrange       value-range of interest
        @param show_projections  0,1 or 2, for projecting nothing, average or maximum 
 
        @param outputfile       filename (If collecting: write to this file)
        @param n_hdf5            if output file is hdf5, combine n events in each output file. 
        @param max_save          Maximum single-shot images to save
        """

        opt = PyanaOptions() # convert option string to appropriate type
        self.plot_every_n  =  opt.getOptInteger(plot_every_n)
        self.max_save = opt.getOptInteger(max_save)
        self.mpl_num = opt.getOptInteger(fignum)

        self.sources = opt.getOptStrings(sources)
        nsources = len(self.sources)
        print "pyana_image, %d sources: " % nsources
        for sources in self.sources :
            print "  ", sources

        self.darkfile = opt.getOptString(inputdark)
        if self.darkfile is not None: print "Input dark image file: ", self.darkfile

        self.algorithms = opt.getOptStrings(algorithms)
        print self.algorithms


        threshold_string = opt.getOptStrings(threshold)
        # format: 'value (xlow:xhigh,ylow:yhigh) type',
        
        self.threshold = None
        if len(threshold_string)>0:
            self.threshold = Threshold()
            self.threshold.value = opt.getOptFloat(threshold_string[0])
            print "Using threshold value ", self.threshold.value
        if len(threshold_string)>1:
            self.threshold.area = np.array([0.,0.,0.,0.])
                
            intervals = threshold_string[1].strip('()').split(',')
            xrange = intervals[0].split(":")
            yrange = intervals[1].split(":")
            self.threshold.area[0] = float(xrange[0])
            self.threshold.area[1] = float(xrange[1])
            self.threshold.area[2] = float(yrange[0])
            self.threshold.area[3] = float(yrange[1])
            
            print "Using threshold area ", self.threshold.area
            
            try:
                type = threshold_string[2]
                self.threshold.type = type
            except:
                pass

        self.n_hdf5 = opt.getOptInteger(n_hdf5)

        self.output_file = opt.getOptString(outputfile)
        #print "Output file name base: ", self.output_file

        self.plot_vmin = None
        self.plot_vmax = None
        if plot_vrange is not None and plot_vrange is not "" : 
            self.plot_vmin = float(plot_vrange.strip("()").split(":")[0])
            self.plot_vmax = float(plot_vrange.strip("()").split(":")[1])
            print "Using plot_vrange = %.2f,%.2f"%(self.plot_vmin,self.plot_vmax)

        # to keep track
        self.n_shots = None
        self.n_saved = 0

        # averages
        self.sum_good_images = {}
        self.sum_dark_images = {}
        self.n_good = {}
        self.n_dark = {}
        for addr in self.sources :
            self.sum_good_images[addr] = None
            self.sum_dark_images[addr] = None
            self.n_good[addr] = 0
            self.n_dark[addr] = 0


#        # output file
#        # can be npy (numpy binary) txt (numpy ascii) or hdf5
#        # only hdf5 need a file handler
#        self.hdf5file_all = None
#        self.hdf5file_events = None
#        if self.output_file is not None :
#            if ".hdf5" in self.output_file  and self.n_hdf5 is None:
#                print "opening HDF5 %s for writing of all events" % self.output_file
#                self.hdf5file = h5py.File(self.output_file, 'w')

        self.plotter = Plotter()        
        self.plotter.settings(7,7) # set default frame size
        self.plotter.threshold = None
        if self.threshold is not None:
            self.plotter.threshold = self.threshold
            self.plotter.vmin, self.plotter.vmax = self.plot_vmin, self.plot_vmax
            
        self.apply_dictionary = { 'rotate': alg.rotate,
                                  'shift' : alg.shift }
        
        #        # Set up the plotter's frame by hand, since
        #        # we need to also tell it about thresholds
        #        for source in self.sources :
        #            self.plotter.add_frame(source)
        #            self.plotter.frames[source].threshold = self.threshold
        
        self.cspad = {}

        self.configtypes = { 'Cspad2x2'  : xtc.TypeId.Type.Id_CspadConfig ,
                             'Cspad'     : xtc.TypeId.Type.Id_CspadConfig ,
                             'Opal'      : xtc.TypeId.Type.Id_Opal1kConfig,
                             '????'     : xtc.TypeId.Type.Id_FrameFexConfig,
                             'TM6740'    : xtc.TypeId.Type.Id_TM6740Config,
                             'pnCCD'     : xtc.TypeId.Type.Id_pnCCDconfig,
                             'Princeton' : xtc.TypeId.Type.Id_PrincetonConfig,
                             'Fccd ?? '  : xtc.TypeId.Type.Id_FrameFccdConfig,
                             'Fccd'      : xtc.TypeId.Type.Id_FccdConfig,
                             'PIM'       : xtc.TypeId.Type.Id_PimImageConfig
                             }

        self.datatypes = {'Cspad2x2'      : xtc.TypeId.Type.Id_Cspad2x2Element, 
                          'Cspad'         : xtc.TypeId.Type.Id_CspadElement,
                          'TM6740'        : xtc.TypeId.Type.Id_Frame,
                          'Opal'          : xtc.TypeId.Type.Id_Frame,
                          'pnCCD'         : xtc.TypeId.Type.Id_pnCCDframe,
                          'Princeton'     : xtc.TypeId.Type.Id_PrincetonFrame
                          }

    def beginjob ( self, evt, env ) : 

        self.n_shots = 0
        self.n_accum = 0

        self.data = {}
        for source in self.sources:
            self.data[source] = ImageData(source)

        for addr in self.sources :

            # pick out the device name from the address
            device = addr.split('|')[1].split('-')[0]
            config = env.getConfig( self.configtypes[device], addr )
            if not config:
                print '*** %s config object is missing ***'%addr
                return

            if addr.find('Cspad')>=0:
                quads = range(4)
                sections = map(config.sections, quads)
                
                self.cspad[addr] = CsPad(sections)

        ## load dark image from file
        #try:
        #    self.dark_image = 

    # process event/shot data
    def event ( self, evt, env ) :

        # this one counts every event
        self.n_shots+=1

        if evt.get('skip_event') :
            return

#        # new hdf5-file every N events
#        if self.output_file is not None :
#            if ".hdf5" in self.output_file and self.n_hdf5 is not None:
#                if (self.n_shots%self.n_hdf5)==1 :
#                    start = self.n_shots # this event
#                    stop = self.n_shots+self.n_hdf5-1
#                    self.sub_output_file = self.output_file.replace('.hdf5',"_%d-%d.hdf5"%(start,stop) )
#                    print "opening %s for writing" % self.sub_output_file
#                    self.hdf5file = h5py.File(self.sub_output_file, 'w')

        # for each event, collect a list of images to be plotted 
        event_display_images = []

        # get the requested images
        for addr in self.sources :
            image = None

            # pick out the device name from the address
            device = addr.split('|')[1].split('-')[0]
            frame = evt.get( self.datatypes[device], addr )

            if addr.find("Cspad2x2")>0 :
                # in this case 'frame' is the MiniElement
                # call cspad library to assemble the image
                image = self.cspad[addr].get_mini_image(frame)

            elif addr.find("Cspad")>0 :
                # in this case we need the specialized getter: 
                quads = evt.getCsPadQuads(addr, env)
                # then call cspad library to assemble the image
                image = self.cspad[addr].get_detector_image(quads)

            else:
                # all other cameras have simple arrays. 
                image = frame.data()

            if image is None:
                print "No frame image from ", addr, " in shot#", self.n_shots
                continue

            # image is a numpy array (pixels)
            if addr.find("Fccd")>0:
                # convert to 16-bit integer
                image.dtype = np.uint16
                

            # check that it has dimensions as expected from a camera image
            dim = np.shape( image )
            if len( dim )!= 2 :
                print "Unexpected dimensions of image array from %s: %s" % (addr,dim)

                                
            # ---------------------------------------------------------------------------------------
            # Apply shift, rotation, scaling of this image if needed:

            if self.algorithms is not None:

                # apply each algorithm in the list
                # (algorithms needs to be a list, because dictionary is unordered.)
                for algos in self.algorithms:

                    # parse the string "algorithm:parameters"
                    [algo_name, algo_pars] = algos.split(':')

                    # turn algo_pars into a list of floats (to be passed as args)
                    algo_pars = [float(n) for n in algo_pars.strip('([])').split(',')]

                    #print "algorithm: ", algo_name, "  parameters: ", algo_pars

                    # look up the dictionary and call the function with  this name
                    image = self.apply_dictionary[algo_name](image,*algo_pars)


            # prepare to apply threshold and see if this is a hit or a dark event
            isDark = False
            name = addr
            title = addr

            # ---------------------------------------------------------------------------------------
            # threshold filter
            roi = None
            if self.threshold is not None:
                # first, assume we're considering the whole image
                roi = image
                if self.threshold.area is not None:
                    # or set it to the selected threshold area
                    roi = image[self.threshold.area[2]:self.threshold.area[3],   # rows (y-range)
                                self.threshold.area[0]:self.threshold.area[1]]   # columns (x-range)
                dims = np.shape(roi)
                maxbin = roi.argmax()
                maxvalue = roi.ravel()[maxbin]
                maxbin_coord = np.unravel_index(maxbin,dims)

                if self.threshold.type == 'average':
                    maxvalue = roi.mean()
                
                if maxvalue < self.threshold.value :
                    isDark = True
                    name += "_dark"
                    title += " (dark)"
                else:
                    print "Not dark, max = ", maxvalue, maxbin_coord

            # ------------- DARK ----------------
            if isDark:
                self.n_dark[addr]+=1
                if self.sum_dark_images[addr] is None :
                    self.sum_dark_images[addr] = np.float_(image)
                else :
                    self.sum_dark_images[addr] += image
            else :
            # ------------- HIT ----------------
                self.n_good[addr]+=1
                if self.sum_good_images[addr] is None :
                    self.sum_good_images[addr] = np.float_(image)
                else :
                    self.sum_good_images[addr] += image
                        
            
            # ---------------------------------------------------------------------------------------
            # Here's where we add the raw (or subtracted) image to the list for plotting
            event_display_images.append( (name, title, image) )
            
            # This is for use by ipython
            self.data[addr].image   = image
            if self.n_good[addr] > 0 :
                self.data[addr].average = self.sum_good_images[addr]/self.n_good[addr]
            if self.n_dark[addr] > 0 :
                self.data[addr].dark    = self.sum_dark_images[addr]/self.n_dark[addr]
        

        if len(event_display_images)==0 :
            return
            
        #if self.image_manipulations is not None: 
        #    for i in range ( 0, len(event_display_images) ):
        #                    
        #        if "Diff" in self.image_manipulations :
        #            lb1,ad1,im1 = event_display_images[i]
        #            lb2,ad2,im2 = event_display_images[i-1]
        #            event_display_images.append( ("diff","Diff %s-%s"%(lb1,lb2), im1-im2) )
        #                        
        #        if "FFT" in self.image_manipulations :
        #            F = np.fft.fftn(im1-im2)
        #            event_display_images.append( \
        #                ("fft","FFT %s-%s"%(lb1,lb2), np.log(np.abs(np.fft.fftshift(F))**2) ) )
                                    
            
                    
        # -----------------------------------
        # Draw images from this event
        # -----------------------------------

        # only call plotter if this is the main thread
        if (env.subprocess()>0):
            return

        if self.plot_every_n != 0 and (self.n_shots%self.plot_every_n)==0 :

            # flag for pyana_plotter
            evt.put(True, 'show_event')

            # --- this works, but needs some tweaking to make it prettier
            #for (name,title,image) in event_display_images:
            #    self.plotter.add_frame(name,addr,(image,))
            #newmode = self.plotter.plot_all_frames(fignum=self.mpl_num,ordered=True)

            newmode = self.plotter.draw_figurelist(self.mpl_num,
                                                   event_display_images,
                                                   title="Cameras shot#%d"%self.n_shots,
                                                   showProj=2   )

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
        # Saving this event to file(s)
        # -----------------------------------
        if (self.output_file is not None) and (self.n_saved < self.max_save) : 
            self.n_saved += 1

            self.save_images( self.output_file, event_display_images, self.n_shots )

    

    # after last event has been processed. 
    def endjob( self, evt, env ) :

#        if self.hdf5file is not None :
#            self.hdf5file.close()

        print "Done processing       ", self.n_shots, " events"

        nsrc = 0
        for addr in self.sources:
            
            nsrc += 1

            # keep a list of images 
            event_display_images = []

            print "# Signal images from %s = %d "% (addr, self.n_good[addr])
            print "# Dark images from %s = %d" % (addr, self.n_dark[addr])

            average_image = None
            if self.n_good[addr] > 0 :
                name = "AvgHit_"+addr
                title = addr+" Average of %d hits"%self.n_good[addr]

                average_image = self.sum_good_images[addr]/self.n_good[addr]
                event_display_images.append( (name, title, average_image ) )

            rejected_image = None
            if self.n_dark[addr] > 0 :
                name = "AvgDark_"+addr
                title = addr+" Average of %d darks"%self.n_dark[addr]
                
                rejected_image = self.sum_dark_images[addr]/self.n_dark[addr]
                event_display_images.append( (name, title, rejected_image ) )
            
            #if self.dark_image is not None :
            #    name = "dark"
            #    title = "Dark image from input file"
            #    event_display_images.append( (name, title, self.dark_image ) )
            
            if len(event_display_images) == 0:
                print "That shouldn't be possible"
                return
            
            
            # plot to a new figure ... thus we must define new frames (if we want them to know about threshold)
            self.plotter.draw_figurelist(self.mpl_num+nsrc,
                                         event_display_images,
                                         title="Endjob:  %s"%addr,
                                         showProj=True)
        
            plt.draw()
        

        # -----------------------------------
        # Saving the final average to file(s)
        # -----------------------------------
        if (self.output_file is not None):

            self.save_images( self.output_file, event_display_images )


        # convert dict to a list:
        data_image = []
        for source in self.sources :
            data_image.append( self.data[source] )
            # give the list to the event object
            evt.put( data_image, 'data_image' )
            



    def save_images(self, filename, image_list, event=None ):

            #if self.hdf5file is not None :
            #    # save this event as a group in hdf5 file:
            #    group = self.hdf5file.create_group("Event%d" % self.n_shots)

            for name,title,array in image_list :

                print "save image ", name, title, array.shape

                parts = filename.split('.')
                label = name #address.replace("|","_").strip()
                
                thename = ''
                for i in range (len(parts)-1):
                    thename+="%s"%parts[i]
                    
                thename+="_%s"%label

                if event is not None:
                    thename+="_ev%d"%event

                thename+=".%s"%parts[-1]

                print "Writing to file ", thename

#                # HDF5
#                if self.hdf5file is not None :
#                    # save each image as a dataset in this event group
#                    dset = group.create_dataset("%s"%ad,data=array)

                # Numpy array
                if ".npy" in thename :
                    np.save(thename, array)
                elif ".txt" in thename :
                    np.savetxt(thename, array) 
                        
                else :
                    print "Output file does not have the expected file extension: ", fname[-1]
                    print "Expected hdf5, txt or npy. Please correct."
                    print "I'm not saving this event... "
