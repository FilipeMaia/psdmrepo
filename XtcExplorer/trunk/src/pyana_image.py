"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or
part of it, please give an appropriate acknowledgment.

@author Ingrid Ofte
"""
__version__ = "$Revision: 3190 $"

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import time
import logging

import numpy as np

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata.xtc import TypeId
from pyana.calib import CalibFileFinder

#-----------------------------
# Imports for local modules --
#-----------------------------
from cspad     import CsPad

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
                   quantities = "image dark average",
                   # plotting options:
                   plot_every_n = None,
                   accumulate_n = None,
                   plot_vrange = None,                   
                   # output options
                   outputfile = None,
                   output_format = None,
                   max_save = "0",
                   fignum = "1",
                   # data/calibration path (needed for CsPad)
                   calib_path = None,
                   small_tilt = False ):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param sources           address string of Detector-Id|Device-ID
        @param inputdark         name (base) of dark image file (numpy array) to be loaded, if any
        @param algorithms        (list of) algorithms to be applied to the image before plotting
        @param threshold         lower= upper= type= roi=
        @param quantities        string, list of quantities to plot: image average maximum projections

        @param plot_every_n      Frequency for plotting. If n=0, no plots till the end
        @param accumulate_n      Not implemented yet
        @param plot_vrange       value-range of interest
 
        @param outputfile        filename (If collecting: write to this file)
        @param max_save          Maximum single-shot images to save

        @param calib_path        path to the calibration directory (expNUMBER/calib)
        @param small_tilt        apply small tilt angles (using array interpolation). Default is False. 
        """

        opt = PyanaOptions() # convert option string to appropriate type
        self.plot_every_n  =  opt.getOptInteger(plot_every_n)
        self.accumulate_n  =  opt.getOptInteger(accumulate_n)
        self.max_save      =  opt.getOptInteger(max_save)
        self.mpl_num = opt.getOptInteger(fignum)

        self.sources = opt.getOptStrings(sources)
        nsources = len(self.sources)
        print "pyana_image, %d sources: " % nsources
        for sources in self.sources :
            print "  ", sources

        self.darkfile = opt.getOptString(inputdark)
        if self.darkfile is not None: print "Input dark image file: ", self.darkfile

        self.algorithms = opt.getOptStrings(algorithms)
        print "Algorithms to apply: ", self.algorithms

        self.quantities = opt.getOptStrings(quantities)
        print "Quantities to plot: ", self.quantities

        self.threshold = Threshold(threshold)
            
        self.output_file = opt.getOptString(outputfile)
        self.output_format = opt.getOptString(output_format) or 'int16'
        print "Output file name base: ", self.output_file

        self.plot_vmin = None
        self.plot_vmax = None
        if plot_vrange is not None and plot_vrange is not "" : 
            self.plot_vmin = float(plot_vrange.strip("()").split(":")[0])
            self.plot_vmax = float(plot_vrange.strip("()").split(":")[1])
            print "Using plot_vrange = %.2f,%.2f"%(self.plot_vmin,self.plot_vmax)

        # to keep track
        self.n_shots = None
        self.n_saved = 0
        self.run = None
        self.initlists()
        
        self.apply_dictionary = { 'rotate': alg.rotate,
                                  'shift' : alg.shift }
        

        self.calib_path = calib_path
        
        self.cspad = {}
        self.small_tilt = opt.getOptBoolean(small_tilt)

        self.configtypes = { 'Cspad2x2'  : TypeId.Type.Id_Cspad2x2Config ,
                             'Cspad'     : TypeId.Type.Id_CspadConfig ,
                             'Opal1000'  : TypeId.Type.Id_Opal1kConfig,
                             'Opal2000'  : TypeId.Type.Id_Opal1kConfig,
                             'Opal4000'  : TypeId.Type.Id_Opal1kConfig,
                             'TM6740'    : TypeId.Type.Id_TM6740Config,
                             'pnCCD'     : TypeId.Type.Id_pnCCDconfig,
                             'Princeton' : TypeId.Type.Id_PrincetonConfig,
                             'Fccd'      : TypeId.Type.Id_FccdConfig,
                             'PIM'       : TypeId.Type.Id_PimImageConfig,
                             'Timepix'   : TypeId.Type.Id_TimepixConfig,
                             'Fli'       : TypeId.Type.Id_FliConfig
                             }

        self.datatypes = {'Cspad2x2'      : TypeId.Type.Id_Cspad2x2Element, 
                          'Cspad'         : TypeId.Type.Id_CspadElement,
                          'TM6740'        : TypeId.Type.Id_Frame,
                          'Opal1000'      : TypeId.Type.Id_Frame,
                          'Opal2000'      : TypeId.Type.Id_Frame,
                          'Opal4000'      : TypeId.Type.Id_Frame,
                          'Fccd'          : TypeId.Type.Id_Frame,
                          'pnCCD'         : TypeId.Type.Id_pnCCDframe,
                          'Princeton'     : TypeId.Type.Id_PrincetonFrame,
                          'Timepix'       : TypeId.Type.Id_TimepixData,
                          'Fli'           : TypeId.Type.Id_FliFrame
                          }
    def initlists(self):
        self.accu_start = 0
        # averages
        self.max_good_images = {}
        self.sum_good_images = {}
        self.sum_dark_images = {}
        self.n_good = {}
        self.n_dark = {}
        for addr in self.sources :
            self.max_good_images[addr] = None
            self.sum_good_images[addr] = None
            self.sum_dark_images[addr] = None
            self.n_good[addr] = 0
            self.n_dark[addr] = 0

            

    def resetlists(self):
        self.accu_start = self.n_shots
        for addr in self.sources :
            del self.max_good_images[addr]
            del self.sum_good_images[addr]
            del self.sum_dark_images[addr] 
            self.max_good_images[addr] = None
            self.sum_good_images[addr] = None
            self.sum_dark_images[addr] = None
            self.n_good[addr] = 0
            self.n_dark[addr] = 0




    def beginjob ( self, evt, env ) : 
        logging.info( "pyana_image.beginjob()" )

        self.n_shots = 0
        self.n_accum = 0
        self.run = evt.run()

        self.data = {}
        for source in self.sources:
            self.data[source] = ImageData(source)

	## load dark image from file
        self.dark_image = None
	try:
            self.dark_image = np.load(self.darkfile)
	except:
            print "No dark image in file ", self.darkfile

        calibfinder = CalibFileFinder(env.calibDir(),"CsPad::CalibV1")
        for addr in self.sources :

            # pick out the device name from the address
            device = addr.split('|')[1].split('-')[0]
            self.config = env.getConfig( self.configtypes[device], addr )
            if not self.config:
                print '*** %s config object is missing ***'%addr
                return

            if addr.find('Cspad2x2')>=0:
                sections = self.config.sections()
                self.cspad[addr] = CsPad(sections, path=self.calib_path)
                
            elif addr.find('Cspad')>=0:
                quads = range(4)
                sections = map(self.config.sections, quads)
                self.cspad[addr] = CsPad(sections, path=self.calib_path)
                if self.small_tilt :
                    self.cspad[addr].small_angle_tilt = True

                try:
                    self.cspad[addr].load_pedestals( calibfinder.findCalibFile(addr,"pedestals",evt.run() ) )
                except OSError, e:
                    print "  ", e
                    print "  ", "No pedestals will be subtracted"
                    
    # process event/shot data
    def event ( self, evt, env ) :
        logging.debug( "pyana_image.event()" )

        # this one counts every event
        self.n_shots+=1

        # -------------------------------------------------
        # get the image(s) from the event datagram
        for addr in self.sources :

            image = None

            # pick out the device name from the address
            device = addr.split('|')[1].split('-')[0]
            frame = evt.get( self.datatypes[device], addr )
            if frame is None:
                print "No frame from ", addr
                return
            
            if addr.find("Cspad2x2")>0 :
                # in this case 'frame' is the MiniElement
                # call cspad library to assemble the image
                image = self.cspad[addr].get_mini_image(frame)

            elif addr.find("Cspad")>0 :
                # in this case we need the specialized getter: 
                quads = evt.getCsPadQuads(addr, env)
                # then call cspad library to assemble the image
                image = self.cspad[addr].get_detector_image(quads)

            elif addr.find("Fli")>0:
                image = frame.data(self.config)

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
            # Subtract dark image
            dark = self.dark_image
            if dark is not None :
            	image = image - dark
                             
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
            isSaturated = False
            name = addr
            title = addr


            # ---------------------------------------------------------------------------------------
            # threshold filter
            if not self.threshold.is_empty:

                # apply mask if requested
                if self.threshold.mask is not None:
                    #hot_mask = np.ma.masked_greater_equal( image, self.threshold.mask )
                    #image = np.ma.filled( hot_mask, 0 )

                    # this should be faster: zero out hot masks: 
                    image[image>self.threshold.mask]=0
                
                # first, assume we're considering the whole image
                roi = image
                if self.threshold.region is not None:
                    # or set it to the selected threshold region
                    roi = image[self.threshold.region[2]:self.threshold.region[3],   # rows (y-range)
                                self.threshold.region[0]:self.threshold.region[1]]   # columns (x-range)

                dims = np.shape(roi)
                maxbin = roi.argmax()
                maxvalue = roi.ravel()[maxbin]
                maxbin_coord = np.unravel_index(maxbin,dims)

                if self.threshold.type == 'average':
                    maxvalue = roi.mean()
                    #print "average value of the ROI = ", maxvalue
                    
                # apply the threshold
                if self.threshold.lower is not None and maxvalue < self.threshold.lower :
                    isDark = True
                    name += "_dark"
                    title += " (dark)"
                    
                if self.threshold.upper is not None and maxvalue > self.threshold.upper:
                    isSaturated = True
                    name += "_sat"
                    title += " (saturated)"


                VeryVerbose = False
                if VeryVerbose :
                    if (not isDark) and (not isSaturated): 
                        print "%d accepting %s #%d, vmax = %.0f, hitrate: %.4f" % \
                              (env.subprocess(), addr, self.n_shots, maxvalue, \
                               float(self.n_good[addr]+1)/float(self.n_shots))
                    else :
                        print "%d rejecting %s #%d, vmax = %.0f"%(env.subprocess(), \
                                                                  addr,self.n_shots, maxvalue)

            # ----------- Event Image -----------
            if "image" in self.quantities:
                self.data[addr].image   = image            
                self.data[addr].vrange = (self.plot_vmin,self.plot_vmax)
                
            if "projections" in self.quantities:
                self.data[addr].showProj = True
                
            # ------------- DARK ----------------
            if isDark:
                self.n_dark[addr]+=1

                if "darks" in self.quantities:
                    try:
                        self.sum_dark_images[addr] += image
                    except TypeError:
                        self.sum_dark_images[addr] = np.array(image,dtype=image.dtype)
                    # collect dark
                    self.data[addr].ndark = self.n_dark[addr]
                    self.data[addr].avgdark = np.float_(self.sum_dark_images[addr])/self.n_dark[addr]

            elif isSaturated: 
                pass 
            else :
                # ------------- HIT ----------------
                self.n_good[addr]+=1

                if "average" in self.quantities: 
                    try:
                        self.sum_good_images[addr] += image
                    except TypeError:
                        self.sum_good_images[addr] = np.array(image,dtype=image.dtype)

                if "maximum" in self.quantities:
                    try:
                        # take the element-wise maximum of array elements
                        # comparing this image with the previous maximum stored
                        self.max_good_images[addr] = np.maximum( image, self.max_good_images[addr] )
                    except TypeError: 
                        self.max_good_images[addr] = np.array(image,dtype=image.dtype)
        


        # make plot? if so, pass this info to the event. 
        if self.plot_every_n != 0 and (self.n_shots%self.plot_every_n)==0 :
            # update averages 
            self.update()
            
            # flag for pyana_plotter
            evt.put(True, 'show_event')
            
            # convert dict to a list:
            data_images = []
            for source in self.sources :
                data_images.append( self.data[source] )
                # give the list to the event object
                evt.put( data_images, 'data_images' )
                                                            

        # -----------------------------------
        # Saving this event to file(s)
        # -----------------------------------
        if (self.output_file is not None) and (self.n_saved < self.max_save) : 
            self.n_saved += 1

            # update averages
            self.update()

            images_for_saving = []
            for addr in self.sources:
                if 'image' in self.quantities:
                    images_for_saving.append( ('image', '%s ev %d'%(addr,self.n_shots), self.data[addr].image ))

                if 'roi' in self.quantities:
                    images_for_saving.append( ('roi', '%s ev %d'%(addr,self.n_shots), self.data[addr].roi ))

                if 'average' in self.quantities:
                    images_for_saving.append( ('average', '%s ev %d'%(addr,self.n_shots), self.data[addr].average ))
                if 'darks' in self.quantities:
                    images_for_saving.append( ('avgdark', '%s ev %d'%(addr,self.n_shots), self.data[addr].avgdark ))
                if 'maximum' in self.quantities:
                    images_for_saving.append( ('maximum', '%s ev %d'%(addr,self.n_shots), self.data[addr].maximum ))
                
            self.save_images( self.output_file, images_for_saving, self.n_shots )


        # --------- Reset -------------
        if self.accumulate_n!=0 and (self.n_shots%self.accumulate_n)==0 :
            self.resetlists()


    

    # after last event has been processed. 
    def endjob( self, evt, env ) :
        logging.info( "pyana_image.endjob()" )

        print "Done processing       ", self.n_shots, " events"

        # no more events to process. Just fetch the collected data and redraw
        if (env.subprocess()>0):
            return

        # flag for pyana_plotter
        evt.put(True, 'show_event')
            
        # convert dict to a list:
        data_image = []
        for source in self.sources :
            data_image.append( self.data[source] )
            # give the list to the event object
            evt.put( data_image, 'data_images' )
            

        if (self.output_file is not None):
            self.update()
            images_for_saving = []
            for addr in self.sources:
                print "Adding to list: ", addr, dir(self.data[addr])
                if 'image' in self.quantities:
                    images_for_saving.append( ('image', '%s ev %d'%(addr,self.n_shots), self.data[addr].image ))

                if 'roi' in self.quantities:
                    images_for_saving.append( ('roi', '%s ev %d'%(addr,self.n_shots), self.data[addr].roi ))

                if 'average' in self.quantities:
                    images_for_saving.append( ('average', '%s ev %d'%(addr,self.n_shots), self.data[addr].average ))
                if 'darks' in self.quantities:
                    images_for_saving.append( ('avgdark', '%s ev %d'%(addr,self.n_shots), self.data[addr].avgdark ))
                if 'maximum' in self.quantities:
                    images_for_saving.append( ('maximum', '%s ev %d'%(addr,self.n_shots), self.data[addr].maximum ))
                
            self.save_images( self.output_file, images_for_saving)

        return

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
                title = "Average of %d hits"%self.n_good[addr]
                print "adding", title
                average_image = np.float_(self.sum_good_images[addr])/self.n_good[addr]
                event_display_images.append( (name, title, average_image ) )

                name = "MaxHit_"+addr
                title = "Maximum projection of %d hits"%self.n_good[addr]
                print "adding", title
                event_display_images.append( (name,title,self.max_good_images[addr]) )

                # and max minus average
                name = "MaxOverAvg_"+addr
                title = "Max normalized to average "
                max_over_avg = np.float_(self.max_good_images[addr]) / average_image
                print "adding", title
                event_display_images.append( (name, title, max_over_avg) )

            rejected_image = None
            if self.n_dark[addr] > 0 :
                name = "AvgDark_"+addr
                title = "Average of %d darks"%self.n_dark[addr]
                
                rejected_image = np.float_(self.sum_dark_images[addr])/self.n_dark[addr]
                event_display_images.append( (name, title, rejected_image ) )
            
            if len(event_display_images) == 0:
                print "That shouldn't be possible"
                return
            
            

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
            evt.put( data_image, 'data_images' )
            



    # ---------------------------------------------------------------------------------------
    def update(self):
        # Update data for storage and plotting
        for addr in self.sources:
            # collect averages
            if 'average' in self.quantities:
                print "saving average from ", self.n_good[addr], " events. "
                self.data[addr].counter = self.n_good[addr]
                self.data[addr].average = np.float_(self.sum_good_images[addr])/self.n_good[addr]

            ## collect maximum
            if 'maximum' in self.quantities:
                self.data[addr].counter = self.n_good[addr]
                self.data[addr].maximum = self.max_good_images[addr]

            ## collect avg darks
            if 'darks' in self.quantities:
                self.data[addr].ndark = self.n_dark[addr]
                self.data[addr].avgdark = np.float_(self.sum_dark_images[addr])/self.n_dark[addr]
                            

    def save_images(self, filename, image_list, event=None ):

        for name,title,array in image_list :
            
            fname = filename.split('.')
            label = "%s_r%03d"%(name,self.run) #address.replace("|","_").strip()
            
            thename = ''
            for i in range (len(fname)-1):
                thename+="%s"%fname[i]
                
                thename+="_%s"%label

                if event is not None:
                    thename+="_ev%d"%event

                thename+=".%s"%fname[-1]

                print "Saving \"%s\" (%s) %s to file %s"% (name, title, array.shape, thename)

                array = array.astype(self.output_format)
                
                # output files... 
                if fname[-1] == "txt" :  # Ascii
                    np.savetxt(thename, array, fmt="%d")

                elif fname[-1] == "npy" : # Numpy binary 
                    np.save(thename, array)

                elif fname[-1] == 'dat' : # Raw binary
                    array.tofile(thename)
                elif fname[-1] == 'bin' : # Raw binary
                    array.tofile(thename)

                elif fname[-1] == "hdf5":
                    #print "HDF5 not implemented yet"
                    import h5py
                    file_handle = h5py.File(thename, 'w')
                    group = file_handle.create_group("Data")
                    dataset = group.create_dataset(title,data=array)
                    file_handle.close()                    

                elif fname[-1] == "mat":
                    import scipy.io
                    scipy.io.savemat(thename,{title:array})

                else :  # Some image format... JPG, tiff, etc... 
                    import scipy.misc 
                    scipy.misc.imsave(thename, array)
                    
