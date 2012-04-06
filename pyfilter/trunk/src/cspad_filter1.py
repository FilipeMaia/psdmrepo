#--------------------------------------------------------------------------
# File and Version Information:
# cspad_filter1
# : Simple test of algorithm1 for CXI filtering for crystals:
# : Require a certain number of peaks with some intensity
#------------------------------------------------------------------------
import sys, os, fnmatch, time
import logging
import numpy as np

from pypdsdata.xtc import TypeId
from cspad         import CsPadAssembler
from pyana.calib   import CalibFileFinder
from pyana import Skip

#import matplotlib.pyplot as plt

class cspad_filter1 (object) :
    """Class whose instance will be used as a user analysis module. """

    def __init__ ( self,
                   source   = "CxiDs1-0|Cspad-0",
                   do_pedestals = "1",
                   do_commonmode = "0",
                   roi = "0,1800,0,1800",
                   adc_thr = "50",
                   min_npix = "1000",
                   max_bkgframes = "10") :
        """
        @param source    Address of detector/device in xtc file.
        @param adc_thr   ADC threshold (count pixels above this threshold)
        @param min_npix  minumum number of pixels above threshold
        """
        self.source = source
        self.adc_thr = int(adc_thr)
        self.min_npix = int(min_npix)
        self.do_pedestals = bool(int(do_pedestals))
        self.do_commonmode = bool(int(do_commonmode))
        self.max_bkgframes = int(max_bkgframes)
        self.roi = map(int,roi.split(','))         
        self.n_std = 3 # n standard deviations (vertical scale region of interest)

    def beginjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter1.beginjob() called" )

        cdir = env.calibDir()
        # experiment calibration directory, specified or default
        
        cversion = fnmatch.filter( os.listdir(cdir), "CsPad::CalibV*")[-1]
        # CsPad calibration directory, specified or highest version number

        self.cfinder = CalibFileFinder(cdir,cversion)

        self.starttime = time.time()
        self.n_proc = 0
        self.n_pass = 0
        
        # images stored for computing running average (for each pixel)
        self.mu_stored = []
        self.img_stored = []

    def beginrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter1.beginrun() called" )

        config = env.getConfig(TypeId.Type.Id_CspadConfig,self.source)
        if not config:
            print '*** %s config object is missing ***'%self.source
            return
        
        quads = range(4)
        self.sections = map(config.sections, quads)
        
        self.cspad = CsPadAssembler(devicename=self.source,
                                    config=config,
                                    calibfinder=self.cfinder,
                                    run=evt.run())
        if self.do_pedestals:
            self.cspad.load_pedestals()
            


    def begincalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter1.begincalibcycle() called" )


                
    def event( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """

        # get CsPad elements from the datagram
        elements = evt.get(TypeId.Type.Id_CspadElement,self.source)
        self.n_proc+=1

        # make one large array (4x8x185x388) of pixels
        # (and fill in any blanks if needed)
        
        self.cspad.fill_pixel_array(elements)
        self.cspad.subtract_pedestals()

        # get the full image: 
        full_image = self.cspad.assemble_image(self.cspad.pixels)

        # get the region of interest
        x1,x2,y1,y2 = self.roi
        image = np.copy(full_image[x1:x2,y1:y2])
        
        # Filter 


        # collect for averaging of background
        self.img_stored.append(image)
        n = len(self.img_stored)
        if n >= self.max_bkgframes:
            del self.img_stored[0]
                

        # compute running average of previously collected background frames
        for img in self.img_stored:
            try:
                running_avg += img
            except:
                running_avg = img
        running_avg = np.float_(running_avg) / n

        # Filter (require peaks)  


        # Event has passed. 
        print "Plotting background based on ", n, " background frames"
        self.n_pass+=1
        
        # put the image
        evt.put(image,self.source)

    def endcalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_filter1.endcalibcycle() called" )

    def endrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter1.endrun() called" )

    def endjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_filter1.endjob() called" )

        duration = time.time() - self.starttime
        logging.info("cspad_filter1: Time elapsed: %.3f s"%duration)
        logging.info("cspad_filter1: %d shots selected out of %d processed"%(self.n_proc,self.n_pass))
