#--------------------------------------------------------------------------
# File and Version Information:
# cspad_hitfinder
# : Simple test of algorithm3 for CXI filtering 
# : veto blank shots by requiring a certain number of pixels
# : above some ADC count, e.g. 1000 pixels > 50 ADU
#------------------------------------------------------------------------
import sys, os, fnmatch, time
import logging
import numpy as np

from pyana import Skip

class cspad_hitfinder (object) :
    """Class whose instance will be used as a user analysis module. """

    def __init__ ( self,
                   source   = "CxiDs1-0|Cspad-0",
                   output = None, 
                   adc_thr = 50,
                   min_npix = 1000) :
        """
        @param source    Address of detector/device in xtc file.
        @param output    Name of output image for plotting, if any
        @param adc_thr   ADC threshold (count pixels above this threshold)
        @param min_npix  minumum number of pixels above threshold
        """
        self.source = source
        self.output = output
        self.adc_thr = int(adc_thr)
        self.min_npix = int(min_npix)

    def beginjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_hitfinder.beginjob() called" )

        self.starttime = time.time()
        self.n_proc = 0
        self.n_pass = 0
        self.npix_vetoed = []
                
    def beginrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_hitfinder.beginrun() called" )

        self.cspad = evt.get("CsPadAssembler:%s"%self.source)

    def begincalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_hitfinder.begincalibcycle() called" )

                      
    def event( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        self.n_proc+=1
        
        # get CsPad pixels 
        pixels = self.cspad.pixels

        # filter:
        pix_above = np.extract( pixels>self.adc_thr, pixels )
        n = pix_above.size  # number of pixels above
        i = pix_above.sum() # Total intensity above

        if n < self.min_npix :
            self.npix_vetoed.append(n)
            return Skip

        self.n_pass+=1

        # store the "results" in the cspad object in case they are needed later
        self.cspad.n_above = n
        self.cspad.i_above = i


    def endcalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_hitfinder.endcalibcycle() called" )

    def endrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_hitfinder.endrun() called" )

    def endjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_hitfinder.endjob() called" )

        duration = time.time() - self.starttime
        logging.info("cspad_hitfinder: Time elapsed: %.3f s"%duration)
        logging.info("cspad_hitfinder: %d shots selected out of %d processed"%(self.n_pass,self.n_proc))

        varray = np.float_(self.npix_vetoed)
        logging.info("cspad_hitfinder: rejected events had a median of %d pixels above threshold"%np.median(varray))
                     
