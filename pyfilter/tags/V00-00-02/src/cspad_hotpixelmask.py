#--------------------------------------------------------------------------
# File and Version Information:
# cspad_hotpixelmask
# : Simple test of algorithm1 for CXI filtering for crystals:
# : Require a certain number of peaks with some intensity
# : within a roi (no background subtaction)
#------------------------------------------------------------------------
import sys, os, fnmatch, time
import logging
import numpy as np
import scipy.ndimage 
import scipy.weave 

from pyana import Skip

class cspad_hotpixelmask (object) :
    """Class whose instance will be used as a user analysis module. """

    def __init__ ( self,
                   source = "CxiDs1-0|Cspad-0",
                   hot_adc_thr = "16000",
                   output = "hot pixels"):
        """
        @param source         Name of source
        @param hot_adc_thr    adc threshold to classify pixel as hot
        @param output         Name of output image
        """
        self.source = source
        self.hot_adc_thr = int(hot_adc_thr)
        self.output = output

    def beginjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_hotpixelmask.beginjob() called" )

        self.starttime = time.time()
        self.n_proc = 0
        self.n_pass = 0

        # compute hot pixels on the fly
        self.hot_pixels = None
        
    def beginrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_hotpixelmask.beginrun() called" )

        self.cspad = evt.get("CsPadAssembler:%s"%self.source)


    def begincalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_hotpixelmask.begincalibcycle() called" )
                
    def event( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """

        self.n_proc+=1
        
        hot_mask = (self.cspad.pixels>self.hot_adc_thr)
        self.cspad.pixels[hot_mask] = 0

        #hots = np.nonzero(hot_mask)
        #print "removed %d hot pixels", len(hots[0])
        
        # Event has passed. 
        self.n_pass+=1
        
    def endcalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_hotpixelmask.endcalibcycle() called" )

    def endrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_hotpixelmask.endrun() called" )

    def endjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_hotpixelmask.endjob() called" )

        duration = time.time() - self.starttime
        logging.info("cspad_hotpixelmask: Time elapsed: %.3f s"%duration)
        logging.info("cspad_hotpixelmask: %d shots selected out of %d processed"%(self.n_proc,self.n_pass))
