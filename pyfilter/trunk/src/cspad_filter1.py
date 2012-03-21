#--------------------------------------------------------------------------
# File and Version Information:
# cspad_filter1
# : Simple test of algorithm1 for CXI filtering for crystals:
# : Require a certain number of peaks with some intensity
#------------------------------------------------------------------------
import sys, os, fnmatch
import logging
import numpy as np

from pypdsdata.xtc import TypeId
from cspad         import CsPadAssembler
from pyana.calib   import CalibFileFinder

class cspad_filter1 (object) :
    """Class whose instance will be used as a user analysis module. """

    def __init__ ( self,
                   source   = "CxiDs1-0|Cspad-0",
                   adc_thr = 50,
                   min_npix = 1000) :
        """
        @param source    Address of detector/device in xtc file.
        @param adc_thr   ADC threshold (count pixels above this threshold)
        @param min_npix  minumum number of pixels above threshold
        """
        self.source = source
        self.adc_thr = int(adc_thr)
        self.min_npix = int(min_npix)

    def beginjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter3.beginjob() called" )

        cdir = env.calibDir()
        # experiment calibration directory, specified or default
        
        cversion = fnmatch.filter( os.listdir(cdir), "CsPad::CalibV*")[-1]
        # CsPad calibration directory, specified or highest version number

        self.cfinder = CalibFileFinder(cdir,cversion)

    def beginrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter3.beginrun() called" )


    def begincalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter3.begincalibcycle() called" )

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
                
    def event( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """

        # get CsPad elements from the datagram
        elements = evt.get(TypeId.Type.Id_CspadElement,self.source)
        
        # make one large array (4x8x185x388) of pixels
        # (and fill in any blanks if needed)
        pixels = self.cspad.get_pixel_array(elements)


        # filter:
        n = np.extract( pixels>self.adc_thr, pixels ).size

        if n < self.min_npix :
            return pyana.Skip

        image = self.cspad.assemble_image()

        # put the image
        evt.put(image,self.source)

    def endcalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_filter3.endcalibcycle() called" )

    def endrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter3.endrun() called" )

    def endjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_filter3.endjob() called" )
