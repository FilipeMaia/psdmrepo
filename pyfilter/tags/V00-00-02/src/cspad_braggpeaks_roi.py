#--------------------------------------------------------------------------
# File and Version Information:
# cspad_braggpeaks_roi
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

# local libraries
def findpeak(image, a , b):
    """Adi Natan's peakfinder
    """
    image = image.astype(float)
    a = a.astype(int)
    b = b.astype(int)
    n = a.shape[0]
    picked = np.zeros(image.shape)
    
    code = """
    int x;
    int y ;
    
    for (int i=0; i<n; ++i) {
    x=a(i);
    y=b(i);
    if( (image(x,y)>image(x-1,y)) && (image(x,y)>image(x,y-1)) &&
        (image(x,y)>=image(x+1,y)) && (image(x,y)>=image(x,y+1)) &&
        (image(x,y)>image(x-1,y-1)) && (image(x,y)>image(x-1,y+1)) &&
        (image(x,y)>=image(x+1,y-1)) && (image(x,y)>=image(x+1,y+1))    ) {
    
        picked(x,y) = picked(x,y) + 1;
       }
    }
    
    return_val = 1;
    """
    scipy.weave.inline(code, ['image', 'n', 'a', 'b', 'picked'],
                       type_converters=scipy.weave.converters.blitz, compiler = 'gcc')
    
    return picked


#import matplotlib.pyplot as plt

class cspad_braggpeaks_roi (object) :
    """Class whose instance will be used as a user analysis module. """

    def __init__ ( self,
                   source = "CxiDs1-0|Cspad-0",
                   input = "Something", 
                   output = "SomethingElse",
                   roi = "0,1800,0,1800" ):
        """
        @param source    Name of source
        @param input     Name of input image (stored by another module)
        @param output    Name of output image
        @param roi       Region of interest (x1,x2,y1,y2)
        """
        self.source = source
        self.input = input
        self.output = output
        self.roi = map(int,roi.split(','))         
        self.n_std = 3 # n standard deviations (vertical scale region of interest)

    def beginjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_braggpeaks_roi.beginjob() called" )

        self.starttime = time.time()
        self.n_proc = 0
        self.n_pass = 0
        self.cent = []
        
    def beginrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_braggpeaks_roi.beginrun() called" )

        self.cspad = evt.get("CsPadAssembler:%s"%self.source)
        logging.info( "cspad_braggpeaks_roi CsPadAssembler??: %s"%self.cspad)

    def begincalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_braggpeaks_roi.begincalibcycle() called" )
                
    def event( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """

        self.n_proc+=1
        #self.cspad.subtract_commonmode(threshold=30)        
        full_image = self.cspad.assemble_image()

        # get the region of interest
        x1,x2,y1,y2 = self.roi
        roi = np.copy(full_image[x1:x2,y1:y2])

        # Filter (require peaks)  
        thr1 = np.max([min(roi.max(0)), min(roi.max(1))])
        roi = roi * (roi > thr1)
        roi = scipy.ndimage.median_filter(roi, (3,3))
        roi = scipy.ndimage.gaussian_filter(roi,1.0)
        roi = roi * (roi > 0.85*thr1)
                            
        edge = 20
        a,b = np.nonzero(roi[edge:roi.shape[0]-edge,edge:roi.shape[1]-edge])
        a = a + edge
        b = b + edge

        m = findpeak(roi,a,b)
        # a mask

        #full_image[x1:x2,y1:y2] = full_image[x1:x2,y1:y2] * m
        
        #neighborhood = scipy.ndimage.morphology.generate_binary_structure(2,2)
        #local_max = scipy.ndimage.filters.maximum_filter(mimage, footprint=neighborhood)==mimage
        #background = (mimage==0)
        #eroded_background = scipy.ndimage.morphology.binary_erosion(background,
        #                                                            structure=neighborhood,
        #                                                            border_value=1)
        #detected_peaks = local_max - eroded_background
        #(xcoord,ycoord) = np.nonzero(detected_peaks)
        #print "Found %d peaks! "% len(xcoord)

        # Event has passed. 
        self.n_pass+=1
        
        # put the image
        evt.put(full_image, self.output)

    def endcalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_braggpeaks_roi.endcalibcycle() called" )

    def endrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_braggpeaks_roi.endrun() called" )

    def endjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_braggpeaks_roi.endjob() called" )

        duration = time.time() - self.starttime
        logging.info("cspad_braggpeaks_roi: Time elapsed: %.3f s"%duration)
        logging.info("cspad_braggpeaks_roi: %d shots selected out of %d processed"%(self.n_proc,self.n_pass))
