#------------------------------
"""User analysis module for pyana and psana frameworks.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------
#from psana import *
import psana
import numpy as np

class image_crop (object) :
    """Get image from the evt store crop it and put it back in the evt store"""

    def __init__ ( self ) :
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param source      string, address of Detector.Id:Device.ID
        @param key_in      string, keyword for input image 2-d array
        @param key_out     string, unique keyword for output image array
        @param rowmin      int,    row minimal to crop image    (dafault =  0 - for full size)
        @param rowmax      int,    row maximal to crop image    (dafault = -1 - for full size)
        @param colmin      int,    column minimal to crop image (dafault =  0 - for full size)
        @param colmax      int,    column maximal to crop image (dafault = -1 - for full size)
        @param print_bits  int, bit-word for verbosity control 
        """

        self.m_src        = self.configSrc  ('source', '*-*|Cspad-*')
        self.m_key_in     = self.configStr  ('key_in',    'image_in')
        self.m_key_out    = self.configStr  ('key_out',   'image_out')
        self.rowmin       = self.configInt  ('rowmin',     0)
        self.rowmax       = self.configInt  ('rowmax',    -1)
        self.colmin       = self.configInt  ('colmin',     0)
        self.colmax       = self.configInt  ('colmax',    -1)
        self.m_print_bits = self.configInt  ('print_bits', 1)

        self.counter = 0

        if self.m_print_bits & 1 : self.print_input_pars()

        self.list_of_dtypes = [
                               psana.ndarray_float32_2,
                               psana.ndarray_float64_2,
                               psana.ndarray_int8_2, 
                               psana.ndarray_int16_2, 
                               psana.ndarray_int32_2,
                               psana.ndarray_int64_2,
                               psana.ndarray_uint8_2, 
                               psana.ndarray_uint16_2, 
                               psana.ndarray_uint32_2,
                               psana.ndarray_uint64_2
                               ]        


    def beginjob( self, evt, env ) : pass
 
    def beginrun( self, evt, env ) : 
        self.run   = evt.run()
        self.exp   = env.experiment()
        self.evnum = 0

    def begincalibcycle( self, evt, env ) : pass

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        # Should work for both pyana and pytonic-psana (as compatability method):

        #print '\nimage_crop: evt.keys():', evt.keys()

        self.arr = None

        if env.fwkName() == "psana":
            #self.arr = evt.get(np.ndarray, self.m_key_in)
            #self.arr = evt.get(np.ndarray, self.m_src, self.m_key_in)
            for dtype in self.list_of_dtypes :
                self.arr = evt.get(dtype, self.m_src, self.m_key_in)
                if self.arr is not None:
                    break
            
        else : 
            self.arr = evt.get(self.m_key_in)

        self.counter +=1     

        if self.arr is None :
            #if self.m_print_bits & 32 :
            msg = __name__ + ': WARNING! CSPAD array object %s is not found in evt' % self.m_key_in
            #logging.info( msg )
            print msg
            return

        if self.m_print_bits & 2 and self.counter == 1 :
            self.print_image_parameters()

        self.img2d = np.array(self.arr[self.rowmin:self.rowmax, self.colmin:self.colmax])
        #self.img2d = self.arr

        #self.img2d.flags.writeable = False
        self.img2d.setflags(write=False)

        #evt.put( self.img2d, self.m_key_out ) # save image in event as 2d numpy array
        evt.put( self.img2d, self.m_src, self.m_key_out ) # save image in event as 2d numpy array

    def endcalibcycle( self, evt, env ) : pass

    def endrun       ( self, evt, env ) : pass

    def endjob       ( self, evt, env ) : pass


#-----------------------------

    def print_input_pars( self ) :
        msg = '\n%s: List of input parameters\n  source %s\n  key_in %s\n  key_out %s\n  print_bits: %4d' % \
              (__name__ , self.m_src, self.m_key_in, self.m_key_out, self.m_print_bits) + \
              '\n  rowmin %s\n  rowmax %s\n  colmin %s\n  colmax %s\n' % \
              (self.rowmin, self.rowmax, self.colmin, self.colmax)
        #logging.info( msg )
        print msg


    def print_image_parameters( self ) :
        msg = '%s: Input image parameters for run = %s:\n' % (__name__, self.run) \
            + '    shape = %s' % str(self.arr.shape) \
            + '    dtype = %s' % str(self.arr.dtype)
        #    + '\narray:\n' + str(self.arr)
        #logging.info( msg )
        print msg

#-----------------------------
#-----------------------------
