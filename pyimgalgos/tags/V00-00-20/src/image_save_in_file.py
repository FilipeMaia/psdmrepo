#------------------------------
"""User analysis module 

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
import os
import sys
import logging
import tempfile

#-----------------------------
# Imports for other modules --
#-----------------------------
#from pypdsdata import xtc

#from pypdsdata.xtc import *
#from psana import *

import psana

import numpy as np
import scipy.misc as scim
from time import sleep
#import tifffile as tiff
#from PythonMagick import Image
#from PIL import Image
import Image
from CorAna.ArrFileExchange import *

from cspad_arr_producer import *

class image_save_in_file (object) :
    """Saves image array in file with specified in the name type."""

    def __init__ ( self ) :
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.

        @param source      string, address of Detector.Id:Device.ID
        @param key_in      string, keyword for input image array of variable shape
        @param ofname      string, output file name (type is selected by extention) supported formats: txt, tiff, gif, pdf, eps, png, jpg, jpeg, npy (default), npz
        @param mode        int, 0-save one event per event, >0-length of the ring buffer (or round robin) for event browser
        @param delay_sec   int, additional sleep time in sec between events for event browser
        @param print_bits  int, bit-word for verbosity control 
        """

        self.m_src        = self.configSrc  ('source', '*-*|Cspad-*')
        self.m_key_in     = self.configStr  ('key_in',    'image')
        self.m_ofname     = self.configStr  ('ofname',    './roi-img')
        self.m_mode       = self.configInt  ('mode',       0)
        self.m_delay_sec  = self.configInt  ('delay_sec',  0)
        self.m_print_bits = self.configInt  ('print_bits', 1)

        if self.m_print_bits & 1 : self.print_input_pars()

        if self.m_mode > 0 :
            pbits = 0377 if self.m_print_bits & 16 else 0
            self.afe = ArrFileExchange(self.m_ofname, self.m_mode, pbits)

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
        #logging.info( "image_save_in_file.beginrun() called" )

        self.run   = evt.run()
        self.exp   = env.experiment()
        self.evnum = 0


    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        self.image = None
        
        if env.fwkName() == "psana":
            #self.image = evt.get(psana.ndarray_float32_2, self.m_src, self.m_key_in)

            for dtype in self.list_of_dtypes :
                self.image = evt.get(dtype, self.m_src, self.m_key_in)
                if self.image is not None:
                    break

        else : 
            self.image = evt.get(self.m_key_in)       

        if self.image is None :
            #if self.m_print_bits & 32 :
            msg = '%s: WARNING! CSPAD image np.ndarray %s is not found in evt' % ( __name__, self.m_key_in )
            #logging.info( msg )
            print msg
            return

        #self.image = evt.get(self.m_key_in)

        if self.image is None :
            return

        self.evnum += 1

        if self.m_print_bits & 8 : self.print_part_of_image_array()

        if self.m_mode > 0 :
            self.afe.save_arr(self.image)
            sleep(self.m_delay_sec)
            return


        name_pref, name_ext = os.path.splitext(self.m_ofname)
        fname = '%s-%s-r%04d-ev%06d%s' % (name_pref, self.exp, self.run, self.evnum, name_ext)
        if self.m_print_bits & 4 :
            msg = 'Save image in file %s' % fname
            print msg

        if name_ext == '.txt' :
            np.savetxt(fname, self.image) # , fmt='%f')

        elif name_ext in ['.tiff'] :
            """Saves 16-bit tiff
            """

            tmp_file = tempfile.NamedTemporaryFile(mode='r+b',suffix='.tiff')
            tfile = tmp_file.name

            #img = Image.fromarray(self.image.astype(np.int16),'I;16B')
            img = Image.fromarray(self.image.astype(np.int16))
            img.save(tfile)

            cmd = 'convert %s -define quantum:format=signed %s' % (tfile, fname) 
            os.system(cmd)
 
        elif name_ext in ['.tiff', '.gif', '.pdf', '.eps', '.png', '.jpg', '.jpeg'] : 
            """Saves 8-bit tiff only...
            """
            scim.imsave(fname, self.image) 
 
        elif name_ext == '.npz' : 
            np.savez(fname, self.image)

        else : 
            np.save(fname, self.image)
 

    def endjob( self, evt, env ) : pass



    def print_input_pars( self ) :
        msg = '\n%s: List of input parameters\n  source %s\n  key_in %s\n  ofname %s\n  print_bits: %4d' % \
              ( __name__, self.m_src, self.m_key_in, self.m_ofname, self.m_print_bits)
        #logging.info( msg )
        print msg


    def print_part_of_image_array( self, r1=50, r2=60, c1=100, c2=110 ) :
        msg =  '%s: Part of the image: image[%d:%d,%d:%d]:' % (__name__, r1, r2, c1, c2)
        msg += '\n%s' % str(self.image[r1:r2,c1:c2])
        msg += '\n image.shape = %s' % str(self.image.shape)
        #logging.info( msg )
        print msg


#-----------------------------
#-----------------------------
#-----------------------------
