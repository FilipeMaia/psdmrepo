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
import os
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------
#from pypdsdata import xtc
#from pypdsdata.xtc import *
#from psana import *

import numpy as np
import psana

import PyCSPadImage.CalibPars         as calp
import PyCSPadImage.CSPadConfigPars   as ccp
import PyCSPadImage.CSPADPixCoords    as pixcoor

import PyCSPadImage.CSPAD2x2CalibPars as calp2x2
import PyCSPadImage.CSPAD2x2PixCoords as pixcoor2x2

#-----------------------------

class cspad_image_producer (object) :
    """Produces cspad image from input array of shape (4, 8, 185, 388)"""

#-----------------------------

    def __init__ ( self ) :
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param calib_dir   string, path to calibration directory for ex.: /reg/d/psdm/mec/meca6113/calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/
        @param source      string, address of Detector.Id:Device.ID
        @param key_in      string, keyword for input array, shape=(4, 8, 185, 388) - for cspad or (185, 388, 2) - for cspad2x2
        @param key_out     string, unique keyword for output image array
        @param print_bits  int, bit-word for verbosity control 
        """

        self.m_src        = self.configSrc  ('source', '*-*|Cspad-*')
        self.m_path_ctypes= self.configStr  ('calib_dir', '')
        self.m_key_in     = self.configStr  ('key_in',    'cspad_array')
        self.m_key_out    = self.configStr  ('key_out',   'cspad_image')
        self.m_print_bits = self.configInt  ('print_bits', 1)

        self.is_configed = False
        self.counter = 0

        self.is_cspad    = False
        self.is_cspad2x2 = False

        self.list_of_dtypes = [
                               psana.ndarray_int16_3, 
                               psana.ndarray_int32_3,
                               psana.ndarray_float32_3,
                               psana.ndarray_float64_3,
                               psana.ndarray_int8_3, 
                               psana.ndarray_int64_3, 
                               psana.ndarray_int16_4, 
                               psana.ndarray_int32_4,
                               psana.ndarray_float32_4,
                               psana.ndarray_float64_4,
                               psana.ndarray_int8_4, 
                               psana.ndarray_int64_4
                               ]        

        if self.m_print_bits & 1 : self.print_input_pars()


#-----------------------------

    def beginjob( self, evt, env ) : pass
 
    def beginrun( self, evt, env ) : pass

    def begincalibcycle( self, evt, env ) : pass

#-----------------------------

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        # Should work for both pyana and pytonic-psana (as compatability method):

        #print '\ncspad_image_producer: evt.keys():', evt.keys()

        self.counter +=1     


        self.arr = None

        if env.fwkName() == "psana":

            for dtype in self.list_of_dtypes :
                self.arr = evt.get(dtype, self.m_src, self.m_key_in)
                if self.arr is not None:
                    break
            
        else : 
            self.arr = evt.get(self.m_key_in)


        if self.needs_in_config() :
            self.set_configuration( evt, env )

        if self.arr is None :
            #if self.m_print_bits & 32 :
            msg = __name__ + ': WARNING! CSPAD array object %s is not found in evt' % self.m_key_in
            #logging.info( msg )
            print msg
            return

        if self.m_print_bits & 8 : self.print_part_of_cspad_array()

        self.img2d = None

        if self.is_cspad :
            self.arr.shape = (32, 185, 388)
            self.img2d = self.coord.get_cspad_image(self.arr, self.config)

        elif self.is_cspad2x2 :
            self.arr.shape = (185, 388, 2)
            self.img2d = self.coord.get_cspad2x2_image(self.arr)

        else :
            if self.m_print_bits & 32 : 
                msg = __name__ + ': WARNING! Image can be reconstructed for FULL SIZE CSPAD(2x2) NDARRAYS with shape (32, 185, 388) or (185, 388, 2)'
                print msg
            

        if self.img2d is not None :

            if self.m_print_bits & 16 :
                msg = __name__ + ': output image shape = ' + str(self.img2d.shape)
                print msg

            evt.put( self.img2d, self.m_src, self.m_key_out ) # save image in event as 2d numpy array


#-----------------------------

    def endcalibcycle( self, evt, env ) : pass

    def endrun       ( self, evt, env ) : pass

    def endjob       ( self, evt, env ) : pass

#-----------------------------

    def needs_in_config( self ) :
        return False if self.is_configed else True

#-----------------------------

    def set_configuration( self, evt, env ) :

        self.run    = evt.run()
        if self.arr is None : return

        if self.arr.shape[0] == 185 : # for (185, 388, 2)
            self.is_cspad    = False
            self.is_cspad2x2 = True
            self.group = 'CsPad2x2::CalibV1'
            
            self.set_path_to_calib_types(env)

            #self.calib  = calp2x2.CSPAD2x2CalibPars()
            self.calib  = calp2x2.CSPAD2x2CalibPars(self.m_path_ctypes, self.run)
            self.coord  = pixcoor2x2.CSPAD2x2PixCoords(self.calib)
            self.config = None 

            msg = __name__ + ': Set configuration for CSPAD2x2'
            self.is_configed = True

        # for (4, 8, 185, 388) or for (32, 185, 388)
        elif self.arr.shape[0] == 4 or self.arr.shape[0] == 32 :
            self.is_cspad    = True
            self.is_cspad2x2 = False
            self.group = 'CsPad::CalibV1'

            self.set_path_to_calib_types(env)
    
            self.calib  = calp.CalibPars(self.m_path_ctypes, self.run)
            self.coord  = pixcoor.CSPADPixCoords(self.calib)
            self.config = ccp.CSPadConfigPars()
            
            msg = __name__ + ': Set configuration for CSPAD'
            self.is_configed = True

        else :
            msg = __name__ + ': WARNING: Array for CSPAD of CSPAD2x2 is not defined.'

        print msg

        if self.m_print_bits & 2 : self.print_calibration_parameters()
        if self.m_print_bits & 4 : self.print_configuration_parameters()

#-----------------------------

    def set_path_to_calib_types( self, env ) :
        """Sets self.m_path_ctypes - path to calibration types, if it does not defined in
           user parameter 'calib_dir', for example:
           '/reg/d/psdm/mec/meca6113/calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/'
        """
        # str(self.m_src) = Source("DetInfo(XcsEndstation.0:Cspad.0)")
        # str_src = 'DetInfo(XcsEndstation.0:Cspad.0'    
        str_src = str(self.m_src).split('DetInfo(')[1].split(')"')[0]

        cdir = env.calibDir()

        msg = '[%s] Variables for path to calib types:\n  : user defined m_path_ctypes = "%s"\n  : %s\n  : %s\n  : %s\n' %\
              (__name__, self.m_path_ctypes, cdir, self.group, str_src)

        if  self.m_path_ctypes == '' :
            self.m_path_ctypes = os.path.join(cdir, self.group, str_src)
            msg += '  : Use default calib path: %s' % (self.m_path_ctypes)
        else :
            msg += '  : Use user-defined calib path: %s' % (self.m_path_ctypes)

        if self.m_print_bits & 64 : print msg

#-----------------------------

    def print_input_pars( self ) :
        msg = '\n[%s] List of input parameters\n  path %s\n  source %s\n  key_in %s\n  key_out %s\n  print_bits: %4d' %\
              (__name__, self.m_path_ctypes, self.m_src, self.m_key_in, self.m_key_out, self.m_print_bits)
        #logging.info( msg )
        print msg


    def print_part_of_cspad_array( self ) :
        msg = __name__ + ': arr[2,4,:] :\n' + str(self.arr[2,4,:]) \
            + '\n arr.shape =' + str(self.arr.shape)
        #logging.info( msg )
        print msg


    def print_calibration_parameters( self ) :
        msg = '%s: Calibration parameters for run = %d' % (__name__, self.run)
        #logging.info( msg )
        print msg
        if   self.is_cspad    : self.coord.print_cspad_geometry_pars()
        elif self.is_cspad2x2 : self.calib.printCalibPars()


    def print_configuration_parameters( self ) :
        msg = '%s: Configuration parameters for run = %s' % (__name__, self.run)
        #logging.info( msg )
        print msg
        if   self.is_cspad    : self.config.printCSPadConfigPars()
        elif self.is_cspad2x2 : msg = '%s: for cspad2x2 configuration is not required.' % (__name__)

#-----------------------------
#-----------------------------
