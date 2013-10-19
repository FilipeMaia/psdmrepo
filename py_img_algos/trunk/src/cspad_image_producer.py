#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module cspad_image_producer...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: template!pyana-module!py 2987 2012-02-25 03:28:58Z salnikov@SLAC.STANFORD.EDU $

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 2987 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------
#from pypdsdata import xtc

import PyCSPadImage.CalibPars       as calp
import PyCSPadImage.CSPadConfigPars as ccp
import PyCSPadImage.CSPADPixCoords  as pixcoor


from pypdsdata.xtc import *
from psana import *
import numpy as np

class cspad_image_producer (object) :
    """Produces cspad image from input array of shape (4, 8, 185, 388)"""

    def __init__ ( self ) :
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param source      string, address of Detector-Id|Device-ID
        @param dtype_str   string, output array data type
        @param key_in      string, keyword for input array, shape=(4, 8, 185, 388)  
        @param key_out     string, unique keyword for output image array
        @param print_bits  int, bit-word for verbosity control 
        """

        #self.m_src        = self.configSrc  ('source', '*-*|Cspad-*')
        #self.m_dtype_str  = self.configStr  ('data_type', 'int')
        self.m_calib_dir  = self.configStr  ('calib_dir', '')
        self.m_key_in     = self.configStr  ('key_in',    'cspad_array')
        self.m_key_out    = self.configStr  ('key_out',   'cspad_image')
        self.m_print_bits = self.configInt  ('print_bits', 1)

        if self.m_print_bits & 1 : self.print_input_pars()


    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job. It should
        do a one-time initialization possible extracting values from event
        data (which is a Configure object) or environment.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_image_producer.beginjob() called" )

 
    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_image_producer.beginrun() called" )

        self.run    = evt.run()
        self.calib  = calp.CalibPars(self.m_calib_dir, self.run)
        self.coord  = pixcoor.CSPADPixCoords(self.calib)
        self.config = ccp.CSPadConfigPars()

        if self.m_print_bits & 2 : self.print_calibration_parameters()
        if self.m_print_bits & 4 : self.print_configuration_parameters()


    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_image_producer.begincalibcycle() called" )


    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        self.arr = None

        if env.fwkName() == "psana":
            self.arr = evt.get(CsPad.Data, self.m_src).quads_list()
            #quads = evt.get("Psana::CsPad::Data", self.m_src).quads_list()
        else:
            self.arr = evt.get(self.m_key_in)
        if self.arr is None :
            return

        if self.m_print_bits & 8 : self.print_part_of_cspad_array()

        self.arr.shape = (32, 185, 388)
        self.img2d = self.coord.get_cspad_image(self.arr, self.config)
        if self.m_print_bits & 16 : print 'Output image shape =', self.img2d.shape

        evt.put( self.img2d, self.m_key_out ) # save image in event as 2d numpy array


    def endcalibcycle( self, evt, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_image_producer.endcalibcycle() called" )


    def endrun( self, evt, env ) :
        """This optional method is called if present at the end of the run.
        
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_image_producer.endrun() called" )


    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_image_producer.endjob() called" )


    def print_input_pars( self ) :
        msg = '\nList of input parameters\n  calib_dir %s\n  key_in %s\n  key_out %s\n  print_bits: %4d' % \
              (self.m_calib_dir, self.m_key_in, self.m_key_out, self.m_print_bits)
        #logging.info( msg )
        print msg


    def print_part_of_cspad_array( self ) :
        print 'arr[2,4,:] =', self.arr[2,4,:]
        print 'arr.shape =', self.arr.shape


    def print_calibration_parameters( self ) :
        print '\nCalibration parameters for run =', self.run
        self.coord.print_cspad_geometry_pars()

    def print_configuration_parameters( self ) :
        print '\nCalibration parameters for run =', self.run
        self.config.printCSPadConfigPars()

#-----------------------------
#-----------------------------
#-----------------------------
