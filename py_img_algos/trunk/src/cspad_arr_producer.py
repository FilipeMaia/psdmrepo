#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module cspad_arr_producer...
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

from pypdsdata.xtc import *
from psana import *
import numpy as np

class cspad_arr_producer (object) :
    """Produces from CSPAD data numpy array of shape=(4, 8, 185, 388) and specified data type."""

    def __init__ ( self ) :
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param source      string, address of Detector-Id|Device-ID
        @param dtype_str   string, output array data type
        @param key_out     string, unique keyword for output array
        @param val_miss    float,  intensity value substituted for missing in data 2x1s
        @param print_bits  int, bit-word for verbosity control 
        """

        self.m_src        = self.configSrc  ('source', '*-*|Cspad-*')
        self.m_dtype_str  = self.configStr  ('data_type', 'int')
        self.m_key_out    = self.configStr  ('key_out',   'cspad_array')
        self.m_val_miss   = self.configFloat('val_miss',   0)
        self.m_print_bits = self.configInt  ('print_bits', 1)

        self.set_dtype()

        if self.m_print_bits & 1 : self.print_input_pars()


    def set_dtype( self ) :
        if   self.m_dtype_str == 'int'    : self.m_dtype = np.int
        elif self.m_dtype_str == 'int8'   : self.m_dtype = np.int8
        elif self.m_dtype_str == 'int16'  : self.m_dtype = np.int16
        elif self.m_dtype_str == 'int32'  : self.m_dtype = np.int32
        elif self.m_dtype_str == 'uint8'  : self.m_dtype = np.uint8
        elif self.m_dtype_str == 'uint16' : self.m_dtype = np.uint16
        elif self.m_dtype_str == 'uint32' : self.m_dtype = np.uint32
        elif self.m_dtype_str == 'float'  : self.m_dtype = np.float
        elif self.m_dtype_str == 'double' : self.m_dtype = np.double
        else                              : self.m_dtype = np.int16


    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job. It should
        do a one-time initialization possible extracting values from event
        data (which is a Configure object) or environment.

        @param evt    event data object
        @param env    environment object
        """

        # Preferred way to log information is via logging package
        logging.info( "cspad_arr_producer.beginjob() called" )

        config = env.getConfig(TypeId.Type.Id_CspadConfig, self.m_src)
        if not config:
            return

        print "cspad_arr_producer: %s: %s" % (config.__class__.__name__, self.m_src)
        print "  numQuads =",     config.numQuads();
        print "  asicMask =",     config.asicMask();
        print "  quadMask =",     config.quadMask();
        print "  numAsicsRead =", config.numAsicsRead();

        try:
            # older versions may not have all methods
            print "  roiMask       : [%s]" % ', '.join([hex(config.roiMask(q)) for q in range(4)])
            print "  numAsicsStored: %s" % str(map(config.numAsicsStored, range(4)))
        except:
            pass
        if env.fwkName() == "pyana":
            self.list_of_sections = map(config.sections, range(4)) 
            print "  sections      : %s" % str(self.list_of_sections)

 
    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_arr_producer.beginrun() called" )


    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_arr_producer.begincalibcycle() called" )


    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        if env.fwkName() == "psana":
            data = evt.get(CsPad.Data, self.m_src).quads_list()
            #quads = evt.get("Psana::CsPad::Data", self.m_src).quads_list()
        else:
            data = evt.get(TypeId.Type.Id_CspadElement, self.m_src)
        if not data :
            return


        #nQuads = data.quads_shape()[0]
        if self.m_val_miss == 0 : self.arr = np.zeros((4, 8, 185, 388), dtype=self.m_dtype)
        else                    : self.arr = np.ones ((4, 8, 185, 388), dtype=self.m_dtype) * self.m_dtype(self.m_val_miss)

        for i, q in enumerate(data): # where quad is an object of pypdsdata.cspad.ElementV2
            #print "  Quadrant #%d" % q.quad()
            #print "    data shape = {}".format(q.data().shape)
            #print "    data = %s" % q.data()
            self.arr[i,:] = q.data()

        if self.m_print_bits & 8 : self.print_part_of_output_array()

        evt.put( self.arr, self.m_key_out )


    def endcalibcycle( self, evt, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_arr_producer.endcalibcycle() called" )


    def endrun( self, evt, env ) :
        """This optional method is called if present at the end of the run.
        
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_arr_producer.endrun() called" )


    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_arr_producer.endjob() called" )


    def print_input_pars( self ) :
        msg = '\nList of input parameters\n  source: %s\n  print_bits: %4d\n  dtype_str: %s\n  dtype_str: %s\n  key_out %s\n  m_val_miss: %s' % \
              (self.m_src, self.m_print_bits, self.m_dtype_str, str(self.m_dtype), self.m_key_out, self.m_val_miss)
        #logging.info( msg )
        print msg


    def print_part_of_output_array( self ) :
        print 'arr[2,4,:] =', self.arr[2,4,:]
        print 'arr.shape =', self.arr.shape

#-----------------------------
#-----------------------------
#-----------------------------
