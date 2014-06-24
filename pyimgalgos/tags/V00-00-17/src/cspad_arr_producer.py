#------------------------------
"""User analysis module for pyana framework.

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
#from pypdsdata import xtc

from pypdsdata.xtc import *
from psana import *
import numpy as np

class cspad_arr_producer (object) :
    """Produces from CSPAD data numpy array of shape=(4, 8, 185, 388) or (185, 388, 2) and specified data type."""

    dic_dtypes = { 'int'    : np.int, \
                   'int8'   : np.int8, \
                   'int16'  : np.int16, \
                   'int32'  : np.int32, \
                   'uint'   : np.uint, \
                   'uint8'  : np.uint8, \
                   'uint16' : np.uint16, \
                   'uint32' : np.uint32, \
                   'float'  : np.float, \
                   'float32': np.float32, \
                   'double' : np.double \
                  }
   

    def __init__ ( self ) :
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param source      string, address of Detector-Id|Device-ID
        @param dtype_type  string, output array data type
        @param key_out     string, unique keyword for output array identification
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
        if self.m_print_bits & 1 : self.print_dtypes()

        self.is_cspad     = False
        self.is_cspad2x2  = False


    def set_dtype( self ) :
        try    :
            self.m_dtype = self.dic_dtypes[self.m_dtype_str]
        except :
            msg = __name__ + ' WARNING: specified data_type = %s is not implemented' % self.m_dtype_str
            print msg
            self.print_dtypes()    
            self.m_dtype = np.int16 

        #if   self.m_dtype_str == 'int'    : self.m_dtype = np.int
        #elif self.m_dtype_str == 'int8'   : self.m_dtype = np.int8
        #elif self.m_dtype_str == 'int16'  : self.m_dtype = np.int16
        #elif self.m_dtype_str == 'int32'  : self.m_dtype = np.int32
        #elif self.m_dtype_str == 'uint8'  : self.m_dtype = np.uint8
        #elif self.m_dtype_str == 'uint16' : self.m_dtype = np.uint16
        #elif self.m_dtype_str == 'uint32' : self.m_dtype = np.uint32
        #elif self.m_dtype_str == 'float'  : self.m_dtype = np.float
        #elif self.m_dtype_str == 'double' : self.m_dtype = np.double
        #else                              : self.m_dtype = np.int16


    def print_dtypes( self ) :
        msg = '\nImplemented data types:'
        for k,v in cspad_arr_producer.dic_dtypes.iteritems() :
            msg += '\n%10s : %10s' % (k,v)
        print msg
        

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job. It should
        do a one-time initialization possible extracting values from event
        data (which is a Configure object) or environment.

        @param evt    event data object
        @param env    environment object
        """

        if env.fwkName() == "psana": self.config = env.getConfig(CsPad.Config, self.m_src)
        else                       : self.config = env.getConfig(TypeId.Type.Id_CspadConfig, self.m_src)
        if self.config :
            self.is_cspad    = True
            self.is_cspad2x2 = False
            if self.m_print_bits & 2 : self.print_config_pars_for_cspad(env)
            return

        if env.fwkName() == "psana": self.config = env.getConfig(CsPad2x2.Config, self.m_src)
        else                       : self.config = env.getConfig(TypeId.Type.Id_Cspad2x2Config, self.m_src)
        if self.config :
            self.is_cspad    = False
            self.is_cspad2x2 = True
            if self.m_print_bits & 2 : self.print_config_pars_for_cspad2x2(env)
            return

        msg = __name__ + ' WARNING: CSPAD or CSPAD2x2 configuration is NOT found!'
        print msg


    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """
        #logging.info( "cspad_arr_producer.beginrun() called" )
        pass


    def begincalibcycle( self, evt, env ) : pass


    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        self.arr = None

        if   self.is_cspad    : self.proc_event_for_cspad    (evt, env)
        elif self.is_cspad2x2 : self.proc_event_for_cspad2x2 (evt, env)            


        if self.arr is None :
            msg =  __name__ + 'WARNING!: image is non-available' 
            print msg
            return

        if self.m_print_bits & 8 : self.print_part_of_output_array()

        #print 'self.arr.dtype =', self.arr.dtype
        evt.put( self.arr, self.m_src, self.m_key_out )

        #print '\ncspad_arr_producer: evt.keys():', evt.keys()

        
    def proc_event_for_cspad( self, evt, env ) :

        # 1. Get data
        if env.fwkName() == "psana":
            data_tot = evt.get(CsPad.Data, self.m_src) # returns psana.CsPad.DataV2 object
            if not data_tot :
                return

            nQuads = data_tot.quads_shape()[0]
            data = map(data_tot.quads, range(nQuads)) # makes list of <psana.CsPad.ElementV2 objects

        else:
            data = evt.get(TypeId.Type.Id_CspadElement, self.m_src)

        if not data :
            msg = __name__ + 'proc_event_for_cspad: DATA IS NOT FOUND'
            print msg
            return

        # 2. Fill output array accounting for dtype and configuration
        if self.m_val_miss == 0 : self.arr = np.zeros((4, 8, 185, 388), dtype=self.m_dtype)
        else                    : self.arr = np.ones ((4, 8, 185, 388), dtype=self.m_dtype) * self.m_dtype(self.m_val_miss)

        for q in data : # where q is an object of pypdsdata.cspad.ElementV2
            quad_num = q.quad()
            roi_mask = self.config.roiMask(quad_num)

            quad_data = q.data()        # shape=(8, 185, 388)
            nsects = quad_data.shape[0]
            if self.m_print_bits & 32 : print 'quad_num=%d  roi_mask(oct)=%o   nsects=%d   data.shape=%s' % (quad_num, roi_mask,  nsects, str(quad_data.shape)) 

            # Copy quad data (N<8, 185, 388) -> to CSPAD arr (4, 8, 185, 388) - changing data type
            ind=0
            for sect in range(8) :
                if roi_mask & (1<<sect) :
                    self.arr[quad_num,sect,:] = quad_data[ind,:] # - copy changing data type
                    ind += 1


    def proc_event_for_cspad2x2( self, evt, env ) :

        if env.fwkName() == "psana":
            elem = evt.get(CsPad2x2.Element, self.m_src) # returns psana.CsPad2x2.ElementV1 object
            if not elem :
                print 'CsPad2x2.Element is not found!'
                return
        else:
            elem = evt.get(TypeId.Type.Id_Cspad2x2Element, self.m_src) # returns CsPad2x2.ElementV1 object
            if not elem :
                return

        if self.m_val_miss == 0 : self.arr = np.zeros((185, 388, 2), dtype=self.m_dtype)
        else                    : self.arr = np.ones ((185, 388, 2), dtype=self.m_dtype) * self.m_dtype(self.m_val_miss)

        self.arr[:] = elem.data()[:] # shape= (185, 388, 2) - copy changing data type

        
    def endcalibcycle( self, evt, env ) : pass

    def endrun( self, evt, env ) : pass

    def endjob( self, evt, env ) : pass


    def print_input_pars( self ) :
        msg = '\n%s: List of input parameters\n  source: %s\n  print_bits: %4d\n  dtype_str: %s\n  dtype_str: %s\n  key_out %s\n  m_val_miss: %s' % \
              (__name__, self.m_src, self.m_print_bits, self.m_dtype_str, str(self.m_dtype), self.m_key_out, self.m_val_miss)
        #logging.info( msg )
        print msg


    def print_part_of_output_array( self ) :
        msg = __name__ + ': arr[2,4,:] :\n' + str(self.arr[2,4,:]) \
            + '\n  arr.shape = %s    arr.dtype = %s' % (str(self.arr.shape), str(self.arr.dtype))
        ##logging.info( msg )
        print msg


    def print_config_pars_for_cspad2x2( self, env ) :
        msg  = '%s: List of configuration parameters for CSPAD2x2' % (__name__)
        print msg
        print "  payloadSize    =", self.config.payloadSize()
        print "  asicMask       =", self.config.asicMask()
        print "  roiMask        =", self.config.roiMask()
        print "  numAsicsRead   =", self.config.numAsicsRead()
        print "  numAsicsStored =", self.config.numAsicsStored()
 

    def print_config_pars_for_cspad( self, env ) :
        msg  = '%s: List of configuration parameters for CSPAD' % (__name__)
        msg += '\n%s: %s' % (self.config.__class__.__name__, self.m_src)
        msg += '\n  numQuads = %d'     % (self.config.numQuads())
        msg += '\n  asicMask = %d'     % (self.config.asicMask())
        msg += '\n  quadMask = %d'     % (self.config.quadMask())
        msg += '\n  numAsicsRead = %d' % (self.config.numAsicsRead())

        try:
            # older versions may not have all methods
            msg +=  '\n  roiMask       : [%s]' % ', '.join([hex(self.config.roiMask(q)) for q in range(4)])
            msg +=  '\n  numAsicsStored: %s' % str(map(self.config.numAsicsStored, range(4)))
        except:
            pass

        #if env.fwkName() == 'pyana':
        #self.list_of_sections = map(self.config.sections, range(4)) 
        #msg +=  '\n  sections      : %s' % str(self.list_of_sections)

        #logging.info( msg )
        print msg

#-----------------------------
#-----------------------------
#-----------------------------
