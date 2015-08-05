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

class ex_peaks_nda (object) :
    """Example module: gets peaks_nda produced by ImgAlgos.CSPadArrPeakFinder and prints its content"""

    def __init__ ( self ) :
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param source      string, address of DetInfo(:Cspad.)
        @param key_in      string, keyword for input image 2-d array
        @param print_bits  int, bit-word for verbosity control 
        """

        self.m_src        = self.configSrc  ('source', ':Cspad.')
        self.m_key_in     = self.configStr  ('key_in',    'peaks_nda')
        self.m_print_bits = self.configInt  ('print_bits', 1)

        self.counter = 0
        self.count_msg = 0

        if self.m_print_bits & 1 : self.print_input_pars()

        self.list_of_dtypes = [
                               psana.ndarray_float32_2,
                               psana.ndarray_float64_2
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

        #print '\nex_peaks_nda: evt.keys():', evt.keys()

        self.arr = None

        if env.fwkName() == "psana":
            for dtype in self.list_of_dtypes :
                self.arr = evt.get(dtype, self.m_src, self.m_key_in)
                if self.arr is not None:
                    break
            
        else : 
            msg = __name__ + ': WARNING!!! THIS MODULE DOES NOT HAVE IMPLEMENTATION FOR PYANA'
            print msg
            return

        self.counter +=1     

        if self.arr is None :
            self.count_msg +=1
            if self.count_msg <20 :
                #if self.m_print_bits & 32 :
                msg = __name__ + ': WARNING! peaks array object %s is not found in evt' % self.m_key_in
                #logging.info( msg )
                print msg
            return


        self.print_nda()


    def endcalibcycle( self, evt, env ) : pass

    def endrun       ( self, evt, env ) : pass

    def endjob       ( self, evt, env ) : pass


#-----------------------------

    def print_input_pars( self ) :
        msg = '\n%s: List of input parameters\n  source %s\n  key_in %s\n  print_bits: %4d' % \
              (__name__ , self.m_src, self.m_key_in, self.m_print_bits)
        #logging.info( msg )
        print msg

#-----------------------------

    def print_nda( self ) :
        print 'Array with peaks: shape=%s' % str(self.arr.shape)

        for row in range(self.arr.shape[0]) :
            arr_row = self.arr[row,:]
            fmt = ' --- q:%d s:%d c:%03d r:%03d sig_c:%8.3f sig_c:%8.3f' \
                + ' Amax:%8.3f Atot:%8.3f Btot:%8.3f noise:%8.3f S/N:%8.3f npix:%d'
            print fmt % (tuple(arr_row))

#-----------------------------
#-----------------------------
