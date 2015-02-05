#--------------------------------------------------------------------------
# Description:
#   Test script for ParCorAna
#   
#------------------------------------------------------------------------


#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
#import os
#import stat
#import tempfile
import unittest
#import subprocess as sb
#import collections
#import math
#import numpy as np
#import glob
#-----------------------------
# Imports for other modules --
#-----------------------------
#import psana
#import h5py
#import psana_test.psanaTestLib as ptl

import ParCorAna as corAna

class ParCorAna( unittest.TestCase ) :

    def setUp(self) :
    	""" 
    	Method called to prepare the test fixture. This is called immediately 
    	before calling the test method; any exception raised by this method 
    	will be considered an error rather than a test failure.  
    	"""
        pass

    def tearDown(self) :
        """
        Method called immediately after the test method has been called and 
        the result recorded. This is called even if the test method raised 
        an exception, so the implementation in subclasses may need to be 
        particularly careful about checking internal state. Any exception raised 
        by this method will be considered an error rather than a test failure. 
        This method will only be called if the setUp() succeeds, regardless 
        of the outcome of the test method. 
        """
        pass

    def test_parseDataSetString(self):
        '''test parseDataSetString function
        '''
        dsOpts = corAna.parseDataSetString('exp=amo123:run=12')
        self.assertEqual(dsOpts['exp'],'amo123')
        self.assertEqual(dsOpts['run'],[12])
        self.assertEqual(dsOpts['h5'],False)
        self.assertEqual(dsOpts['xtc'],True)
        self.assertEqual(dsOpts['live'],False)
        self.assertEqual(dsOpts['shmem'],False)
        

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
