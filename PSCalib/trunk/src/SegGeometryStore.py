#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module SegGeometryStore...
#
#------------------------------------------------------------------------

"""
:py:class:`PSCalib.SegGeometryStore` - is a factory class/method to switch between different device-dependent
segments/sensors to access their pixel geometry uling :py:class:`PSCalib.SegGeometry` interface.

Usage::

    from PSCalib.SegGeometryStroe import sgs

    sg = sgs.Create('SENS2X1:V1', pbits=0377)
    sg.print_seg_info(pbits=0377)
    size_arr = sg.size()
    rows     = sg.rows()
    ...


@see other interface methods in :py:class:`PSCalib.SegGeometry`, :py:class:`PSCalib.SegGeometryCspad2x1V1`


This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: 2013-03-08$

@author Mikhail S. Dubrovin
"""

#--------------------------------
#  Module's version from CVS --
#--------------------------------
__version__ = "$Revision$"
# $Source$
#--------------------------------

import sys
#import math
#import numpy as np
#from time import time

#from PSCalib.SegGeometry import *
#import PSCalib.GlobalGraphics as gg # For test purpose in main only

#------------------------------

from PSCalib.SegGeometryCspad2x1V1 import SegGeometryCspad2x1V1 
#from PSCalib.SegGeometryCspad2x1V2 import SegGeometryCspad2x1V2 
#from PSCalib.SegGeometryCspad2x1V3 import SegGeometryCspad2x1V3 

#------------------------------
class SegGeometryStore() :
    """Factory class for SegGeometry objects of different detectors"""

#------------------------------

    def __init__(sp) :
        pass

#------------------------------

    def Create(sp, segname='SENS2X1:V1', pbits=0 ) :
        """ Factory method returns device dependent object with interface implementation  
        """        
        if segname=='SENS2X1:V1' : return SegGeometryCspad2x1V1(use_wide_pix_center=False)
        #if segname=='SENS2X1:V2' : return SegGeometryCspad2x1V2(use_wide_pix_center=False)
        #if segname=='SENS2X1:V3' : return SegGeometryCspad2x1V3(use_wide_pix_center=False)
        return None

#------------------------------

sgs = SegGeometryStore()

#------------------------------
#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------
#------------------------------

def test_seggeom() :
    sg = sgs.Create('SENS2X1:V1', pbits=0377)
    sg.print_seg_info(pbits=0377)

#------------------------------

if __name__ == "__main__" :

    test_seggeom()

    #if len(sys.argv)==1   : print 'For other test(s) use command: python', sys.argv[0], '<test-number=1-3>'
    #elif sys.argv[1]=='0' : test_seggeom()
    #elif sys.argv[1]=='1' : test_seggeom()
    #else : print 'Non-expected arguments: sys.argv=', sys.argv

    sys.exit( 'End of test.' )

#------------------------------
