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

    from PSCalib.SegGeometryStore import sgs

    sg = sgs.Create('SENS2X1:V1', pbits=0377)
    sg2= sgs.Create('EPIX100:V1', pbits=0377)
    sg3= sgs.Create('PNCCD:V1',   pbits=0377)

    sg.print_seg_info(pbits=0377)
    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = sg.pixel_scale_size()
    area     = sg.pixel_area_array()
    mask     = sg.pixel_mask(mbits=0377)    
    sizeX    = sg.pixel_size_array('X')
    sizeX, sizeY, sizeZ = sg.pixel_size_array()
    X        = sg.pixel_coord_array('X')
    X,Y,Z    = sg.pixel_coord_array()
    xmin = sg.pixel_coord_min('X')
    ymax = sg.pixel_coord_max('Y')
    xmin, ymin, zmin = sg.pixel_coord_min()
    xmax, ymax, zmax = sg.pixel_coord_mas()
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

from PSCalib.SegGeometryCspad2x1V1 import cspad2x1_one # SegGeometryCspad2x1V1 
#from PSCalib.SegGeometryCspad2x1V2 import SegGeometryCspad2x1V2 
#from PSCalib.SegGeometryCspad2x1V3 import SegGeometryCspad2x1V3 
from PSCalib.SegGeometryEpix100V1 import epix2x2_one # SegGeometryEpix100V1 
from PSCalib.SegGeometryMatrixV1 import segment_one # SegGeometryMatrixV1 

#------------------------------
class SegGeometryStore() :
    """Factory class for SegGeometry objects of different detectors"""

#------------------------------

    def __init__(sp) :
        pass

#------------------------------

    def Create(sp, segname='SENS2X1:V1', pbits=0 ) :
        """ Factory method returns device dependent SINGLETON object with interface implementation  
        """        
        if segname=='SENS2X1:V1' : return cspad2x1_one # SegGeometryCspad2x1V1(use_wide_pix_center=False)
        #if segname=='SENS2X1:V2' : return SegGeometryCspad2x1V2(use_wide_pix_center=False)
        #if segname=='SENS2X1:V3' : return SegGeometryCspad2x1V3(use_wide_pix_center=False)
        if segname=='EPIX100:V1' : return epix2x2_one # SegGeometryEpix100V1(use_wide_pix_center=False)
        if segname=='PNCCD:V1' :   return segment_one # SegGeometryMatrixV1()
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

    if len(sys.argv)==1   : print 'For test(s) use command: python', sys.argv[0], '<test-number=1-3>'

    elif(sys.argv[1]=='1') :
        sg1 = sgs.Create('SENS2X1:V1', pbits=0377)
        sg1.print_seg_info(pbits=0377)
        
    elif(sys.argv[1]=='2') :
        sg2 = sgs.Create('EPIX100:V1', pbits=0377)
        sg2.print_seg_info(pbits=0377)

    elif(sys.argv[1]=='3') :
        sg2 = sgs.Create('PNCCD:V1', pbits=0377)
        sg2.print_seg_info(pbits=0377)

    else : print 'Non-expected arguments: sys.argv=', sys.argv, ' use 0,1,2,...'

#------------------------------

if __name__ == "__main__" :
    test_seggeom()
    sys.exit( 'End of test.' )

#------------------------------
