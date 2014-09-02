#!/usr/bin/env python
#------------------------------
""" SegGeometry - abstract class with interface description.

Methods of this class should be re-implemented in derived classes with name pattern SegGeometry<SensorVers> 
for pixel geometry description of all sensors.
For example, CSPAD 2x1 sensor is implemented in class :py:class:`PSCalib.SegGeometryCspad2x1V1`.
Access to all implemented sensors is available through the factory method in class :py:class:`PSCalib.SegGeometry`.


Usage of interface methods::

    from SegGeometryCspad2x1V1 import cspad2x1_one as sg

    sg.print_seg_info(pbits=0377)

    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = pixel_scale_size()

    areaA = sg.pixel_area_array()
    
    sizeX = sg.pixel_size_array('X')
    sizeX, sizeY, sizeZ = sg.pixel_size_array()

    X     = sg.pixel_coord_array('X')
    X,Y,Z = sg.pixel_coord_array()
    print 'X.shape =', X.shape

    xmin, ymin, zmin = sg.pixel_coord_min()
    ymax = sg.pixel_coord_max('Y')
    ...


@see :py:class:`PSCalib.SegGeometry`, :py:class:`PSCalib.SegGeometryCspad2x1V1`, :py:class:`PSCalib.SegGeometryStore`

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

Revision: $Revision$

@version $Id$

@author Mikhail S. Dubrovin
"""
#--------------------------------
__version__ = "$Revision$"
#--------------------------------

import sys
#import os
#import math
#import numpy as np

#------------------------------

class SegGeometry :
    AXIS = ['X', 'Y', 'Z']
    DIC_AXIS = {'X':0, 'Y':1, 'Z':2}
    wmsg = 'WARNING! %s - interface method from the base class \nneeds to be re-implemented in the derived class'

    def __init__(self) : 
        pass

#------------------------------

    def print_seg_info(self, pbits=0) :
        """ Prints segment info for selected bits
        """
        print self.wmsg % 'print_seg_info(pbits=0)'

    def size(self) :
        """ Returns segment size - total number of pixels in segment
        """
        print self.wmsg % 'size()'

    def rows(self) :
        """ Returns number of rows in segment
        """
        print self.wmsg % 'rows()'

    def cols(self) :
        """ Returns number of cols in segment
        """
        print self.wmsg % 'cols()'

    def shape(self) :
        """ Returns shape of the segment [rows, cols]
        """
        print self.wmsg % 'shape()'

    def pixel_scale_size(self) :
        """ Returns pixel size in um for indexing
        """
        print self.wmsg % 'pixel_scale_size()'

    def pixel_area_array(self) :
        """ Returns array of pixel relative areas of shape=[rows, cols]
        """
        print self.wmsg % 'pixel_area_array()'

    def pixel_size_array(self, axis) :
        """ Returns array of pixel size in um for AXIS
        """
        print self.wmsg % 'pixel_size_array(axis)'

    def pixel_coord_array(self, axis) :
        """ Returns array of segment pixel coordinates in um for AXIS
        """
        print self.wmsg % 'pixel_coord_array(axis)'

    def pixel_coord_min(self, axis) :
        """ Returns minimal value in the array of segment pixel coordinates in um for AXIS
        """
        print self.wmsg % 'pixel_coord_min(axis)'

    def pixel_coord_max(self, axis) :
        """ Returns maximal value in the array of segment pixel coordinates in um for AXIS
        """
        print self.wmsg % 'pixel_coord_max(axis)'
  
#------------------------------

#    def print_geo(self) :
#        print 'parent:%10s %2d   geo: %10s %2d' % (self.pname, self.pindex, self.oname, self.oindex) + \
#              '  x0:%8.0f  y0:%8.0f  z0:%8.0f' % (self.x0, self.y0, self.z0) + \
#              '  rot_z:%6.1f  rot_y:%6.1f  rot_x:%6.1f' % (self.rot_z, self.rot_y, self.rot_x) + \
#              '  tilt_z:%8.5f  tilt_y:%8.5f  tilt_x:%8.5f' % (self.tilt_z, self.tilt_y, self.tilt_x)

#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    print 'Module %s describes interface methods for segment pixel geometry' % sys.argv[0]

    sg = SegGeometry()
    sg.print_seg_info()
    sg.size()
    sys.exit ('End of %s' % sys.argv[0])

#------------------------------
#------------------------------
#------------------------------


