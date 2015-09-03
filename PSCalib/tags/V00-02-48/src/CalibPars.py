#!/usr/bin/env python
#------------------------------
""" CalibPars - abstract class with interface description.

Methods of this class should be re-implemented in derived classes with name pattern CalibPars<Detector> 
for different type of detectore.
For example, CSPAD is implemented in class :py:class:`PSCalib.CalibParsCspadV1` etc.
Access to all implemented sensors is available through the factory method in class :py:class:`PSCalib.CalibParsStore`.


Usage::

    from PSCalib.CalibPars import CalibPars

    cp = CalibPars()
    cp.print_attrs()
    size = cp.size()

@see :py:class:`PSCalib.CalibPars`, :py:class:`PSCalib.CalibParsCspad2x1V1`, :py:class:`PSCalib.CalibParsStore`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

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
import PSCalib.GlobalUtils as gu

#------------------------------

class CalibPars :

#------------------------------

    def __init__(self) : 
        """ Constructor
        """
        self.wmsg = 'WARNING! %s.%s' % (self.__class__.__name__,\
                    '%s - interface method from the base class needs to be re-implemented in the derived class.')
        pass

#------------------------------

    def print_attrs(self) :
        """ Prints attributes
        """
        print self.wmsg % 'print_attrs()'

#------------------------------

    def pedestals(self) :
        """ Returns pedestals
        """
        print self.wmsg % 'pedestals()'

#------------------------------

    def pixel_status(self) :
        """ Returns pixel_status
        """
        print self.wmsg % 'pixel_status()'

#------------------------------

    def pixel_rms(self) :
        """ Returns pixel_rms
        """
        print self.wmsg % 'pixel_rms()'

#------------------------------

    def pixel_gain(self) :
        """ Returns pixel_gain
        """
        print self.wmsg % 'pixel_gain()'

#------------------------------

    def pixel_mask(self) :
        """ Returns pixel_mask
        """
        print self.wmsg % 'pixel_mask()'

#------------------------------

    def pixel_bkgd(self) :
        """ Returns pixel_bkgd
        """
        print self.wmsg % 'pixel_bkgd()'

#------------------------------

    def common_mode(self) :
        """ Returns common_mode
        """
        print self.wmsg % 'common_mode()'

#------------------------------
#------------------------------
#------------------------------
#------------------------------

    def ndim(self, ctype=gu.PEDESTALS) :
        """ Returns ndim
        """
        print self.wmsg % 'ndim(ctype)'

#------------------------------

    def shape(self, ctype=gu.PEDESTALS) :
        """ Returns shape
        """
        print self.wmsg % 'shape(ctype)'

#------------------------------

    def size(self, ctype=gu.PEDESTALS) :
        """ Returns size
        """
        print self.wmsg % 'size(ctype)'

#------------------------------

    def status(self, ctype=gu.PEDESTALS) :
        """ Returns status
        """
        print self.wmsg % 'size(status)'

#------------------------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    print 'Module %s describes interface methods to access calibration parameters' % sys.argv[0]

    cp = CalibPars()
    cp.print_attrs()
    size = cp.size()
    sys.exit ('End of %s' % sys.argv[0])

#------------------------------
#------------------------------
#------------------------------
#------------------------------


