#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  class PyAlgos
#
#------------------------------------------------------------------------

"""Class provides access to C++ algorithms from python.


Usage::

    # !!! None is returned everywhere when requested information is missing.

    import psana
    from ImgAlgos.PyAlgos import PyAlgos    

    ...

    det = PyAlgos(src, env, pbits=0)

    # set parameters, if changed
    det.set_env(env)

    det.print_member_data()    

    # get pixel array shape, size, and nomber of dimensions
    shape = det.shape(evt)

    # access intensity calibration parameters
    peds   = det.pedestals(evt)

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------
__version__ = "$Revision$"
# $Source$
##-----------------------------

#import psana                   # moved in __init__().py
#from imgalgos_ext import *     # moved in __init__().py

import sys
import numpy as np

import ImgAlgos

#from ImgAlgos import *
#import Detector.GlobalUtils as gu
#from ImgAlgos.AlgArrProc import AlgArrProc


##-----------------------------

def reshape_nda_to_3d(arr) :
    """Reshape np.array to 3d
    """
    sh = arr.shape
    if len(sh)<4 : return arr

    arr.shape = (arr.size/sh[-1]/sh[-2], sh[-2], sh[-1])
    return arr

##-----------------------------

def print_arr_attr(arr, cmt='') :
    if arr is None :
        print '  %s attributes: array is None' % (cmt)
        return
    nda = np.array(arr)
    print '  %s attributes: size = %d, shape = %s, dtype = %s' % (cmt, nda.size, nda.shape, nda.dtype)

##-----------------------------

def print_arr(arr, cmt='') :
    if arr is None :
        print '  %s attributes: array is None' % (cmt)
        return
    nda = np.array(arr)
    print '\nprint_arr: %s:\n%s' % (cmt, str(nda))
    print '  %s attributes: size = %d, shape = %s, dtype = %s' % (cmt, nda.size, nda.shape, nda.dtype)

##-----------------------------

class PyAlgos :
    """Python wrapper for C++ algorithms

    Low level algorithms are implemented on C++

    @see AlgArrProc - c++ array processing algorithms
    """

##-----------------------------

    def __init__(self, windows=None, mask=None, pbits=0) :
        """Constructor.
        @param windows - tuple, list or numpy array of windows or None
        @param mask  - n-d array with mask or None
        @param pbits - print control bit-word
        """
        #print 'In python c-tor PyAlgos'

        self.pbits = pbits

        self.set_mask(mask)

        if windows is None : self.windows = np.empty((0,0), dtype=np.uint32)
        else               : self.windows = np.array(windows, dtype=np.uint32)

        self.aap = ImgAlgos.AlgArrProc(self.windows, self.pbits)

        self.aap.print_input_pars()
        
        #if pbits : self.print_attributes()

##-----------------------------

    def set_windows(self, windows) :
        """
        @param windows - tuple of windows
        """
        self.windows = np.array(windows, dtype=np.uint32)
        self.aap.set_windows(self.windows)
        
##-----------------------------

    def set_mask(self, mask) :
        """
        @param mask - array with mask 1/0 - good/bad pixel
        """
        if mask is None : self.mask = None
        else : self.mask = np.array(mask, dtype=np.uint16)

##-----------------------------

    def print_attributes(self) :
        print '%s attributes:' % self.__class__.__name__, \
              '\n  pbits  : %d' % self.pbits # , \
              #'\n  windows:%s' % str(self.windows), \
              #'\n  size: %s  shape: %s  dtype: %s' % (self.windows.size, self.windows.shape, self.windows.dtype)
        print_arr_attr(self.windows, cmt='windows')
        print_arr_attr(self.mask, cmt='mask')

##-----------------------------

    def check_mask(self, ndim, dtype=np.uint16) :
        """Returns empty mask for None or re-shaped mask for ndim>3, or self.mask
        """
        if self.pbits & 16 : print_arr_attr(self.mask, cmt='self.mask')
        if self.mask is None :
            if ndim>2 : self.mask = np.empty((0,0,0), dtype=dtype)
            else      : self.mask = np.empty((0,0),   dtype=dtype)
            return

        if ndim>3 : self.mask = reshape_nda_to_3d(self.mask)

##-----------------------------

    def number_of_pix_above_thr(self, arr, thr=0) :

        if self.pbits & 16 : print_arr_attr(arr, cmt='number_of_pix_above_thr input arr:')

        ndim, dtype = len(arr.shape), arr.dtype
        self.check_mask(ndim)
        nda, msk = arr, self.mask
        
        if ndim == 2 :
            if dtype == np.float32: return self.aap.number_of_pix_above_thr_f2(nda, msk, thr)
            if dtype == np.float64: return self.aap.number_of_pix_above_thr_d2(nda, msk, thr)
            if dtype == np.int    : return self.aap.number_of_pix_above_thr_i2(nda, msk, thr)
            if dtype == np.int16  : return self.aap.number_of_pix_above_thr_s2(nda, msk, thr)
            if dtype == np.uint16 : return self.aap.number_of_pix_above_thr_u2(nda, msk, thr)

        if ndim>3 :
            nda = reshape_nda_to_3d(arr)
        
        if dtype == np.float32: return self.aap.number_of_pix_above_thr_f3(nda, msk, thr)
        if dtype == np.float64: return self.aap.number_of_pix_above_thr_d3(nda, msk, thr)
        if dtype == np.int    : return self.aap.number_of_pix_above_thr_i3(nda, msk, thr)
        if dtype == np.int16  : return self.aap.number_of_pix_above_thr_s3(nda, msk, thr)
        if dtype == np.uint16 : return self.aap.number_of_pix_above_thr_u3(nda, msk, thr)

        #if self.pbits :
        print 'WARNING: number_of_pix_above_thr(.) method is not implemented for ndim = %d, dtype = %s' % (ndim, str(dtype))

        return None

##-----------------------------

    def intensity_of_pix_above_thr(self, arr, thr):

        if self.pbits & 16 : print_arr_attr(arr, cmt='intensity_of_pix_above_thr input arr:')

        ndim, dtype = len(arr.shape), arr.dtype
        self.check_mask(ndim)
        nda, msk = arr, self.mask
        
        if ndim == 2 :
            if dtype == np.float32: return self.aap.intensity_of_pix_above_thr_f2(nda, msk, thr)
            if dtype == np.float64: return self.aap.intensity_of_pix_above_thr_d2(nda, msk, thr)
            if dtype == np.int    : return self.aap.intensity_of_pix_above_thr_i2(nda, msk, thr)
            if dtype == np.int16  : return self.aap.intensity_of_pix_above_thr_s2(nda, msk, thr)
            if dtype == np.uint16 : return self.aap.intensity_of_pix_above_thr_u2(nda, msk, thr)

        if ndim>3 :
            nda = reshape_nda_to_3d(arr)
        
        if dtype == np.float32: return self.aap.intensity_of_pix_above_thr_f3(nda, msk, thr)
        if dtype == np.float64: return self.aap.intensity_of_pix_above_thr_d3(nda, msk, thr)
        if dtype == np.int    : return self.aap.intensity_of_pix_above_thr_i3(nda, msk, thr)
        if dtype == np.int16  : return self.aap.intensity_of_pix_above_thr_s3(nda, msk, thr)
        if dtype == np.uint16 : return self.aap.intensity_of_pix_above_thr_u3(nda, msk, thr)

        #if self.pbits :
        print 'WARNING: intensity_of_pix_above_thr(.) method is not implemented for ndim = %d, dtype = %s' % (ndim, str(dtype))

        return None

##-----------------------------

    def peak_finder_v1(self, arr, thr_low, thr_high, radius=5, dr=0.05) :

        if self.pbits & 16 : print_arr_attr(arr, cmt='peak_finder_v1 input arr:')

        ndim, dtype = len(arr.shape), arr.dtype
        self.check_mask(ndim)
        nda, msk = arr, self.mask
        
        if ndim == 2 :
            if dtype == np.float32: return self.aap.peak_finder_v1_f2(nda, msk, thr_low, thr_high, radius, dr)
            if dtype == np.float64: return self.aap.peak_finder_v1_d2(nda, msk, thr_low, thr_high, radius, dr)
            if dtype == np.int    : return self.aap.peak_finder_v1_i2(nda, msk, thr_low, thr_high, radius, dr)
            if dtype == np.int16  : return self.aap.peak_finder_v1_s2(nda, msk, thr_low, thr_high, radius, dr)
            if dtype == np.uint16 : return self.aap.peak_finder_v1_u2(nda, msk, thr_low, thr_high, radius, dr)

        if ndim>3 :
            nda = reshape_nda_to_3d(arr)
        
        if dtype == np.float32: return self.aap.peak_finder_v1_f3(nda, msk, thr_low, thr_high, radius, dr)
        if dtype == np.float64: return self.aap.peak_finder_v1_d3(nda, msk, thr_low, thr_high, radius, dr)
        if dtype == np.int    : return self.aap.peak_finder_v1_i3(nda, msk, thr_low, thr_high, radius, dr)
        if dtype == np.int16  : return self.aap.peak_finder_v1_s3(nda, msk, thr_low, thr_high, radius, dr)
        if dtype == np.uint16 : return self.aap.peak_finder_v1_u3(nda, msk, thr_low, thr_high, radius, dr)

        #if self.pbits :
        print 'WARNING: peak_finder_v1(.) method is not implemented for ndim = %d, dtype = %s' % (ndim, str(dtype))

        return None

##----------------------------

    def set_son_parameters(r0=5, dr=0.05) :
        self.aap.set_son_parameters(r0, dr)

##-----------------------------

    def peak_finder_v2(self, arr, thr=0, r0=5, dr=0.05) :

        #self.set_son_parameters(r0, dr)

        if self.pbits & 16 : print_arr_attr(arr, cmt='peak_finder_v2 input arr:')

        ndim, dtype = len(arr.shape), arr.dtype
        self.check_mask(ndim)
        nda, msk = arr, self.mask
        
        if ndim == 2 :
            if dtype == np.float32: return self.aap.peak_finder_v2_f2(nda, msk, thr, r0, dr)
            if dtype == np.float64: return self.aap.peak_finder_v2_d2(nda, msk, thr, r0, dr)
            if dtype == np.int    : return self.aap.peak_finder_v2_i2(nda, msk, thr, r0, dr)
            if dtype == np.int16  : return self.aap.peak_finder_v2_s2(nda, msk, thr, r0, dr)
            if dtype == np.uint16 : return self.aap.peak_finder_v2_u2(nda, msk, thr, r0, dr)

        if ndim>3 :
            nda = reshape_nda_to_3d(arr)
        
        if dtype == np.float32: return self.aap.peak_finder_v2_f3(nda, msk, thr, r0, dr)
        if dtype == np.float64: return self.aap.peak_finder_v2_d3(nda, msk, thr, r0, dr)
        if dtype == np.int    : return self.aap.peak_finder_v2_i3(nda, msk, thr, r0, dr)
        if dtype == np.int16  : return self.aap.peak_finder_v2_s3(nda, msk, thr, r0, dr)
        if dtype == np.uint16 : return self.aap.peak_finder_v2_u3(nda, msk, thr, r0, dr)

        #if self.pbits :
        print 'WARNING: peak_finder_v2(.) method is not implemented for ndim = %d, dtype = %s' % (ndim, str(dtype))

        return None

##-----------------------------

#    def raw_data(self, evt) :

#        # get data using python methods
#        rdata = self.pyda.raw_data(evt, self.env)
#        if rdata is not None : return rdata

#        if self.pbits :
#            print '!!! PyAlgos: Data for source %s is not found in python interface, trying C++' % self.source,

#        # get data using C++ methods
#        if   self.dettype == gu.CSPAD    : rdata = self.da.data_int16_3 (evt, self.env)
#        elif self.dettype == gu.CSPAD2X2 : rdata = self.da.data_int16_3 (evt, self.env)
#        elif self.dettype == gu.PNCCD    : rdata = self.da.data_uint16_3(evt, self.env)
#        else :                             rdata = self.da.data_uint16_2(evt, self.env)
#        return self._nda_or_none_(rdata)

##-----------------------------

#    def common_mode_apply(self, evt, nda) :
#        """Apply common mode correction to nda (assuming that nda is data ndarray with subtracted pedestals)
#           nda.dtype = np.float32 (or 64) is considered only, because common mode does not make sense for int data.
#        """
#        shape0 = nda.shape
#        nda.shape = (nda.size,)
#        if nda.dtype == np.float64 : self.da.common_mode_double(evt, self.env, nda)
#        if nda.dtype == np.float32 : self.da.common_mode_float (evt, self.env, nda)
#        nda.shape = shape0
        
##-----------------------------

from time import time

##-----------------------------

if __name__ == "__main__" :

    t0_sec = time()

    windows = ((0, 0, 185, 0, 388), \
               (1, 0, 185, 0, 388), \
               (3, 0, 185, 0, 388))
    print_arr(windows, "windows")
    
    alg = PyAlgos(windows, pbits=0)
    alg.print_attributes()

    print '\nC++ consumed time to get raw data (sec) = %10.6f' % (time()-t0_sec)

    sys.exit ('Self test is done.')

##-----------------------------
