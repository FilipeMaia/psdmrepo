#!/usr/bin/env python

#--------------------------------
""":py:class:`NDArrSpectrum` - support creation of spectral histogram for arbitrary shaped numpy array.

Usage::

    # Import
    # ==============
    from pyimgalgos.NDArrSpectrum import NDArrSpectrum


    # Initialization
    # ==============
    # 1) for bins of equal size:
         range = (vmin, vmax)
         nbins = 100
         spec = NDArrSpectrum(range, nbins)
    # 2) for variable size bins:
         bins = (v0, v1, v2, ..., v<n-1>, n<n>) 
         spec = NDArrSpectrum(bins)


    # Fill spectrum
    # ==============
    nda = ... (get it for each event somehow)
    spec.fill(nda)


    # Get spectrum
    # ==============
    histarr, edges, nbins = spec.spectrum()


    # Optional
    # ==============
    spec.print_attrs()

@see :py:class:`pyimgalgos.NDArrSpectrum`

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
import numpy as np

#------------------------------
#------------------------------

def arr_bin_indexes(arr, vmin, vmax, nbins) :
    """ Evaluates pixel intensity indexes for spectral histogram in case of equidistant nbins in the range [vmin, vmax].
    """    
    factor = float(nbins)/(vmax-vmin)
    nbins1 = nbins-1
    dtype = np.int32 if nbins>256 else np.int16
    ind = np.array((arr-vmin)*factor, dtype = dtype)
    return np.select((ind<0, ind>nbins1), (0, nbins1), default=ind)

#------------------------------

def arr_varbin_indexes(arr, edges) :
    """ Evaluates pixel intensity indexes for spectral histogram in case of variable size bins.
        For histogram with N bins: number of boarders is N+1, and indexes are in range [0,N-1].
    """
    nbins = len(edges)-1
    dtype = np.int32 if nbins>256 else np.int16
    conds = [arr<edge for edge in edges]
    ivals  = np.array(range(len(edges)), dtype=dtype)
    ivals -= 1 
    ivals[0] = 0 # re-define index for underflow
    return np.select(conds, ivals, default=nbins-1)

#------------------------------

BINS_EQUIDISTANT = 0
BINS_VARSIZE     = 1

#------------------------------

class NDArrSpectrum :
    def __init__(self, edges, nbins=None, pbits=0) :
        """ Constructor
        @param edges - sequence of bin edges
        @param nbins - number of bins in spectrum, if None - edges are used
        @param pbits - print control bits; =0 - print nothing, 1 - object attributes.
        """
        self.vmin, self.vmax = min(edges), max(edges)
        self.edges = edges
        self.pbits = pbits
        self.entry = 0

        if nbins is None :
            self.mode = BINS_VARSIZE
            self.nbins = len(edges)-1
            #sys.exit('ERROR in %s initialization: Variable size bin mode is not implemented yet!' % (__class__.__name__))
        else :
            self.mode = BINS_EQUIDISTANT
            self.nbins = nbins

        if self.pbits : self.print_attrs()


    def print_attrs(self) :
        """ Prints object attributes
        """
        print 'Class %s object attributes:' % (self.__class__.__name__)
        print 'Binning mode: %d, where 0/1 stands for equidistant/variable size bins' % (self.mode)
        print 'Number of bins: %d' % self.nbins
        print 'Bin edges: %s' % str(self.edges)
        print 'vmin = %f\nvmax = %f' % (self.vmin, self.vmax)
        print 'pbits: %d' % (self.pbits)


    def init_spectrum(self, nda) :
        """ Initialization of the spectral histogram array at 1-st entrance in fill(nda)
            @param nda - numpy n-d array with intensities for spectral histogram.
        """         
        self.ashape = nda.shape
        self.asize = 1
        for d in self.ashape : self.asize *=d
        self.hshape = (self.asize, self.nbins)
        self.histarr = np.zeros(self.hshape, dtype=np.uint16)
        self.pix_inds = np.array(range(self.asize), dtype=np.uint32)
        if self.pbits & 1 :
            print 'n-d array shape = %s, size = %d, dtype = %s' % (str(self.ashape), self.asize, str(nda.dtype))
            print 'histogram shape = %s, size = %d, dtype = %s' % (str(self.hshape), self.histarr.size, str(self.histarr.dtype))


    def fill(self, nda) :
        """ Fills n-d array spectrum histogram-array
            @param nda - numpy n-d array with intensities for spectral histogram.
        """         
        if not self.entry : self.init_spectrum(nda)
        self.entry += 1

        arr = nda.flatten() if len(nda.shape)>1 else nda

        bin_inds = arr_bin_indexes(arr, self.vmin, self.vmax, self.nbins) if self.mode == BINS_EQUIDISTANT else \
                   arr_varbin_indexes(arr, self.edges)

        self.histarr[self.pix_inds, bin_inds] += 1


    def spectrum(self) :
        """ Returns accumulated n-d array spectrum, histogram bin edges, and number of bins
        """ 
        return self.histarr, self.edges, self.nbins

#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------

from time import time
import pyimgalgos.GlobalGraphics as gg

#------------------------------

def random_standard_array(shape=(185,388), mu=50, sigma=10) :
    """Returns n-d array of specified shape with random intensities generated for Gaussian parameters.
    """
    return mu + sigma*np.random.standard_normal(shape)

#------------------------------

def example_equidistant() :
    print """Test NDArrSpectrum for equidistant bins"""

    vmin, vmax, nbins = 0, 100, 50 # binning parameters
    mu, sigma = 50, 10             # parameters of random Gaussian distribution of intensities
    nevts = 10                     # number of events in this test
    ashape = (32,185,388)          # data array shape

    spec = NDArrSpectrum((vmin, vmax), nbins, pbits=0377)

    for ev in range(nevts) :
      arr = random_standard_array(ashape, mu, sigma)
      t0_sec = time()
      spec.fill(arr)
      print 'Event:%3d, t = %10.6f sec' % (ev, time()-t0_sec)


    if True :
      histarr, edges, nbins = spec.spectrum()

      #gg.plotImageLarge(arr, amp_range=(vmin,vmax), title='random')
      gg.plotImageLarge(histarr[0:500,:], amp_range=(0,nevts/3), title='indexes')
      gg.show()

#------------------------------

def example_varsize() :
    print """Test NDArrSpectrum for variable size bins"""

    edges = (0, 30, 40, 50, 60, 70, 100) # array of bin edges
    mu, sigma = 50, 10                   # parameters of random Gaussian distribution of intensities
    nevts = 10                           # number of events in this test
    ashape = (32,185,388)                # data array shape

    spec = NDArrSpectrum(edges, pbits=0377)

    for ev in range(nevts) :
      arr = random_standard_array(ashape, mu, sigma)
      t0_sec = time()
      spec.fill(arr)
      print 'Event:%3d, t = %10.6f sec' % (ev, time()-t0_sec)


    if True :
      histarr, edges, nbins = spec.spectrum()

      #gg.plotImageLarge(arr, amp_range=(vmin,vmax), title='random')
      gg.plotImageLarge(histarr[0:500,:], amp_range=(0,nevts/3), title='indexes')
      gg.show()

#------------------------------

def usage() : return 'Use command: python %s <test-number [1-2]>' % sys.argv[0]

def main() :    
    print '\n%s\n' % usage()
    if len(sys.argv) != 2 : example_equidistant()
    elif sys.argv[1]=='1' : example_equidistant()
    elif sys.argv[1]=='2' : example_varsize()
    else                  : sys.exit ('Test number parameter is not recognized.\n%s' % usage())

#------------------------------

if __name__ == "__main__" :
    main()
    sys.exit('End of test')

#------------------------------
