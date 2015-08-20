##-----------------------------
"""Collection of global utilities

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

##-----------------------------
#  Module's version from CVS --
##-----------------------------
__version__ = "$Revision$"
# $Source$

##-----------------------------

import sys
import numpy as np

##-----------------------------

def list_of_windarr(nda, winds=None) :
    """Converts 2-d or 3-d ndarray in the list of 2-d arrays for windows
    """
    ndim = len(nda.shape)
    #print 'len(nda.shape): ', ndim

    if ndim == 2 :
        return [nda] if winds is None else \
               [nda[rmin:rmax, cmin:cmax] for (s, rmin, rmax, cmin, cmax) in winds]

    elif ndim == 3 :
        return [nda[s,:,:] for s in range(ndim.shape[0])] if winds is None else \
               [nda[s, rmin:rmax, cmin:cmax] for (s, rmin, rmax, cmin, cmax) in winds]

    else :
        print 'ERROR in list_of_windarr (with winds): Unexpected number of n-d array dimensions: ndim = %d' % ndim
        return []
##-----------------------------

def mean_of_listwarr(lst_warr) :
    """Evaluates the mean value of the list of 2-d arrays
    """
    s1, sa = 0., 0. 
    for warr in lst_warr :
        sa += np.sum(warr, dtype=np.float64)
        s1 += warr.size
    return sa/s1 if s1 > 0 else 1

##-----------------------------

def subtract_bkgd(data, bkgd, mask=None, winds=None, pbits=0) :
    """Subtracts numpy array of bkgd from data using normalization in windows.
       Each window is specified by 5 parameters: (segment, rowmin, rowmax, colmin, colmax)
       For 2-d arrays segment is not used, but still 5 parameters needs to be specified.
    """
    mdata = data if mask is None else data*mask
    mbkgd = bkgd if mask is None else bkgd*mask

    lwdata = list_of_windarr(mdata, winds)
    lwbkgd = list_of_windarr(mbkgd, winds)
    
    mean_data = mean_of_listwarr(lwdata)
    mean_bkgd = mean_of_listwarr(lwbkgd)

    frac = mean_data/mean_bkgd
    if pbits : print 'subtract_bkgd, fraction = %10.6f' % frac

    return data - bkgd*frac

##-----------------------------
##-----------------------------
##-----------------------------
##-----------------------------
##-----------------------------
##-----------------------------

from pyimgalgos.TestImageGenerator import random_normal

def test_01() :
    shape1 = (32,185,388)

    winds = [ (s, 10, 155, 20, 358) for s in (0,1)]
    data = random_normal(shape=shape1, mu=300, sigma=50, pbits=0377)
    bkgd = random_normal(shape=shape1, mu=100, sigma=10, pbits=0377)

    cdata = subtract_bkgd(data, bkgd, mask=None, winds=winds, pbits=0377)

##-----------------------------

if __name__ == "__main__" :

    test_01()
    sys.exit ( 'End of test.' )

##-----------------------------

