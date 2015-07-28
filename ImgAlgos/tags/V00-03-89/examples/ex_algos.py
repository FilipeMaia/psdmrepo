#!/usr/bin/env python

import sys
import numpy as np
from time import time

from ImgAlgos.PyAlgos import PyAlgos, print_arr, print_arr_attr

##-----------------------------
##-----------------------------
#winds = ((1, 0, 185, 0, 388), \
#         (1, 0, 185, 0, 388))
#winds = ((1, 0, 185, 0, 388),)
#winds = ((1, 10, 103, 10, 204),)
winds = ((0,  0, 185,   0, 388), \
         (1, 10, 103,  10, 204), \
         (1, 10, 103, 250, 380))
winds = None

#print_arr(winds, 'windows')

shape = (2, 185, 388)

mu, sigma = 0, 20
data = np.array(mu + sigma*np.random.standard_normal(shape), dtype=np.float64)
mask = np.ones(shape)
#mask = None
#fname = 'cspad2x2-random.npy'
#np.save(fname, data)
#print 'Random image saved in file %s' % fname


alg = PyAlgos(windows=winds, mask=mask, pbits=0)
#alg = PyAlgos()

alg.print_attributes()
#alg.set_windows(windows)

print_arr_attr(data, 'data')

thr = 20

##-----------------------------

t0_sec = time()
n1 = alg.number_of_pix_above_thr(data, thr)
print '%s\n  alg.number_of_pix_above_thr = %d, fr = %8.6f' % (80*'_', n1, float(n1)/data.size)
print '  Time consumed by the test = %10.6f(sec)' % (time()-t0_sec)

##-----------------------------

t0_sec = time()
a1 = alg.intensity_of_pix_above_thr(data, thr)
print '%s\n  alg.intensity_of_pix_above_thr = %12.3f' % (80*'_', a1)
print '  Time consumed by the test = %10.6f(sec)' % (time()-t0_sec)

##-----------------------------
##-----------------------------
##-----------------------------


sys.exit ('Test example is done.')

##-----------------------------
