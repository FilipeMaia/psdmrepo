#!/usr/bin/env python
import sys
import numpy as np
from PSCalib.GeometryObject import two2x1ToData2x2, data2x2ToTwo2x1

fin = '/reg/neh/home2/tkroll/gain_map/gains_clean_v1_jano_29_0.3.txt'
fout= 'arr_as_data.txt'

arr = np.loadtxt(fin, dtype=np.float32)
print 'Input arr.shape=', arr.shape
arr.shape = (2,185,388)
print 'Re-shape to arr.shape=', arr.shape
arr_as_data = two2x1ToData2x2(arr)
print 'Shape of the converted arr_as_data.shape=', arr_as_data.shape
arr_as_data.shape = (2,185*388)
print 'Save array with shape=%s in file %s' % (str(arr_as_data.shape), fout)
np.savetxt(fout, arr_as_data, fmt='%f', delimiter=' ')

sys.exit('Done')
