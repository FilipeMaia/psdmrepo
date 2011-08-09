#!/usr/bin/env python
#--------------------

import numpy as np
import matplotlib.pyplot as plt
import sys

print ' sys.argv[0]: ', sys.argv[0]
print ' len(sys.argv): ', len(sys.argv)

if len(sys.argv) == 2 : fname = sys.argv[1]
else                  : fname = '/reg/neh/home/dubrovin/LCLS/PSANA-test/image.txt'

#fname    = '/reg/neh/home/dubrovin/LCLS/HDF5Explorer-v01/camera-ave-CxiDg1.0:Tm6740.0.txt'
print 'fname=', fname





arr = np.loadtxt(fname, dtype=np.float32)

print arr

#plt.imshow(arr, origin='upper', interpolation='nearest', aspect='auto') #,extent=Range)
#plt.clim(1000,2000)
#plt.show()


fig = plt.figure(num=1, figsize=(12,12), dpi=80, facecolor='w',edgecolor='w',frameon=True)
fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)
figAxes = fig.add_subplot(111)
##Range = (ymin, ymax, xmax, xmin)
imAxes = figAxes.imshow(arr, origin='upper', interpolation='nearest', aspect='auto') #,extent=Range)
#imAxes.set_clim(1300,2000)
imAxes.set_clim(0,4000)

colbar = fig.colorbar(imAxes, pad=0.03, fraction=0.04, shrink=1.0, aspect=40, orientation=1)
plt.show()

