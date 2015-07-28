#!/usr/bin/env python

import sys
import psana
import numpy as np
from time import time
from Detector.PyDetector import PyDetector
from ImgAlgos.PyAlgos import PyAlgos, print_arr, print_arr_attr


##-----------------------------
# Initialization of graphics
import matplotlib.pyplot as plt
import matplotlib.patches as patches # for patches.Circle

from pyimgalgos.GlobalGraphics import store as sp
import pyimgalgos.GlobalGraphics as gg
#from pyimgalgos.GlobalGraphics import fig_axes, plot_img
##-----------------------------

ntest = int(sys.argv[1]) if len(sys.argv)>1 else 1
print 'Test # %d' % ntest

##-----------------------------

def plot_peaks_on_img(peaks, axim, iX, iY, color='w', pbits=0) :  
    """ Extra drawing on the top of image axes (axim)
        Plots peaks from array as circles in coordinates of image
    """
    if peaks is None : return

    anorm = np.average(peaks,axis=0)[4] if len(peaks)>1 else peaks[0][4] if peaks.size>0 else 100    
    for rec in peaks :
        s, r, c, amax, atot, npix = rec[0:6]
        if pbits & 1 : print 's, r, c, amax, atot, npix=', s, r, c, amax, atot, npix
        x=iX[int(s),int(r),int(c)]
        y=iY[int(s),int(r),int(c)]
        if pbits & 2 : print ' x,y=',x,y        
        xyc = (y,x)
        r0  = 2+6*atot/anorm
        circ = patches.Circle(xyc, radius=r0, linewidth=2, color=color, fill=False)
        axim.add_artist(circ)

##-----------------------------

###================
EVTMAX      = 1
SKIP_EVENTS = 0
EVTSKIPPLOT = 1 #10
DO_PLOT     = False
DO_PLOT     = True
###================


dsname = 'exp=cxif5315:run=169'
src    = psana.Source('DetInfo(CxiDs2.0:Cspad.0)')
print '%s\nExample for\n  dataset: %s\n  source : %s' % (85*'_',dsname, src)

# Non-standard calib directory
#psana.setOption('psana.calib-dir', './calib')
#psana.setOption('psana.calib-dir', './empty/calib')

ds  = psana.DataSource(dsname)
evt = ds.events().next()
env = ds.env()

#for key in evt.keys() : print key

##-----------------------------

det = PyDetector(src, env, pbits=0)
print 85*'_', '\nInstrument: ', det.instrument()

##-----------------------------

fname_mask = '../rel-mengning/work/roi_mask_nda_arc.txt'
#fname_mask = '../rel-mengning/work/roi_mask_nda_equ.txt'
#fname_mask = '../rel-mengning/work/roi_mask_nda_equ_arc.txt'
mask = np.loadtxt(fname_mask)
mask.shape = (32,185,388)
#print_arr_attr(mask, 'mask')

winds_arc = (( 0, 0, 185, 0, 388), \
             ( 1, 0, 185, 0, 388), \
             ( 7, 0, 185, 0, 388), \
             ( 8, 0, 185, 0, 388), \
             ( 9, 0, 185, 0, 388), \
             (15, 0, 185, 0, 388), \
             (16, 0, 185, 0, 388), \
             (17, 0, 185, 0, 388), \
             (23, 0, 185, 0, 388), \
             (24, 0, 185, 0, 388), \
             (25, 0, 185, 0, 388), \
             (31, 0, 185, 0, 388))

winds_equ = (( 0, 0, 185, 0, 388), \
             ( 1, 0, 185, 0, 388), \
             ( 3, 0, 185, 0, 388), \
             ( 8, 0, 185, 0, 388), \
             ( 9, 0, 185, 0, 388), \
             (11, 0, 185, 0, 388), \
             (16, 0, 185, 0, 388), \
             (17, 0, 185, 0, 388), \
             (19, 0, 185, 0, 388), \
             (24, 0, 185, 0, 388), \
             (25, 0, 185, 0, 388), \
             (27, 0, 185, 0, 388))


#winds = None
winds = winds_arc
#print_arr_attr(winds, 'Windows')

alg = PyAlgos(windows=winds, mask=mask, pbits=1)
alg.set_peak_selection_pars(npix_min=2, npix_max=200, amax_thr=0, atot_thr=500, son_min=3)
alg.print_attributes()

##-----------------------------

fig, axim, axcb = gg.fig_axes() if DO_PLOT else (None, None, None)

iXshaped, iYshaped = None, None

xoffset, yoffset =  400,  400
xsize,   ysize   =  950,  950

if DO_PLOT :
    iXshaped = det.indexes_x(evt) - xoffset
    iYshaped = det.indexes_y(evt) - yoffset
    iXshaped.shape = iYshaped.shape = (32, 185, 388)

#plt.ion() # do not hold control on show
#plt.show()

##-----------------------------
t0_sec_evloop = time()
nda = None
peaks = None
i = 0

for i, evt in enumerate(ds.events()) :

    if i<SKIP_EVENTS : continue
    if i>=EVTMAX     : break

    nda = det.calib(evt)

    if nda is not None :

        print 85*'_'
        print 'Event %d' % (i)

        #print_arr_attr(nda, 'calibrated data')
        t0_sec = time()
        #peaks = alg.peak_finder_v1(nda, thr_low=10, thr_high=150, radius=5, dr=0.05) # dr is used for S/N evaluation
        peaks = alg.peak_finder_v2(nda, thr=10, r0=5, dr=0.05)
        print ' ----> peak_finder consumed time = %f sec' % (time()-t0_sec)
        #print_arr_attr(peaks, 'peaks')

        if DO_PLOT and i%EVTSKIPPLOT==0 :
            img = det.image(evt, mask*nda)[xoffset:xoffset+xsize,yoffset:yoffset+ysize]
            ave, rms = img.mean(), img.std()
            amin, amax = ave-1*rms, ave+8*rms
            gg.plot_img(img, mode='do not hold', amin=amin, amax=amax)
            plot_peaks_on_img(peaks, axim, iXshaped, iYshaped, color='w') #, pbits=3)

            #gg.plotHistogram(img, amp_range=(1,100), bins=99, title='Event %d' % i)

            fig.canvas.set_window_title('Event: %d' % i)    
            fig.canvas.draw() # re-draw figure content


print ' ----> Event loop time = %f sec' % (time()-t0_sec_evloop)

##-----------------------------

#if img is None :
#    print 'Image is not available'
#    sys.exit('FURTHER TEST IS TERMINATED')

#gg.plotImageLarge(img, amp_range=(ave-1*rms, ave+5*rms))
#gg.show()

##-----------------------------


plt.ioff() # hold control on show() after the last image
plt.show()
 
##-----------------------------

sys.exit('Test is completed')

##-----------------------------
