#!/usr/bin/env python

"""
find_blobs.py

This script extracts OPAL images and does a basic peakfinding algorithm on them.

-- TJ Lane 9.4.41

CHANGELOG
---------

12/11/14 :: TJL
-- Fixed "firsttime" bug
-- Added discard_small_blobs functionality
-- Fixed x/y flip in draw_blobs


"""


from glob import glob
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage



def find_blobs(image, sigma_threshold=5.0, discard_border=1,
               discard_small_blobs=0):
    """
    Find peaks, or `blobs`, in a 2D image.
    
    This algorithm works based on a simple threshold. It finds continuous
    regions of intensity that are greater than `sigma_threshold` standard
    deviations over the mean, and returns each of those regions as a single
    blob.
    
    Parameters
    ----------
    image : np.ndarray, two-dimensional
        An image to peakfind on.
        
    Returns
    -------
    centers : list of tuples of floats
        A list of the (x,y)/(col,row) positions of each peak, in pixels.
        
    widths : list of tuples of floats
        A list of the (x,y)/(col,row) size of each peak, in pixels.
        
    Optional Parameters
    -------------------
    sigma_threshold : float
        How many standard deviations above the mean to set the binary threshold.
    
    discard_border : int
        The size of a border region to ignore. In many images, the borders are
        noisy or systematically erroneous.
    
    discard_small_blobs : int
        Discard few-pixel blobs, which are the most common false positives
        for the blob finder. The argument specifies the minimal area
        (in pixels) a blob must encompass to be counted. Default: no
        rejections (0 pixels).

    Notes
    -----
    Tests indicate this algorithm takes ~200 ms to process a single image, so
    can run at ~5 Hz on a single processor.
    """
    
    if not len(image.shape) == 2:
        raise ValueError('Can only process 2-dimensional images')
    
    # discard the borders, which can be noisy...
    image[ :discard_border,:] = 0
    image[-discard_border:,:] = 0
    image[:, :discard_border] = 0
    image[:,-discard_border:] = 0
    
    # find the center of blobs above `sigma_threshold` STDs
    binary = (image > (image.mean() + image.std() * sigma_threshold))
    labeled, num_labels = ndimage.label(binary)
    centers = ndimage.measurements.center_of_mass(binary, 
                                                  labeled,
                                                  range(1,num_labels+1))
                                                    
                                                  
    # for each peak, find it's x- & y-width
    #   we do this by measuring how many pixels are above 5-sigma in both the
    #   x and y direction at the center of each blob
    
    widths = []
    warning_printed = False

    for i in range(num_labels)[::-1]: # backwards so pop works below
        
        c = centers[i]
        r_slice = labeled[int(c[0]),:]
        zy = np.where( np.abs(r_slice - np.roll(r_slice, 1)) == i+1 )[0]
        
        c_slice = labeled[:,int(c[1])]
        zx = np.where( np.abs(c_slice - np.roll(c_slice, 1)) == i+1 )[0]
        
        
        if not (len(zx) == 2) or not (len(zy) == 2):
            if not warning_printed:
                print "WARNING: Peak algorithm confused about width of peak at", c
                print "         Setting default peak width (5,5). This warning"
                print "         will only be printed ONCE. Proceed w/caution!"
                warning_printed = True
            widths.append( (5.0, 5.0) )
        else:
            x_width = zx[1] - zx[0]
            y_width = zy[1] - zy[0]

            # if the blob is a "singleton" and we want to get rid
            # of it, we do so, otherwise we add the widths
            if (x_width * y_width) < discard_small_blobs:
                #print "Discarding small blob %d, area %d" % (i, (x_width * y_width))
                centers.pop(i)
            else:
                widths.append( (x_width, y_width) )
        
    assert len(centers) == len(widths), 'centers and widths not same len'

    return centers, widths
    
    
def draw_blobs(image, centers, widths):
    """
    Draw blobs (peaks) on an image.
    
    Parameters
    ----------
    image : np.ndarray, two-dimensional
        An image to render.
    
    centers : list of tuples of floats
        A list of the (x,y) positions of each peak, in pixels.
        
    widths : list of tuples of floats
        A list of the (x,y) size of each peak, in pixels.
    """
    
    plt.figure()
    plt.imshow(image.T, interpolation='nearest')
    
    centers = np.array(centers)
    widths = np.array(widths)
    
    # flip the y-sign to for conv. below
    diagonal_widths = widths.copy()
    diagonal_widths[:,1] *= -1

    for i in range(len(centers)):
       
        # draw a rectangle around the center 
        pts = np.array([
               centers[i] - widths[i] / 2.0,          # bottom left
               centers[i] - diagonal_widths[i] / 2.0, # top left
               centers[i] + widths[i] / 2.0,          # top right
               centers[i] + diagonal_widths[i] / 2.0, # bottom right
               centers[i] - widths[i] / 2.0           # bottom left
              ])
        
        plt.plot(pts[:,0], pts[:,1], color='orange', lw=3)
        
    plt.xlim([0, image.shape[0]])
    plt.ylim([0, image.shape[1]])
    plt.show()
    
    return


def _parse_args():
    
    parser = argparse.ArgumentParser(description='Analyze OPAL images')
    
    parser.add_argument('-r', '--run', type=int,
        default=-1, help='Which run to analyze, -1 for live stream')
    parser.add_argument('-n', '--num-max', type=int,
        default=0, help='Stop after this number of shots is reached')
    parser.add_argument('-v', '--view', action='store_true',
        default=False, help='View each OPAL image (good for debugging)')
    parser.add_argument('-s', '--sigma', type=float,
        default=6.0, help='The number of std above the mean to search for blobs')
    
    args = parser.parse_args()
    
    return args


def _main():

    import psana
    
    args = _parse_args()
    
    if args.run == 0:
        print 'Analyzing data from shared memory...'
        try:
            ds = psana.DataSource('shmem=1_1_psana_XCS.0:stop=no')
        except:
            raise IOError('Cannot find shared memory stream.')
    else:
        print 'Analyzing run: %d' % args.run
        ds = psana.DataSource('exp=sxra8513:run=%d' % args.run) # CHANGE THIS FOR NEW EXPT
    
    # this may also need to change for the new expt
    opal_src = psana.Source('DetInfo(SxrEndstation.0:Opal1000.1)')

    # iterate over events and extract peaks
    for i,evt in enumerate(ds.events()):
        
        # gets an "opal" object
        opal = evt.get(psana.Camera.FrameV1, opal_src)
        if opal:
            image = opal.data16().copy() # this is a numpy array of the opal image
            centers, widths = find_blobs(image, sigma_threshold=args.sigma)
            n_blobs = len(centers)
            print 'Shot %d :: found %d blobs :: %s' % (i, n_blobs, str(centers))
            
            if args.view and (n_blobs > 0):
                draw_blobs(image, centers, widths)
                
        # if we reach the max number of shots, stop
        if i+1 == args.num_max:
            print 'Reached max requested number of shots (%d)' % (i+1,)
            break
        
    return


def _test_on_local_image():
    
    for f in glob('*.npy'):
        print f
        img = np.load(f)
        b = find_blobs(img)
        if len(b[0]) > 0:
            draw_blobs(img, *b)
            
    return


def _test_on_sxra8513_data():

    import psana

    ds = psana.DataSource('exp=sxra8513:run=50')
    opal_src = psana.Source('DetInfo(SxrEndstation.0:Opal1000.2)')

    for i,evt in enumerate(ds.events()):
        opal = evt.get(psana.Camera.FrameV1, opal_src)
        opal_img = opal.data16().copy()

        if opal_img.sum() > 33900000:

            centers, widths = find_blobs(opal_img, 20.0, 1, 5)
            draw_blobs(opal_img, centers, widths)

            print i, opal.data16().sum()

    return

    
if __name__ == '__main__':
    #_main()
    _test_on_sxra8513_data()
    
    
    
