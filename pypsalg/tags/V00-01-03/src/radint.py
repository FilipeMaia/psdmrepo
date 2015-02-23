# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:09:56 2014

@author: Michael Schneider <mschneid@physik.tu-berlin.de>

perform radial integration of 2d array by sorting a recarray with computed
distances from a user-defined centre and add up chunkwise.
I.e. sum intensities in a ring around (x0, y0) between radii r0, r1

EDIT 03/07/14: included azimuth limits and 'buffered' mode: set up sorting
array and chunk limits once and reuse them!
"""

import numpy as np
import time


def distance_array(shape, center, r0, r1, phi0=0, phi1=2 * np.pi):
    xlim, ylim = shape
    x0, y0 = center
    x, y = np.ogrid[-x0:xlim - x0, -y0:ylim - y0]
    distance = np.hypot(x, y)
    if not (phi0 == 0 and phi1 == 2 * np.pi):
        azimuth = np.mod(np.arctan2(x, y), np.pi)
        distance *= azimuth > phi0
        distance *= azimuth < phi1
    mask = np.array(distance, dtype=np.bool)
    start = np.count_nonzero(distance)
    size = shape[0] * shape[1]
    selector = np.argsort(distance, axis=None)
    return selector[size - start:], mask


def compute_chunks(shape, center, r0, r1, dr, phi0=0, phi1=2 * np.pi):
    rmax = np.min(np.array(shape) - np.array(center))
    r1 = np.min([rmax, r1])
    radius = np.arange(r0, r1 + 1, np.min([dr, r1 - r0]))
    phi = np.abs(phi1 - phi0)
    chunks = np.array([phi / 2 * r ** 2 for r in radius], dtype=np.int)
    return radius, chunks


def radint_buffered(image, sort_array, chunks):
    image_flat = image.flatten()[sort_array]
    radialIntegrate = [image_flat[chunks[i]:chunks[i + 1]].sum()
                       for i in range(len(chunks) - 1)]
    return np.array(radialIntegrate)


def prepare_radint(shape, center, r0, r1, dr, phi0=0, phi1=2 * np.pi):
    distance, mask = distance_array(shape, center, r0, r1, phi0, phi1)
    radii, chunks = compute_chunks(shape, center, r0, r1, dr, phi0, phi1)
    return radii[1:], distance, chunks, mask


def benchmark(runs=1000):
    print('generating test data')
    data = np.random.randint(0, 2 ** 14, [1024, 1024])
    print('calculating sorting and selecting indices')
    prepare_start = time.time()
    radii, sorter, chunks, mask = prepare_radint(data.shape, center=[512, 512],
                                                 r0=0, r1=512, dr=2,
                                                 phi0=np.pi / 16, phi1=np.pi / 14)
    prepare_end = time.time()
    print('running buffered radial integration (%d times)\n' % runs)
    radint_start = time.time()
    [radint_buffered(data, sorter, chunks) for N in range(runs)]
    radint_end = time.time()
    preptime = prepare_end - prepare_start
    runtime = radint_end - radint_start
    print('total preparation time (1024 x 1024 array): %.2fs\n' % preptime)
    print('total runtime: %.2fs' % runtime)
    print('time for single (buffered) radial integration: %.4fms' % (runtime / runs * 1000))





# Code to test AngularIntegration module
if __name__ == "__main__" :
    def MakeImage(start, end, points) :
        # Create test image - a sinc function, centered in the middle of
        # the image  
        # Integrating in phi about center, the integral will become sin(x)    
        print "Creating test image",
        axis_image = np.linspace(start,end,points)
        axis_image_X, axis_image_Y = np.meshgrid(axis_image, axis_image)
        axis_image_Radius = np.sqrt(axis_image_X**2 + axis_image_Y**2)
        testImage = np.abs(np.sinc(axis_image_Radius))    
        print "Done"

        return testImage



    
    # Import matplotlib for drawing
    import matplotlib.pyplot as plt

    testImage = MakeImage(-5.0, 5.0, 1024)

    # For sanity checking later, normalise testImage to 1.0
    testImage /= testImage.sum()
    print "Test Image Integral: ", testImage.sum()

    
    # Now do angular integration
    print "Doing integration"


    radii, sorter, chunks, mask = prepare_radint(testImage.shape, center=[512, 512],
                                                 r0=0, r1=512, dr=2)
    radialIntgral = radint_buffered(testImage, sorter, chunks)
