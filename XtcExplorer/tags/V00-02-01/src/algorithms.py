#import scipy.constants as scipy
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal

def rotate(image, angle):
    image = scipy.ndimage.rotate( image, angle, reshape=False )
    return image

def shift(image, shiftx, shifty=0, reshape = True):
    image = np.roll(image,int(shiftx),1) # column 
    image = np.roll(image,int(shifty),0) # row
    return image


# ------------------------------
# For XCS30412
# ------------------------------

def qvalue(image, energy=None,distance=None):
        
    energy = 9026.0794
    distance = 0
    wavelength = (scipy.h*scipy.c)/energy
    q = (4*scipy.pi)/wavelength

    return 
    
def autocorrelation():
    pass





# ------------------------------
# For AMO42112
# ------------------------------

def apply_median_filter(array,m=2):
    """ Replace each pixel by the median of the surrounding pixels, with a window m pixels wide/high """
    filtered_array = scipy.ndimage.filters.median_filter(array, size=m)
    return filtered_array
    
def apply_threshold(array,thr):
    """ rescale amplitude to value above threshold. Any values below thresholds gets set to zero """
    # rescale:
    array = array - thr

    # set negative values to zero
    array[array<0.0]=0
    
    return array


def gaussian_blur(array, N = 1, sigma = 2000.0, center = 500, edge = 1 ):
    """ convolute image with Gaussian, to blur the image """

    nx,ny = array.shape

    # simple
    #xx,yy = np.mgrid[-nx/2:nx/2+1, -ny/2:ny/2+1]
    #gauss_kern = np.exp( -(xx**2/float(nx/2) + yy**2/float(ny/2) ) )
    #gauss_kern = gauss_kern / gauss_kern.sum()

    xx,yy = np.mgrid[0:nx,0:ny]
    gauss_kern = N * np.exp( - (  (xx-center)**2 / (2* float(sigma) ) 
                                  +(yy-center)**2 / (2* float(sigma) ) ) )
    blurred_image = scipy.signal.fftconvolve(array, gauss_kern, mode='same')
    return blurred_image


    
def find_peaks(array):
    """ find all maxima in the image. Return coordinates (list of (x,y) tuples) """

    # define an 8-connected neighborhood
    neighborhood = scipy.ndimage.morphology.generate_binary_structure(2,2)

    # apply the local maximum filter; all pixel of maximal value 
    # in their neighborhood are set to 1
    local_max = scipy.ndimage.filters.maximum_filter(array, footprint=neighborhood)==array
    # local_max is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (array==0)

    # a little technicality: we must erode the background in order to 
    # successfully subtract it form local_max, otherwise a line will 
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = scipy.ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background
    # a boolean mask.

    # but I want the coordinates!

    (xcoord,ycoord) = np.nonzero(detected_peaks)
    return (xcoord,ycoord)
