"""
Functions to create masks for images (2D arrays) 

"""

# Import numpy
import numpy as np

def angularmask(image, start, stop, xcenter, ycenter):
    """
    Purpose: Mask out an angular region from start to stop using
    (xcenter, ycenter) as origin. All pixels outside angular region
    are masked out (FALSE).

    Parameters
    ----------
    image: 2D array representing the image
    start: starting angle in radians
    stop:  end angle in radians
    xcenter: x co-ordinate for origin
    ycenter: y co-ordinate for origin

    Masks (blocks) a set of pixels (array elements) within angular
    range from start to stop, evaluating the angle from
    (xcenter,ycenter).

    Angles defined as:
             
                  |
      pi/2 --> pi | 0--> pi/2
                  |  
          --------+----------
    -pi/2 --> -pi | 0 --> -pi/2
                  |
                  |
                  
     Angles increase in counter (anti) clockwise direction


    Returns
    -------
    result: boolean array, same dimensions as image
            TRUE  - Unmasked pixel
            FALSE - Masked pixel 
    """

    # Create 2D array of positions w.r.t to origin
    xSize, ySize = image.shape
    y,x = np.ogrid[-ycenter:ySize-ycenter,-xcenter:xSize-xcenter]
    angle_array = np.arctan2(y,x)    

    mask = None
    if stop > start :
        mask = (angle_array >= start) & (angle_array <= stop)
    else:
        mask = ~((angle_array >= stop) & (angle_array <= start))

    return mask



def circularmask(image, radius, xcenter, ycenter):
    """
    Purpose: Mask out a circular region centered at (xcenter,ycenter)
    with radius of 'radius'.  All pixels outside circular region are
    masked out (FALSE).

    Parameters
    ----------
    image: 2D array representing the image
    radius: radius of masked out region
    xcenter: x co-ordinate for origin
    ycenter: y co-ordinate for origin

    Returns
    -------
    result: boolean array, same dimensions as image
            TRUE  - Unmasked pixel
            FALSE - Masked pixel 

    """
    xSize, ySize = image.shape
    y,x = np.ogrid[-ycenter:ySize-ycenter,-xcenter:xSize-xcenter]
    radii_array = np.hypot(x,y)    
    
    mask = (radii_array < radius)

    return mask
    






# Test code for mask
if __name__ == "__main__" :
    import matplotlib.pyplot as plt

    testImage = np.ones([1024,1024])

    mask = angularmask(testImage, 
                       0.125*np.pi,0.25*np.pi, 
                       512,512)
    plt.imshow(testImage*mask,origin='lower')
    plt.draw()
    
    mask = circularmask(testImage, 256,
                        512,512)

    plt.imshow(testImage*mask,origin='lower')
    plt.draw()
    
