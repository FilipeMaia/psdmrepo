import numpy as np

def CommonModeRemoval(image, stripSize, method) :
    """
    Assume image array is written in 'natural order'
    - array is arrange as [Row][Column]
    - Each row readout simultaneously by Column MOD stripSize ASCIS

    method: string argument to set if median or mean to calculate
    common-mode noise in for 'stripSize' length of pixels.
         - MEDIAN : median is used
         - MEAN : mean is used

    Returns image with common-mode noise removed
    """

    # Set up method
    function = None
    if "MEDIAN" == method.upper() : function = np.median
    if "MEAN" == method.upper(): function = np.mean

    if function is None :
        print method," is not known"
        return


    # Reshape the image array to be stripSize-wide rows.
    # re-ordering the array avoids writing python loops and takes
    # adavantge of numpy's fast methods
    re_ordered_image = image.reshape(-1,stripSize)
    
    # Calculate common mode along each row
    # reshape is needed so cm_noise is a column vector
    cm_noise = function(re_ordered_image,axis=1).reshape(-1,1)

    # subtract cm_noise off re_ordered_image
    # and re-order back to original image shape
    clean_image = (re_ordered_image - cm_noise).reshape(image.shape)

    return clean_image
