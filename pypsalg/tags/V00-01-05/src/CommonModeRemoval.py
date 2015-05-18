import numpy as np

def CommonModeRemoval(image, stripSize, method) :
    """
    Assume image array is written in 'natural order'
    - array is arrange as [Row][Column]
    - Each row readout simultaneously by Column MOD stripSize ASCIS

    Input image is written over by this function
    """

    # Set up method
    function = None
    if method == "MEDIAN" : function = np.median
    if method == "MEAN" : function = np.mean

    if function is None :
        print method," is not known"
        return

        
    # Get number of columns and rows 
    nrows, ncols = image.shape
    
    # Buffer some indicies we'll be using
    colStart = np.arange(0, ncols, stripSize)
    colEnd = np.arange(colStart[1], ncols+1, stripSize)
    
    # loop over rows
    for index, row in enumerate(image) :
        # loop over chunks of columns for current row
        for start,end in zip(colStart,colEnd) :
            # Extract 1 ASIC of data
            strip = row[start:end]

            # Find the median & RMS of pixel
            #            median = np.median(strip)
            cut = function(strip)
            #            rms = np.std(strip)

            # All pixel values less than mean+4*sigma are set to zero
            #            cut = median + (4.0 * rms)
            #            strip[strip<cut] = 0.0

            #            strip -= median
            strip -= cut
