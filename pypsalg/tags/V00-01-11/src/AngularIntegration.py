"""
Angular Integration Method

Given a center, integrate over phi and histogram values as a
function of radial distance from center.
"""

# Import numpy for fast array calculations & manipulation
import numpy as np

# class AngularIntegration
# One of the drawbacs is the radial array is always recreated. It
# should only be calculated when needed (be lazy).
#
# Flags to monitor how much the radial-center moves by should be used
# to trigger radial array calculation.
#
# This can be done without going to OO. However, these algorithms are
# very general, and one could image multiple integarators running for
# different area-detectors. In that case, having a class, each with
# its own internal data would be handy.
#
# Class AngularIntegrator
#    self.center
#    self.tollerance ---> use to define how far center has to move to
#                         trigger radial array calculation
#    self.radialArray --> internal array of radii
#    self.imageWidth, self.imageHeight --> any change should trigger
#                                          radial array calculation
#    self.1dhistogram ---> to be included when general 1D histogram
#                          class is defined


class AngularIntegrator:
    """
    Class to perform angular integration of a 2D array (typically an image)
    """
    def __init__(self, tollerance=1.0):

        # How much must the center move to trigger a recalulation of
        # the radial array
        self.tollerance = tollerance

        # Cache the x,y center values
        self.__xCenterOld = None
        self.__yCenterOld = None

        # Cache the radial array
        self.__radialArray = None

        # Cache the maximum radial value
        self.__radialMax = None

        # create the radial axis for FAST calculation
#        radialAxis = histaxis

    def __createRadialArray(self, radialCenterX, radialCenterY,
                            imageWidth, imageHeight):
        """
        Purpose: Fill a 2D array of radii using
        (radialCenterX,radialCenterY) as the center

        Parameters
        ----------
        radialCenterX : x co-ordinate of center (column)
        radialCenterY : y co-ordinate of center (row)        

        NB: This assumes the origin is at the bottom-left
        
        imageWidth : Width of the image (number of columns)
        imageHeight: Height of the image (number of rows)
        
        Pixel position is evaulated at its center
        
        Returns
        -------
        result: numpy.ndarray of the 2D array of radii
        """
        
#        print "Making radial matrix...",
        
        xRange = np.linspace(-radialCenterX + 0.5,
                            imageWidth - radialCenterX - 0.5,
                            imageWidth)
        yRange = np.linspace(-radialCenterY+0.5, 
                            imageHeight-radialCenterY-0.5,
                            imageHeight)
        xCoords, yCoords = np.meshgrid(xRange, yRange)
        self.__radial = np.sqrt(xCoords**2 + yCoords**2)

        # update maximum radial value
        self.__radialMax = self.__radial.max()

#        print "Done"
        
        
    def __recalculateRadialArray(self, xCenter, yCenter):
        """
        Method to determine whether to recalculate the radial array 
        """

        # If xCenterOld and yCenterOld are not defined, calculate
        # radial array
        if (self.__xCenterOld is None) or (self.__yCenterOld is None):  
            self.__xCenterOld = xCenter
            self.__yCenterOld = yCenter 
            return True

        # Calculate how far center has moved        
        deltaX = xCenter - self.__xCenterOld
        deltaY = yCenter - self.__yCenterOld
        deltaR = np.sqrt(deltaX**2 + deltaY**2)

        return True if deltaR > self.tollerance else False
    
    
    def angularIntegration(self, image, xCenter, yCenter, 
                           nbins=100, rStart=0.0, rEnd=None):
        """
        Purpose: Integrate in phi about (xCenter, yCenter) histogramming
        data as function of distance(radius) from (xCenter, yCenter)
        
        Paramters
        ---------
        image : 2D numpy array
        Input data for integration
        
        xCenter : float 
        x-position of center for angular integration
        
        yCenter : float 
        y-position of center for angular integration
        
        nBins : interger, optional, default: 100
        Number of bins for output histogram.
        
        rStart : float, optional, default: 0.0
        Lower bound output histogram x-axis
        
        rEnd : float, optional, default None
        Upper bound output histogram x-axis
        If no value is given, it will use the largest radii given the
        image, xCenter, and yCenter
        
        Returns
        -------
        tuple : (array of radial points, array of integrated value at r)
        """

#        print "Starting angular integration calculation"

        # Make Array of radii if needed
        if  self.__recalculateRadialArray(xCenter, yCenter) :
            self.__createRadialArray(xCenter, yCenter,
                                     image.shape[1], image.shape[0])

        # If rEnd isn't defined, use largest value from radii array
        if rEnd is None:
            rEnd = self.__radialMax

        # Make histogram of radii weighted by pixel intesity & return it
        radialInteg, binEdges =                                  \
            np.histogram(self.__radial, 
                         bins=nbins, range=(rStart, rEnd),
                         weights=image) 

        # Take center of binEdges to convert from histogram to X,Y
        # plot 
        radialPoints = (binEdges + 0.5)[:-1]

#        print "Finished angular integration calculation"

        # Now return an array of X,Y points
        return radialPoints, radialInteg




#    def angularIntegration_FAST(self, image, xCenter, yCenter, 
#                                nbins=100, rStart=0.0, rEnd=None):

#        # Make Array of radii if needed
#        if  self.__recalculateRadialArray(xCenter, yCenter) :
#            self.__createRadialArray(xCenter, yCenter,
#                                     image.shape[1], image.shape[0])

#        # If rEnd isn't defined, use largest value from radii array
#        if rEnd is None:
#            rEnd = self.__radialMax

    
##        # Fill histogram
#        # Loop over all rows
#        for row in xrange(image.shape[0]) :
#            # loop over all columns
#            for column in xrange(image.shape[1]) :
#                # get pixel at row,column
#                pixel = image[row][column]

                # get radius
#                radius = self.__radial[row][column]

                # get bin
#                bin = 


                



# AUX class to do histogram axis
#class histaxis:
#   def __init__(self,low,high,nbin):
#       self.low = low
#       self.high = high
#       self.nbin = nbin
#       self.binsize = (high-low)/float(nbin+1)
#   def bin(self,val):
#       return int(math.floor((val-self.low)/self.binsize))                



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

    
    # Instatiate the AngularIntegration class
    pnCCDInteg = AngularIntegrator()


    # Now do angular integration
    print "Doing integration"
    radialPoints, radialInteg = pnCCDInteg.angularIntegration(testImage,
                                                              512,512,500)
    
    print "Sanity check => histogram integral: ", radialInteg.sum()

    # Display images
    print "Display images"
    
    plt.figure(1)
    plt.clf()
    plt.subplot(221)
    plt.imshow(testImage)

    #    ----> plot of radial slice    
    #    plt.subplot(222)
    #   plt.plot(testImage[testImage.shape[1]/2,:])

    # plot radial integral
    plt.subplot(222)
    plt.plot(radialPoints, radialInteg)

    # Move half a pixel
    #    testImage2 = MakeImage(-4.5, 5.5, 1024)
    
    print "Doing integration"
    radialPoints2, radialInteg2 = pnCCDInteg.angularIntegration(testImage,
                                                                512.5,512,500)
        
#    plt.subplot(223)
#    plt.imshow(testImage2)

    plt.subplot(224)
    plt.plot(radialPoints2, radialInteg2)

    plt.show()    
