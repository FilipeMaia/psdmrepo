#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotsForCSpad...
#
#------------------------------------------------------------------------

"""Plots for CSpad detector in the EventeDisplay project.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
#from matplotlib.patches import Rectangle
#from matplotlib.artist  import Artist
#from matplotlib.lines   import Line2D
import time
from numpy import *  # for use like       array(...)
import numpy as np
import scipy.ndimage as spi # rotate(...)
import math # cos(x), sin(x), radians(x), degrees()

#-----------------------------
# Imports for other modules --
#-----------------------------

import ConfigParameters as cp
import ConfigCSpad      as cs
import PrintHDF5        as printh5
import GlobalMethods    as gm

#---------------------
#  Class definition --
#---------------------
class PlotsForCSpad ( object ) :
    """Plots for CSpad detector in the EventeDisplay project."""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor - initialization."""

        #print '\n Initialization of the PlotsForCSpad'
        #print 'using MPL version: ', matplotlib.__version__
        #self.fig1_window_is_open = False

        self.eventWithAlreadyGeneratedCSpadDetImage   = None
        self.dsnameWithAlreadyGeneratedCSpadDetImage  = None

    #-------------------
    #  Public methods --
    #-------------------

    def plotCSpadV1Image( self, arr1ev, fig, plot=8 ):
        """Plot 2d image from input array. V1 for run ~546, array for entire detector"""
        self.quad=cp.confpars.cspadQuad
        arr1quad = arr1ev[self.quad,...] 

        if plot ==  1     : self.plotCSpadPairImage   ( arr1quad, fig )
        if plot ==  8     : self.plotCSpad08PairsImage( arr1quad, fig )
       #if plot == 'Quad' : self.plotCSpadQuadImage   ( arr1quad, fig )

    def plotCSpadV2Image( self, arr1ev, fig, plot=8 ):
        """Plot 2d image from input array. V2 for run ~900 contain array for quad 2"""
        self.quad=cp.confpars.cspadQuad        
        #pair1 = cs.confcspad.firstPairInQuad[self.quad]
        #pairN = cs.confcspad. lastPairInQuad[self.quad]
        ##arr1quad = arr1ev[pair1:pairN, 0:185, 0:388]
        #arr1quad = arr1ev[pair1:pairN,...]

        #print 'Quad=',          self.quad
        #print 'arr1quad.shape=',arr1quad.shape
        #print 'arr1quad.size =',arr1quad.size
        #print 'arr1quad.dtype=',arr1quad.dtype
        #print 'arr1quad.ndim =',arr1quad.ndim

        if plot ==  1     : self.plotCSpadPairImage   ( arr1ev,    fig )
        if plot ==  8     : self.plotCSpad08PairsImage( arr1ev,    fig )
        if plot == 'Quad' : self.plotCSpadQuadImage   ( arr1ev,    fig )
        if plot == 'Det'  : self.plotCSpadDetImage    ( arr1ev,    fig )


    def plotCSpadV1Spectrum( self, arr1ev, fig, plot=16 ):
        """Plot 2d image from input array. V1 for run ~546, array for entire detector"""
        self.quad=cp.confpars.cspadQuad
        arr1quad = arr1ev[self.quad,...] 
        if plot ==  8 : self.plotCSpadQuad08SpectraOf2x1( arr1quad, fig )
        if plot == 16 : self.plotCSpadQuad16Spectra     ( arr1quad, fig )


    def plotCSpadV2Spectrum( self, arr1ev, fig, plot=16 ):
        """Plot 2d image from input array. V2 for run ~900 contain array for quad 2"""
        self.quad=cp.confpars.cspadQuad
        #pair1 = cs.confcspad.firstPairInQuad[self.quad]
        #pairN = cs.confcspad. lastPairInQuad[self.quad]
        #arr1quad = np.zeros( (8,185,388) ) # In order to have array for all 8 pairs!!!!!
        #arr1quad[0:pairN-pair1, 0:185, 0:388] += arr1ev[pair1:pairN,0:185, 0:388]

        if plot ==  8        : self.plotCSpadQuad08SpectraOf2x1( arr1ev, fig )
        if plot == 16        : self.plotCSpadQuad16Spectra     ( arr1ev, fig )
        if plot == 'DetSpec' : self.plotCSpadDetSpectrum       ( arr1ev, fig )


    def plotCSpad08PairsImage( self, arr1ev, fig ):
        """Plot 2d image of 8 2x1 pairs of ASICs' from input array."""

        if cs.confcspad.isCSPad2x2 : # For 2x2 Mini
            print 'WARNING: plotCSpad08PairsImage(...) - these images are not available for CSPad2x2 ...'
            return

        #print 'plot_CSpadQuad()'       

        str_event = 'Event ' + str(cp.confpars.eventCurrent) + '  Quad ' + str(self.quad)
        #plt.title(str_event,color='r',fontsize=40) # pars like in class Text

        fig.canvas.set_window_title('CSpad image ' + str_event)
        plt.clf() # clear plot


        arrgap=zeros( (185,4) ) # make additional 2D-array of 0-s for the gap between two 1x1 pads
        
        for ind in xrange(8): # loop over ind = 0,1,2,...,7
            pair = cs.confcspad.indPairsInQuads[self.quad][ind]
            #print 'quad,ind,pair=', self.quad, ind, pair
            if pair == -1 : continue

            asic1x2  = arr1ev[pair,...]
            #print 'asic1x2=',asic1x2    
        
            asics    = hsplit(asic1x2,2)
            arr      = hstack((asics[0],arrgap,asics[1]))
        
            panel = 421+ind
            pantit='ASIC ' + str(2*ind) + ', ' + str(2*ind+1)

            plt.subplot(panel)
            #plt.imshow(arr,  origin='down', interpolation='nearest') # Just a histogram
            plt.imshow(arr, interpolation='nearest') # Just a histogram
            plt.clim(cp.confpars.cspadImageAmin,cp.confpars.cspadImageAmax)
            plt.title(pantit,color='r',fontsize=20) # pars like in class Text
            if ind==0 :
                plt.text(280, -10, str_event, fontsize=24)


    def getImageArrayForPair( self, arr1ev, pairNum=None ):
        """Returns the image array for pair of ASICs"""
        if pairNum == None :
            self.pair = cp.confpars.cspadPair
        else :
            self.pair = pairNum

        if cs.confcspad.isCSPad2x2 : # For 2x2 Mini
            return self.getImageArrayForMiniElementPair(arr1ev,self.pair)

        #print 'getImageArrayForPair(), pair=', self.pair

        asic2x1 = arr1ev[self.pair,...]
        #print 'asic2x1=',asic2x1    
        asics   = hsplit(asic2x1,2)
        #arrgap=zeros( (185,3), dtype=np.float ) # make additional 2D-array of 0-s for the gap between two 1x1 pads
        arrgap=zeros( (185,3), dtype=np.int16 ) # make additional 2D-array of 0-s for the gap between two 1x1 pads
        arr2d = hstack((asics[0],arrgap,asics[1]))
        #print 'arr2d=\n', arr2d    
        return arr2d


    def getImageArrayForQuad( self, arr1ev, quadNum=None ):
        """Returns the image array for one quad"""

        if cs.confcspad.isCSPad2x2 : # For 2x2 Mini
            return self.getImageArrayForCSPadMiniElement( arr1ev )

        if quadNum == None :
            self.quad = cp.confpars.cspadQuad
        else :
            self.quad = quadNum            
        #print 'getImageArrayForQuad(), quad=', self.quad

        #arr2dquad = np.zeros( (850,850), dtype=np.int16 )
        arr2dquad = np.zeros( (cs.confcspad.quadDimX,cs.confcspad.quadDimY), dtype=np.float ) # dtype=np.int16 
        #print 'arr2dquad.shape=',arr2dquad.shape

        for ind in xrange(8): # loop over ind = 0,1,2,...,7
#        for ind in xrange(1): # loop over ind = 0,1,2,...,7
            pair = cs.confcspad.indPairsInQuads[self.quad][ind]
            #print 'quad,ind,pair=', self.quad, ind, pair
            if pair == -1 : continue

            asic2x1 = self.getImageArrayForPair( arr1ev, pair )
            rotarr2d_0 = np.rot90(asic2x1,cs.confcspad.pairInQaudOriInd[self.quad][ind])
            #print 'rotarr2d_0.shape=',rotarr2d_0.shape
            #print 'rotarr2d.base is asic2x1 ? ',rotarr2d.base is asic2x1 

            rotarr2d = rotarr2d_0

            offset = cs.confcspad.preventiveRotationOffset
            ixOff  = offset + cs.confcspad.pairXInQaud[self.quad][ind] + cs.confcspad.dXInQaud[self.quad][ind] 
            iyOff  = offset + cs.confcspad.pairYInQaud[self.quad][ind] + cs.confcspad.dYInQaud[self.quad][ind]
            #print 'ixOff, iyOff :', ixOff, iyOff

            # 0:185, 0:388 -> 185x391
            rot_index = cs.confcspad.pairInQaudOriInd[self.quad][ind] 

            offS = 0.5*185
            offL = 0.5*(388+3)
            #print 'offS, offL :', offS, offL

            if rot_index == 0 or rot_index == 2 :
                self.lx0 = offS  
                self.ly0 = offL  
            else :
                self.lx0 = offL  
                self.ly0 = offS  

            ixOff -= self.lx0  
            iyOff -= self.ly0  

            #-------- Apply tilt angle of 2x1 sensors
            if cp.confpars.cspadApplyTiltAngle :

                r0      = math.sqrt( self.lx0*self.lx0 + self.ly0*self.ly0 )
                sinPhi  = self.ly0 / r0
                cosPhi  = self.lx0 / r0

                angle  = cs.confcspad.dPhi[self.quad][ind]
                rotarr2d = spi.rotate(rotarr2d_0, angle, reshape=True, output=np.float32 )
                dimX0,dimY0 = rotarr2d_0.shape

                rdphi = r0 * abs(math.radians(angle))
                #print 'rdphi :',rdphi

                ixOff -= rdphi * sinPhi
                iyOff -= rdphi * cosPhi

                #print 'Tilt offset dx, dy=', rdphi * sinPhi, rdphi * cosPhi

            #-------- 

            ixOff = int( ixOff )
            iyOff = int( iyOff )

            dimX, dimY = rotarr2d.shape
            #print 'ixOff, iyOff =', ixOff, iyOff,           
            #print ' dimX,  dimY =', dimX, dimY           
            
            arr2dquad[ixOff:dimX+ixOff, iyOff:dimY+iyOff] += rotarr2d[0:dimX, 0:dimY]

        #print 'arr2dquad=\n', arr2dquad
        return arr2dquad


    def resetEventWithAlreadyGeneratedCSpadDetImage( self ):
        """This method is used in order to update image for averaging"""
        self.eventWithAlreadyGeneratedCSpadDetImage   = None
        self.dsnameWithAlreadyGeneratedCSpadDetImage  = None


    def getQuadNumberForIndex( self, index ):
        quad = int( cs.confcspad.quad_nums_in_event[ind] )
        print 'index -> quad :', index, quad
        return quad


    def getIndexForQuadNumber( self, quad ):
        for index in range(len(cs.confcspad.quad_nums_in_event)) :
            quad_i = int(cs.confcspad.quad_nums_in_event[index])
            if quad_i == quad :
                print 'quad -> index :', quad, index               
                return index


    def getImageArrayForDet( self, arr1ev ):
        """Returns the image array for entire CSpad detector"""       

        if  cp.confpars.eventCurrent == self.eventWithAlreadyGeneratedCSpadDetImage and cp.confpars.cspadCurrentDSName == self.dsnameWithAlreadyGeneratedCSpadDetImage :
            #print 'Use already generated image for CSpad and save time'
            return self.arr2dCSpad

        if cs.confcspad.isCSPad2x2 : # For 2x2 Mini
            self.arr2dCSpad = self.getImageArrayForCSPadMiniElement( arr1ev )

        else : # For regular CSPad detector
            self.arr2dCSpad = self.getImageArrayForCSPadElement( arr1ev )

        self.eventWithAlreadyGeneratedCSpadDetImage  = cp.confpars.eventCurrent
        self.dsnameWithAlreadyGeneratedCSpadDetImage = cp.confpars.cspadCurrentDSName

        if cp.confpars.bkgdSubtractionIsOn : self.arr2dCSpad -= cp.confpars.arr_bkgd
        if cp.confpars.gainCorrectionIsOn  : self.arr2dCSpad *= cp.confpars.arr_gain

        return self.arr2dCSpad


    def getImageArrayForCSPadElement( self, arr1ev ):
        """Returns the image array for the CSPad detector for dataset CSPadElement"""

        #self.arr2dCSpad = np.zeros( (1710,1710), dtype=np.int16 )
        #self.arr2dCSpad = np.zeros( (1750,1750), dtype=np.int16 )
        #self.arr2dCSpad = np.zeros( (1765,1765), dtype=np.int16 )
        self.arr2dCSpad = np.zeros( (cs.confcspad.detDimX,cs.confcspad.detDimY), dtype=np.float ) # dtype=np.int16

        #for quad in range(0,4) :
        for quad in range(len(cs.confcspad.quad_nums_in_event)) :
            #self.quad_index = self.getIndexForQuadNumber( quad )
            arr2dquad = self.getImageArrayForQuad(arr1ev, quad)
            rotarr2d = np.rot90(arr2dquad,cs.confcspad.quadInDetOriInd[quad])
            #print 'rotarr2d.shape=',rotarr2d.shape
            dimX,dimY = rotarr2d.shape

            ixOff = cs.confcspad.quadXOffset[quad]
            iyOff = cs.confcspad.quadYOffset[quad]

            self.arr2dCSpad[ixOff:dimX+ixOff, iyOff:dimY+iyOff] += rotarr2d[0:dimX, 0:dimY]

        return self.arr2dCSpad


    def getImageArrayForMiniElementPair( self, arr1ev, pairNum=None ):
        """Returns the image array for pair of ASICs"""
        if pairNum == None :
            self.pair = cp.confpars.cspadPair
        else :
            self.pair = pairNum

        #arr2x1 = arr1ev[0:185,0:388,self.pair]
        arr2x1 = arr1ev[:,:,self.pair]
        #print 'arr2x1=',arr2x1 

        asics = hsplit(arr2x1,2)
        arrgap=zeros( (185,3), dtype=np.float ) #dtype=np.int16 # make additional 2D-array of 0-s for the gap between two 1x1 pads

        #print 'asics[0].shape=',asics[0].shape
        #print 'asics[1].shape=',asics[1].shape
        #print 'arrgap.shape=',arrgap.shape

        arr2d = hstack((asics[0],arrgap,asics[1]))
        return arr2d


    def getImageArrayForCSPadMiniElement( self, arr1ev ):
        """Returns the image array for the CSpadMiniElement or CSpad2x2"""       

        arr2x1Pair0 = self.getImageArrayForMiniElementPair(arr1ev,0)
        arr2x1Pair1 = self.getImageArrayForMiniElementPair(arr1ev,1)
        wid2x1      = arr2x1Pair0.shape[0]
        len2x1      = arr2x1Pair0.shape[1]

        arrgapV = zeros( (20,len2x1), dtype=np.float ) # dtype=np.int16 
        arr2d   = vstack((arr2x1Pair0, arrgapV, arr2x1Pair1))

        #print 'arr2d.shape=', arr2d.shape
        #print 'arr2d=',       arr2d
        return arr2d








    def plotCSpadQuadImage( self, arr1ev, fig ):
        """Plot 2d image of the quad from input array."""
        #print 'plotCSpadQuadImage()'       

        arr2d = self.getImageArrayForQuad( arr1ev )
        #arr2d = self.getImageArrayForPair( arr1ev )

        str_event = 'Event ' + str(cp.confpars.eventCurrent)
        fig.canvas.set_window_title('CSpad image ' + str_event)
        plt.clf() # clear plot
        fig.subplots_adjust(left=0.03, bottom=0.03, right=0.98, top=0.97, wspace=0, hspace=0)
        
        plottit='Event ' + str(cp.confpars.eventCurrent) + '  Quad '+str(cp.confpars.cspadQuad)
        axes = plt.imshow(arr2d, interpolation='nearest') # ,origin='down'
        #axes = plt.imshow(arr2d, origin='down', interpolation='nearest') # ,origin='down'

        plt.clim(cp.confpars.cspadImageAmin,cp.confpars.cspadImageAmax)
        self.colb = plt.colorbar(axes, pad=0.03, orientation=1, fraction=0.10, shrink = 0.86, aspect = 20)#, ticks=coltickslocs
        plt.title(plottit,color='r',fontsize=20) # pars like in class Text


    def plotCSpadDetImage( self, arr1ev, fig ):
        """Plot 2d image of the detector from input array."""
        #print 'plotCSpadDetImage()'       
        t_plotCSpadDetImage = time.clock()
        self.arr2d = self.getImageArrayForDet( arr1ev )
        #print 'Time to getImageArrayForDet() (sec) = %f' % (time.clock() - t_plotCSpadDetImage)
        self.str_event = 'Event ' + str(cp.confpars.eventCurrent)
        self.figDet = fig
        self.figDet.canvas.set_window_title('CSpad image ' + self.str_event)
        self.drawCSpadDetImage(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax)


    def drawCSpadDetImage( self, xmin=None, xmax=None, ymin=None, ymax=None ):
        plt.clf() # clear plot  t=0.05s
        self.figDet.subplots_adjust(left=0.03, bottom=0.03, right=0.98, top=0.97, wspace=0, hspace=0)

        if xmin == None :
            self.arrwin  = self.arr2d
            self.range   = None # original image range in pixels
        else :
            self.arrwin = self.arr2d[ymin:ymax,xmin:xmax]
            self.range  = [xmin, xmax, ymax, ymin]
    
        axescb = plt.imshow(self.arrwin, origin='upper',interpolation='nearest',extent=self.range) # Just a histogram t=0.08s

        self.addSelectionRectangle()

        self.axesDet = plt.gca()
        plt.clim(cp.confpars.cspadImageAmin,cp.confpars.cspadImageAmax)     #t=0
        self.colb = plt.colorbar(axescb, pad=0.03, orientation=1, fraction=0.10, shrink = 0.86, aspect = 20)#, ticks=coltickslocs #t=0.04s
        plt.title(self.str_event,color='r',fontsize=20) # pars like in class Text

        #self.figDet.canvas.mpl_connect('button_press_event',   self.processMouseButtonPressForDetImage)
        #rect_props=dict(edgecolor='black', linewidth=2, linestyle='dashed', fill=False)
        #self.figDet.span = RectangleSelector(self.axesDet, self.onRectangleSelect, drawtype='box',rectprops=rect_props)

        self.figDet.canvas.mpl_connect('button_release_event', self.processMouseButtonReleaseForImage)
        #self.drawCSpadDetSpectrum(self.arrwin)


    def processMouseButtonReleaseForImage(self, event) :
        #print 'processMouseButtonReleaseForImage'
        fig = self.figDet = event.canvas.figure # or plt.gcf()
        figNum = fig.number 
        
        if event.button == 1 :
            bounds = fig.gca().viewLim.bounds
            fig.myXmin = Xmin = bounds[0]
            fig.myXmax = Xmax = bounds[0] + bounds[2] 
            fig.myYmin = Ymin = bounds[1] + bounds[3]
            fig.myYmax = Ymax = bounds[1]
            fig.myZoomIsOn = True
            #print ' Xmin, Xmax, Ymin, Ymax =', Xmin, Xmax, Ymin, Ymax

        if event.button == 2 or event.button == 3 : # middle or right button
            fig.myXmin = None
            fig.myXmax = None
            fig.myYmin = None
            fig.myYmax = None
            self.drawCSpadDetImage()
            #plt.draw() # redraw the current figure
            fig.myZoomIsOn = False

        arrwin = plt.gci().get_array() # this returns the full size image...
        self.drawCSpadDetSpectrum(arrwin)

            
    def onRectangleSelect(self, eclick, erelease) :
        if eclick.button == 1 : # left button

            self.figDet = plt.gcf() # Get current figure
            #if self.figDet.myZoomIsOn :
            #    print 'Zoom is already applied. Click other mouse buttons to unzoom first.'
            #    return

            xmin = int(min(eclick.xdata, erelease.xdata))
            ymin = int(min(eclick.ydata, erelease.ydata))
            xmax = int(max(eclick.xdata, erelease.xdata))
            ymax = int(max(eclick.ydata, erelease.ydata))
            print 'xmin, xmax, ymin, ymax: ', xmin, xmax, ymin, ymax

            if xmax-xmin < 20 or ymax-ymin < 20 : return
            self.drawCSpadDetImage( xmin, xmax, ymin, ymax )
            plt.draw() # redraw the current figure

            self.figDet.myXmin = xmin
            self.figDet.myXmax = xmax
            self.figDet.myYmin = ymin
            self.figDet.myYmax = ymax
            self.figDet.myZoomIsOn = True


    def processMouseButtonPressForDetImage(self, event) :
        #print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y,
        #print ' xdata=',event.xdata,' ydata=', event.ydata
        self.figDet = plt.gcf() # Get current figure
        print 'mouse click button=', event.button
        if event.button == 2 or event.button == 3 : # middle or right button
            self.figDet.myXmin = None
            self.figDet.myXmax = None
            self.figDet.myYmin = None
            self.figDet.myYmax = None
            self.drawCSpadDetImage()
            plt.draw() # redraw the current figure
            self.figDet.myZoomIsOn = False


    def addSelectionRectangle( self ):
        if cp.confpars.selectionIsOn :
            for win in range(cp.confpars.selectionNWindows) :

                #print 'Selection for dataset:', cp.confpars.selectionWindowParameters[win][6]
                if gm.CSpadIsInTheName(cp.confpars.selectionWindowParameters[win][6]) :

                    xy = cp.confpars.selectionWindowParameters[win][2],  cp.confpars.selectionWindowParameters[win][4]
                    w  = cp.confpars.selectionWindowParameters[win][3] - cp.confpars.selectionWindowParameters[win][2]
                    h  = cp.confpars.selectionWindowParameters[win][5] - cp.confpars.selectionWindowParameters[win][4]

                    rec = plt.Rectangle(xy, width=w, height=h, edgecolor='w', linewidth=2, fill=False)
                    plt.gca().add_patch(rec)


    def plotCSpadDetSpectrum( self, arr1ev, fig ):
        """Plot 2d spectrum of the detector from input array."""
        if not cp.confpars.cspadSpectrumDetIsOn : return
        print 'plotCSpadDetSpectrum()'       
        self.arr2d = self.getImageArrayForDet( arr1ev )
        self.figDetSpec = fig
        #plt.clf()
        self.drawCSpadDetSpectrum(self.arr2d)


    def drawCSpadDetSpectrum( self, arr2d ):
        if not cp.confpars.cspadSpectrumDetIsOn : return

        try :
            fig = self.figDet
        except AttributeError :
            fig = self.figDetSpec

        if fig.myXmin == None :
            self.arrwin = arr2d
        else :
            self.arrwin = arr2d[fig.myYmin:fig.myYmax,fig.myXmin:fig.myXmax]

        self.str_event = 'Event ' + str(cp.confpars.eventCurrent)
        fig = plt.figure(num=self.figDetSpec.number)
        fig.canvas.set_window_title('CSpad spectrum ' + self.str_event)
        fig.subplots_adjust(left=0.12, bottom=0.03, right=0.98, top=0.97, wspace=0, hspace=0)
        plt.clf()

        plt.hist(self.arrwin.flatten(),
                 bins = cp.confpars.cspadSpectrumNbins,
                 range=(cp.confpars.cspadSpectrumAmin,
                        cp.confpars.cspadSpectrumAmax))

        #print 'drawCSpadDetSpectrum for event ' + str(self.str_event)
        plt.draw() # redraw the current figure


    def plotCSpadQuad08SpectraOf2x1( self, arr1ev, fig ):
        """Amplitude specra from 2d array."""

        if cs.confcspad.isCSPad2x2 : # For 2x2 Mini
            print 'WARNING: plotCSpadQuad08SpectraOf2x1(...) - Spectra are not available for CSPad2x2 ...'
            return

        fig.canvas.set_window_title('CSpad Quad Specra of 2x1')
        plt.clf() # clear plot
        #plt.title('Spectra',color='r',fontsize=20)
        fig.subplots_adjust(left=0.10, bottom=0.05, right=0.98, top=0.95, wspace=0.2, hspace=0.1)

        t_start = time.clock()
        
        #for pair in xrange(8): # loop for pair = 0,1,2,...,7
        for ind in xrange(8): # loop over ind = 0,1,2,...,7
            pair = cs.confcspad.indPairsInQuads[self.quad][ind]
            #print 'quad,ind,pair=', self.quad, ind, pair
            if pair == -1 : continue

            #print 20*'=',' Pair =', pair

            asic1x2  = arr1ev[pair,...]
            #print 'asic1x2.shape =', asic1x2.shape
            arrdimX,arrdimY = asic1x2.shape
            asic1d = asic1x2

            #asic1d.shape=(arrdimX*arrdimY,1)  
            asic1d.resize(arrdimX*arrdimY)            

            plt.subplot(421+ind)
            plt.hist(asic1d, bins=cp.confpars.cspadSpectrumNbins, range=(cp.confpars.cspadSpectrumAmin,cp.confpars.cspadSpectrumAmax))

            xmin, xmax = plt.xlim()
            plt.xticks( arange(int(xmin), int(xmax), int((xmax-xmin)/3)) )
            ymin, ymax = plt.ylim()
            plt.yticks( arange(int(ymin), int(ymax), int((ymax-ymin)/3)) )

            pantit='ASIC ' + str(2*ind) + ', ' + str(2*ind+1)
            ax = plt.gca()
            plt.text(0.04,0.84,pantit,color='r',fontsize=20,transform = ax.transAxes)

            if ind==0 :
                title = 'Event ' + str(cp.confpars.eventCurrent) + '  Quad ' + str(self.quad)
                plt.text(0.8,1.05,title ,color='b',fontsize=24,transform = ax.transAxes)

        print 'Time to generate all histograms (sec) = %f' % (time.clock() - t_start)


    def plotCSpadQuad16Spectra( self, arr1ev, fig ):
        """Amplitude specra from 2d array."""

        if cs.confcspad.isCSPad2x2 : # For 2x2 Mini
            print 'WARNING: plotCSpadQuad16Spectra(...) - Spectra are not available for CSPad2x2 ...'
            return

        fig.canvas.set_window_title('CSpad Quad Specra of 16 ASICs')
        plt.clf() # clear plot
        #plt.title('Spectra',color='r',fontsize=20)
        fig.subplots_adjust(left=0.10, bottom=0.05, right=0.98, top=0.95, wspace=0.35, hspace=0.3)

        t_start = time.clock()
        
        asicN_vs_plot_posN = {0:9, 1:10, 2:13, 3:14, 4:2, 5:6, 6:1, 7:5, 8:8, 9:7, 10:4, 11:3, 12:12, 13:16, 14:11, 15:15}

        #for pair in xrange(8): # loop for pair = 0,1,2,...,7
        for ind in xrange(8): # loop over ind = 0,1,2,...,7
            pair = cs.confcspad.indPairsInQuads[self.quad][ind]
            #print 'quad,ind,pair=', self.quad, ind, pair
            if pair == -1 : continue

            #print 20*'=',' Pair =', pair
            asic1x2  = arr1ev[pair,...]
            #print 'asic1x2.shape =', asic1x2.shape
            arrdimX,arrdimY = asic1x2.shape
            asic1d = asic1x2
            #print 'asic1d.shape =', asic1d.shape
            #asic1d.shape=(arrdimX*arrdimY,1)  
            asic1d.resize(arrdimX*arrdimY)            
            
            asics=hsplit(asic1d,2)

            for inpair in xrange(2) :
                asic = asics[inpair]
                #plt.subplot(4,4,2*ind+inpair+1)
                plt.subplot(4,4,asicN_vs_plot_posN[2*ind+inpair])

                Amin  = cp.confpars.cspadSpectrumAmin
                Amax  = cp.confpars.cspadSpectrumAmax
                plt.hist(asic, bins=cp.confpars.cspadSpectrumNbins,range=(Amin,Amax))

                xmin, xmax = plt.xlim()
                plt.xticks( arange(int(xmin), int(xmax), int((xmax-xmin)/3)) )
                ymin, ymax = plt.ylim()
                plt.yticks( arange(int(ymin), int(ymax), int((ymax-ymin)/3)) )

                pantit='ASIC ' + str(2*ind+inpair)
                plt.title(pantit,color='r',fontsize=20)

                if ind==2 and inpair==0:
                    title = 'Event ' + str(cp.confpars.eventCurrent) + '  Quad ' + str(self.quad)
                    #ax = plt.gca()
                    plt.text(0.8,1.08,title,color='b',fontsize=24,transform = plt.gca().transAxes)

        print 'Time to generate all histograms (sec) = %f' % (time.clock() - t_start)


    def plotCSpadPairImage( self, arr1ev, fig ):
        """Plot 2d image from input array for a single pair"""

        #print 'plotCSpadPairImage()'       

        self.fig = fig
        fig.canvas.set_window_title('CSpad image')
        plt.clf() # clear plot
        fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)        
        arrgap=zeros( (185,4) ) # make additional 2D-array of 0-s for the gap between two 1x1 pads
        
        ind=cp.confpars.cspadPair
        self.quad = cp.confpars.cspadQuad
        self.pair = cs.confcspad.indPairsInQuads[self.quad][ind]
        #print 'pair=', self.pair
        if self.pair == -1 : 
            print 'quad,ind,pair=', self.quad, ind, self.pair
            print 'This pair of ASICs is currently unavailable, see configuration'
            return
        
        #For Image 
        if cs.confcspad.isCSPad2x2 : # For 2x2 Mini
            if cp.confpars.cspadPair > 0 : self.pair = 1
            else                         : self.pair = 0            
            self.arr = self.getImageArrayForMiniElementPair(arr1ev,self.pair)
        else : 
            asic1x2  = arr1ev[self.pair,...]
            asics    = hsplit(asic1x2,2)
            self.arr = hstack((asics[0],arrgap,asics[1]))

        #For spectrum
        #arrdimX,arrdimY = asic1x2.shape
        #self.asic1d = asic1x2
        #self.asic1d.resize(arrdimX*arrdimY)            

        self.asic1d = self.arr.flatten()


        self.pantit =    'Event '   + str(cp.confpars.eventCurrent) 
        self.pantit += ( '   Quad ' + str(cp.confpars.cspadQuad) )
        self.pantit += ( '   Pair ' + str(cp.confpars.cspadPair) )          
        self.pantit += ( '   ASIC ' + str(2*cp.confpars.cspadPair) + ', ' + str(2*cp.confpars.cspadPair+1) )
        
        self.drawCSpadPairImage(cp.confpars.cspadImageAmin,cp.confpars.cspadImageAmax)



    def drawCSpadPairImage(self, Amin=None, Amax=None):
        """Plot 2d image from input array for a single pair"""

        plt.subplot(212)
        plt.hist(self.asic1d, bins=cp.confpars.cspadSpectrumNbins, range=(Amin, Amax))
        #plt.xticks( arange(int(Amin), int(Amax), int((Amax-Amin)/3)) )
        colmin, colmax = plt.xlim()
        coltickslocs, coltickslabels = plt.xticks()
        #print 'colticks =', coltickslocs, coltickslabels
        
        plt.subplot(211)
        self.axes = plt.imshow(self.arr, interpolation='nearest') # Just a histogram, origin='down'
        plt.title(self.pantit,color='r',fontsize=20) # pars like in class Text

        #plt.text(50, -20, pantit, fontsize=24)
        self.colb = plt.colorbar(self.axes, pad=0.10, orientation=2, fraction=0.10, shrink = 1, ticks=coltickslocs)

        plt.clim(colmin,colmax)
        #self.orglims = self.axes.get_clim()
           
        self.fig.canvas.mpl_connect('button_press_event', self.processMousButtonClick)

    def processMousButtonClick(self, event) :
       #print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y,
       #print ' xdata=',event.xdata,' ydata=', event.ydata
       if event.inaxes :
           lims = self.axes.get_clim()

           colmin = lims[0]
           colmax = lims[1]
           range = colmax - colmin
           value = colmin + event.xdata * range
           #print colmin, colmax, range, value

           # left button
           if event.button is 1 :
               if value > colmin and value < colmax :
                   colmin = value
                   print "new mininum: ", colmin
               else :
                   print "min has not been changed (click inside the color bar to change the range)"

           # middle button
           elif event.button is 2 :
               colmin, colmax = cp.confpars.cspadImageAmin, cp.confpars.cspadImageAmax
               print "reset"

           # right button
           elif event.button is 3 :
               if value > colmin and value < colmax :
                   colmax = value
                   print "new maximum: ", colmax
               else :
                   print "max has not been changed (click inside the color bar to change the range)"

           plt.clim(colmin,colmax)
           plt.clf()
           self.drawCSpadPairImage(colmin,colmax)
           plt.draw() # redraw the current figure



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
