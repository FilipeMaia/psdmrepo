#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotsForImage...
#
#------------------------------------------------------------------------

"""Plots for any 'image' record in the EventeDisplay project.

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
import os
import sys
from numpy import *

import matplotlib
#matplotlib.use('GTKAgg') # forse Agg rendering to a GTK canvas (backend)
#matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg       as FigureCanvas
#from matplotlib.backend_bases           import NavigationToolbar2      as NavigationToolbar2
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar2

from PyQt4 import QtCore, QtGui

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp
import PrintHDF5        as printh5

#---------------------
#  Class definition --
#---------------------


# THIS STUFF DOES NOT RE-IMPLEMENT NECESSARY METHODS !!!

class MyNavigationToolbar ( NavigationToolbar2 ) :
    """ Need to re-emplement a few methods in order to get control on toolbar button click"""

    def __init__(self, canvas):
        print 'MyNavigationToolbar.__init__'
        #self.canvas = canvas
        #self.coordinates = True
        ##QtGui.QToolBar.__init__( self, parent )
        #NavigationToolbar2.__init__( self, canvas, None )
        self = canvas.toolbar

    def press(self, event):
        print 'my press'

    def press_zoom(self, event):
        print 'zoom is clicked'
 
    def home(self, *args) :
        print 'Home is clicked'
        NavigationToolbar2.home()
  
    def zoom(self, *args) :
        print 'Zoom is clicked'
        NavigationToolbar2.zoom()

#---------------------
#---------------------

class PlotsForImage ( object ) :
    """Plots for any 'image' record in the EventeDisplay project.

    @see BaseClass
    @see OtherClass
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor, initialization"""
        pass

    #-------------------
    #  Public methods --
    #-------------------

    def setWindowParameters(fig) :
        self.fig = fig
        cp.confpars.imageDataset         = cp.confpars.imageWindowParameters[self.fig.nwin][0]
        cp.confpars.imageImageAmin       = cp.confpars.imageWindowParameters[self.fig.nwin][1]
        cp.confpars.imageImageAmax       = cp.confpars.imageWindowParameters[self.fig.nwin][2]
        cp.confpars.imageSpectrumAmin    = cp.confpars.imageWindowParameters[self.fig.nwin][3]
        cp.confpars.imageSpectrumAmax    = cp.confpars.imageWindowParameters[self.fig.nwin][4]
        cp.confpars.imageSpectrumNbins   = cp.confpars.imageWindowParameters[self.fig.nwin][5]
        cp.confpars.imageSpectrumBinWidth= cp.confpars.imageWindowParameters[self.fig.nwin][6]
        cp.confpars.imageImALimsIsOn     = cp.confpars.imageWindowParameters[self.fig.nwin][7]
        cp.confpars.imageSpALimsIsOn     = cp.confpars.imageWindowParameters[self.fig.nwin][8]
        cp.confpars.imageBinWidthIsOn    = cp.confpars.imageWindowParameters[self.fig.nwin][9]

  
    def plotImage( self, arr2d1ev, fig ):
        """Plot 2d image from input array."""

        self.fig = fig
        self.fig.myarr = arr2d1ev
        self.fig.canvas.set_window_title(cp.confpars.current_item_name_for_title)
        
        self.drawImage(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax)


    def drawImage( self, xmin=None, xmax=None, ymin=None, ymax=None ):
        plt.clf() # clear plot
        self.fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
        plt.title('Event ' + str(cp.confpars.eventCurrent),color='r',fontsize=20) # pars like in class Text
        
        if xmin == None :
            self.arrwin  = self.fig.myarr
            self.range   = None # original image range in pixels
        else :
            self.arrwin = self.fig.myarr[ymin:ymax,xmin:xmax]
            self.range  = [xmin, xmax, ymax, ymin]

        self.axescb = plt.imshow(self.arrwin, origin='upper', interpolation='nearest', extent=self.range) # Just a histogram t=0.08s

        self.arr2d = self.arrwin
        self.addSelectionRectangleForImage()

        self.axesDet = plt.gca()
        Amin, Amax = self.getImageAmpLimitsFromWindowParameters()
        #colmin = {True: Amin, False: self.fig.myCmin}[self.fig.myCmin == None]
        #colmax = {True: Amax, False: self.fig.myCmax}[self.fig.myCmax == None]
        #plt.clim(colmin,colmax)
        plt.clim(Amin,Amax)
        self.colb = plt.colorbar(self.axescb, pad=0.05, orientation=1, fraction=0.10, shrink = 1, aspect = 20)#, ticks=coltickslocs #t=0.04s
            
        canvas  = self.fig.canvas
        toolbar = self.fig.canvas.toolbar

        #self.fig.canvas.toolbar.home = self.processHome
        #self.fig.canvas.toolbar.zoom = self.processHome
        
        #print 'XXX =', canvas.window
        #self.fig.canvas.toolbar = MyNavigationToolbar(self.fig.canvas)
        #toolbar = MyNavigationToolbar(self.fig.canvas.toolbar)

        #mytoolbar = MyNavigationToolbar(canvas)

        #toolbar.canvas.mpl_connect('button_press_event',   self.processTestEvent)
        #self.fig.canvas.mpl_connect('button_press_event',   self.processMouseButtonClickForImageColorbar)
        #self.fig.canvas.mpl_connect('button_press_event',   self.processMouseButtonPressForImage)

        #rect_props=dict(edgecolor='black', linewidth=2, linestyle='dashed', fill=False)
        #self.fig.span = RectangleSelector(self.axesDet, self.onRectangleSelect, drawtype='box',rectprops=rect_props)

        self.fig.canvas.mpl_connect('button_release_event',   self.processMouseButtonReleaseForImage)
        plt.draw()


    def processHome(self, *args) :
        print 'Home is clicked'
        #NavigationToolbar2.home()


    def processTestEvent(self, event) :
        print 'TestEvent=', event 


    def processMouseButtonReleaseForImage(self, event) :

        fig = self.fig = event.canvas.figure # or plt.gcf()
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
            self.drawImage()
            #plt.draw() # redraw the current figure
            fig.myZoomIsOn = False


    def onRectangleSelect(self, eclick, erelease) :
        if eclick.button == 1 : # left button

            self.fig = plt.gcf() # Get current figure

            xmin = int(min(eclick.xdata, erelease.xdata))
            ymin = int(min(eclick.ydata, erelease.ydata))
            xmax = int(max(eclick.xdata, erelease.xdata))
            ymax = int(max(eclick.ydata, erelease.ydata))
            print 'xmin, xmax, ymin, ymax: ', xmin, xmax, ymin, ymax

            if xmax-xmin < 20 or ymax-ymin < 20 : return
            self.drawImage( xmin, xmax, ymin, ymax )
            #plt.draw() # redraw the current figure

            self.fig.myXmin = xmin
            self.fig.myXmax = xmax
            self.fig.myYmin = ymin
            self.fig.myYmax = ymax
            self.fig.myZoomIsOn = True



    def processMouseButtonPressForImage(self, event) :
        #print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y,
        #print ' xdata=',event.xdata,' ydata=', event.ydata
        self.fig = event.canvas.figure # plt.gcf() # Get current figure
        print 'mouse click button=', event.button
        if event.button == 2 or event.button == 3 : # middle or right button
            self.fig.myXmin = None
            self.fig.myXmax = None
            self.fig.myYmin = None
            self.fig.myYmax = None
            self.drawImage()
            #plt.draw() # redraw the current figure
            self.fig.myZoomIsOn = False


    def addSelectionRectangleForImage( self ):
        if cp.confpars.selectionIsOn :
            for win in range(cp.confpars.selectionNWindows) :

                if cp.confpars.selectionWindowParameters[win][6] == self.fig.mydsname:
                    #print 'Draw the selection box for dataset:', cp.confpars.selectionWindowParameters[win][6]
                    xy = cp.confpars.selectionWindowParameters[win][2],  cp.confpars.selectionWindowParameters[win][4]
                    w  = cp.confpars.selectionWindowParameters[win][3] - cp.confpars.selectionWindowParameters[win][2]
                    h  = cp.confpars.selectionWindowParameters[win][5] - cp.confpars.selectionWindowParameters[win][4]

                    rec = plt.Rectangle(xy, width=w, height=h, edgecolor='w', linewidth=2, fill=False)
                    plt.gca().add_patch(rec)


    def processMouseButtonClickForImageColorbar(self, event) :
       #print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y,
       #print ' xdata=',event.xdata,' ydata=', event.ydata

       #fig = self.fig = plt.gcf() # Get current figure
       fig = self.fig = event.canvas.figure # or plt.gcf()
        
       if event.inaxes :
           lims = self.axescb.get_clim()

           colmin = lims[0]
           colmax = lims[1]
           range = colmax - colmin
           value = colmin + event.xdata * range
           #print colmin, colmax, range, value

           # left button
           if event.button is 1 :
               if value > colmin and value < colmax :
                   self.fig.myCmin = value
                   print "new mininum: ", self.fig.myCmin
               else :
                   print "min has not been changed (click inside the color bar to change the range)"

           # middle button
           elif event.button is 2 :
               self.fig.myCmin, self.fig.myCmax = self.getImageAmpLimitsFromWindowParameters()
               print "reset"

           # right button
           elif event.button is 3 :
               if value > colmin and value < colmax :
                   self.fig.myCmax = value
                   print "new maximum: ", self.fig.myCmax
               else :
                   print "max has not been changed (click inside the color bar to change the range)"

           self.drawImage(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax)
           plt.draw() # redraw the current figure

#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------


    def plotImageSpectrum( self, arr2d1ev, fig ):
        """Spectrum of amplitudes in the 2d input array."""

        self.fig   = fig
        self.arr2d = arr2d1ev

        plt.clf() # clear plot
        fig.canvas.set_window_title(cp.confpars.current_item_name_for_title) 
        pantit='Specrum, event ' + str(cp.confpars.eventCurrent)
        plt.title(pantit,color='r',fontsize=20) # pars like in class Text
        arrdimX,arrdimY = arr2d1ev.shape
        #print 'arr2d1ev.shape=', arr2d1ev.shape, arrdimX, arrdimY 
        #print 'arr2d1ev=\n', arr2d1ev
        arr1d1ev = copy(arr2d1ev)
        arr1d1ev.resize(arrdimX*arrdimY)
        #print 'arr1d1ev=\n', arr1d1ev
        #plt.hist(arr1d1ev,100)

        ampRange = self.getSpectrumAmpLimitsFromWindowParameters() 
        plt.hist(arr1d1ev, bins=cp.confpars.imageWindowParameters[self.fig.nwin][5], range=ampRange)
        #plt.hist(arr1d1ev)


    def plotImageAndSpectrum( self, arr2d1ev, fig ):
        """Image and spectrum of amplitudes in the 2d input array."""
        #print 'Image and spectrum'

        self.fig = fig
        fig.canvas.set_window_title(cp.confpars.current_item_name_for_title)
        plt.clf() # clear plot
        fig.subplots_adjust(left=0.15, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)        
        
        #For Image 
        self.arr2d = arr2d1ev

        #print 'arr2d1ev.shape=', arr2d1ev.shape
        #print 'self.arr2d.shape=', self.arr2d.shape

        #For spectrum
        arrdimX,arrdimY = self.arr2d.shape
        self.arr1d = copy(arr2d1ev)
        self.arr1d.resize(arrdimX*arrdimY)            

        self.pantit =    'Event ' + str(cp.confpars.eventCurrent) 

        Amin, Amax = self.getImageAmpLimitsFromWindowParameters()
        self.drawImageAndSpectrum(Amin, Amax)


    def drawImageAndSpectrum(self, Amin=None, Amax=None):
        """Plot 2d image from input array for a single pair"""

        ax2 = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
        #plt.subplot(212)
        self.axes1d = plt.hist(self.arr1d, bins=cp.confpars.imageWindowParameters[self.fig.nwin][5], range=(Amin, Amax))
        #plt.xticks( arange(int(Amin), int(Amax), int((Amax-Amin)/3)) )
        colmin, colmax = plt.xlim()
        coltickslocs, coltickslabels = plt.xticks()
        #print 'colticks =', coltickslocs, coltickslabels
        
        ax1 = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=4)
        #plt.subplot(211)
        #print 'self.arr2d.shape=', self.arr2d.shape
        self.axes = plt.imshow(self.arr2d, interpolation='nearest') # Just a histogram, origin='upper', origin='down'
        plt.title(self.pantit,color='r',fontsize=20) # pars like in class Text

        #plt.text(50, -20, pantit, fontsize=24)
        self.colb = plt.colorbar(self.axes, pad=0.10, orientation=2, fraction=0.10, shrink = 1, aspect = 8, ticks=coltickslocs)

        plt.clim(colmin,colmax)
        #self.orglims = self.axes.get_clim()
           
        self.fig.canvas.mpl_connect('button_press_event', self.processMouseButtonClick)


    def processMouseButtonClick(self, event) :
       #print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y,
       #print ' xdata=',event.xdata,' ydata=', event.ydata

       fig = self.fig = event.canvas.figure 

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
               colmin, colmax = self.getImageAmpLimitsFromWindowParameters()
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
           self.drawImageAndSpectrum(colmin,colmax)
           plt.draw() # redraw the current figure


    def getImageAmpLimitsFromWindowParameters(self) :
        if  cp.confpars.imageWindowParameters[self.fig.nwin][7] : # ImALimsIsOn 
            return [cp.confpars.imageWindowParameters[self.fig.nwin][1], cp.confpars.imageWindowParameters[self.fig.nwin][2]]
        else :
            return [self.arr2d.min(),self.arr2d.max()]


    def getSpectrumAmpLimitsFromWindowParameters(self) :
        if  cp.confpars.imageWindowParameters[self.fig.nwin][8] : # ImALimsIsOn 
            return [cp.confpars.imageWindowParameters[self.fig.nwin][3], cp.confpars.imageWindowParameters[self.fig.nwin][4]]
        else :
            return [self.arr2d.min(),self.arr2d.max()]
        
#--------------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#--------------------------------
