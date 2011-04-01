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

  
    def plotImage( self, arr2d1ev, fig ):
        """Plot 2d image from input array."""

        self.figDet = fig
        self.figDet.myarr = arr2d1ev
        self.figDet.canvas.set_window_title(cp.confpars.current_item_name_for_title)
        
        self.drawImage(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax)


    def drawImage( self, xmin=None, xmax=None, ymin=None, ymax=None ):
        plt.clf() # clear plot
        self.figDet.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
        plt.title('Event ' + str(cp.confpars.eventCurrent),color='r',fontsize=20) # pars like in class Text
        
        if xmin == None :
            self.arrwin  = self.figDet.myarr
            self.range   = None # original image range in pixels
        else :
            self.arrwin = self.figDet.myarr[ymin:ymax,xmin:xmax]
            self.range  = [xmin, xmax, ymax, ymin]

        self.axescb = plt.imshow(self.arrwin, origin='upper', interpolation='nearest', extent=self.range) # Just a histogram t=0.08s

        self.addSelectionRectangleForImage()

        self.axesDet = plt.gca()

        colmin = {True: cp.confpars.imageImageAmin, False: self.figDet.myCmin}[self.figDet.myCmin == None]
        colmax = {True: cp.confpars.imageImageAmax, False: self.figDet.myCmax}[self.figDet.myCmax == None]
        plt.clim(colmin,colmax)
        self.colb = plt.colorbar(self.axescb, pad=0.05, orientation=1, fraction=0.10, shrink = 1, aspect = 20)#, ticks=coltickslocs #t=0.04s
            
        canvas  = self.figDet.canvas
        toolbar = self.figDet.canvas.toolbar

        #print 'XXX =', canvas.window
        #self.figDet.canvas.toolbar = MyNavigationToolbar(self.figDet.canvas)
        #toolbar = MyNavigationToolbar(self.figDet.canvas.toolbar)

        #mytoolbar = MyNavigationToolbar(canvas)

        #toolbar.canvas.mpl_connect('button_press_event',   self.processTestEvent)
        #self.figDet.canvas.mpl_connect('button_press_event',   self.processMouseButtonClickForImageColorbar)
        #self.figDet.canvas.mpl_connect('button_press_event',   self.processMouseButtonPressForImage)

        #rect_props=dict(edgecolor='black', linewidth=2, linestyle='dashed', fill=False)
        #self.figDet.span = RectangleSelector(self.axesDet, self.onRectangleSelect, drawtype='box',rectprops=rect_props)

        self.figDet.canvas.mpl_connect('button_release_event',   self.processMouseButtonReleaseForImage)
        plt.draw()


    def processTestEvent(self, event) :
        print 'TestEvent=', event 


    def processMouseButtonReleaseForImage(self, event) :

        fig         = event.canvas.figure # or plt.gcf()
        figNum      = fig.number 
        self.figDet = fig
        
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

            self.figDet = plt.gcf() # Get current figure

            xmin = int(min(eclick.xdata, erelease.xdata))
            ymin = int(min(eclick.ydata, erelease.ydata))
            xmax = int(max(eclick.xdata, erelease.xdata))
            ymax = int(max(eclick.ydata, erelease.ydata))
            print 'xmin, xmax, ymin, ymax: ', xmin, xmax, ymin, ymax

            if xmax-xmin < 20 or ymax-ymin < 20 : return
            self.drawImage( xmin, xmax, ymin, ymax )
            #plt.draw() # redraw the current figure

            self.figDet.myXmin = xmin
            self.figDet.myXmax = xmax
            self.figDet.myYmin = ymin
            self.figDet.myYmax = ymax
            self.figDet.myZoomIsOn = True



    def processMouseButtonPressForImage(self, event) :
        #print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y,
        #print ' xdata=',event.xdata,' ydata=', event.ydata
        self.figDet = plt.gcf() # Get current figure
        print 'mouse click button=', event.button
        if event.button == 2 or event.button == 3 : # middle or right button
            self.figDet.myXmin = None
            self.figDet.myXmax = None
            self.figDet.myYmin = None
            self.figDet.myYmax = None
            self.drawImage()
            #plt.draw() # redraw the current figure
            self.figDet.myZoomIsOn = False


    def addSelectionRectangleForImage( self ):
        if cp.confpars.selectionIsOn :
            for win in range(cp.confpars.selectionNWindows) :

                if cp.confpars.selectionWindowParameters[win][6] == self.figDet.mydsname:
                    #print 'Draw the selection box for dataset:', cp.confpars.selectionWindowParameters[win][6]
                    xy = cp.confpars.selectionWindowParameters[win][2],  cp.confpars.selectionWindowParameters[win][4]
                    w  = cp.confpars.selectionWindowParameters[win][3] - cp.confpars.selectionWindowParameters[win][2]
                    h  = cp.confpars.selectionWindowParameters[win][5] - cp.confpars.selectionWindowParameters[win][4]

                    rec = plt.Rectangle(xy, width=w, height=h, edgecolor='w', linewidth=2, fill=False)
                    plt.gca().add_patch(rec)


    def processMouseButtonClickForImageColorbar(self, event) :
       #print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y,
       #print ' xdata=',event.xdata,' ydata=', event.ydata

       fig = self.figDet = plt.gcf() # Get current figure

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
                   self.figDet.myCmin = value
                   print "new mininum: ", self.figDet.myCmin
               else :
                   print "min has not been changed (click inside the color bar to change the range)"

           # middle button
           elif event.button is 2 :
               self.figDet.myCmin, self.figDet.myCmax = cp.confpars.imageImageAmin, cp.confpars.imageImageAmax
               print "reset"

           # right button
           elif event.button is 3 :
               if value > colmin and value < colmax :
                   self.figDet.myCmax = value
                   print "new maximum: ", self.figDet.myCmax
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

        cp.confpars.imageSpectrumRange=(15,45)
        #cp.confpars.imageSpectrumNbins=30       
        #cp.confpars.imageSpectrumRange=None        
        #cp.confpars.imageSpectrumNbins=None        
        plt.hist(arr1d1ev, bins=cp.confpars.imageSpectrumNbins, range=(cp.confpars.imageSpectrumAmin,cp.confpars.imageSpectrumAmax))
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
        self.drawImageAndSpectrum(cp.confpars.imageImageAmin,cp.confpars.imageImageAmax)



    def drawImageAndSpectrum(self, Amin=None, Amax=None):
        """Plot 2d image from input array for a single pair"""

        ax2 = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
        #plt.subplot(212)
        self.axes1d = plt.hist(self.arr1d, bins=cp.confpars.imageSpectrumNbins, range=(Amin, Amax))
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
               colmin, colmax = cp.confpars.imageImageAmin, cp.confpars.imageImageAmax
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
    
        
#--------------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#--------------------------------
