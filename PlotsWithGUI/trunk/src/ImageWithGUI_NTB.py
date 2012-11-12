#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImageWithGUI_NTB...
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
import sys
import os
import random

import numpy as np

import matplotlib
#matplotlib.use('GTKAgg') # forse Agg rendering to a GTK canvas (backend)
#matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PyQt4 import QtGui, QtCore

#---------------------
#  Class definition --
#---------------------

class MyNavigationToolbar ( NavigationToolbar ) :
    """ Need to re-emplement a few methods in order to get control on toolbar button click"""

    #def __init__(self, canvas):
        #print 'MyNavigationToolbar.__init__'
        #self.canvas = canvas
        #self.coordinates = True
        ##QtGui.QToolBar.__init__( self, parent )
        #NavigationToolbar2.__init__( self, canvas, None )
        #self = canvas.toolbar

    def setParentWidget(self, parentWidget) :
        self.parentWidget = parentWidget

    def home(self, *args) :
        print 'Home is clicked'
        fig = self.canvas.figure
        fig.myXmin = None
        fig.myXmax = None
        fig.myYmin = None
        fig.myYmax = None
        NavigationToolbar.home(self)
        self.parentWidget.on_draw()

    def zoom(self, *args) :
        print 'Zoom is clicked'
        NavigationToolbar.zoom(self)
        fig = self.canvas.figure
        #fig.myZoomIsOn = True
        if fig.myZoomIsOn : fig.myZoomIsOn = False
        else              : fig.myZoomIsOn = True


    def pan(self,*args):
        print 'Pan is clicked'
        NavigationToolbar.pan(self)

    def save_figure(self, *args):
        print 'Save is clicked'
        NavigationToolbar.save_figure(self)

#---------------------


class ImageWithGUI_NTB (QtGui.QMainWindow) :
#class ImageWithGUI_NTB (QtGui.QWidget) :
    """Plots for any 'image' record in the EventeDisplay project."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, fig=None, arr=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('PyQt GUI with matplotlib')

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        self.arr = arr        
        self.fig = fig

        self.fig.myXmin = None
        self.fig.myXmax = None
        self.fig.myYmin = None
        self.fig.myYmax = None
        self.fig.myZmin = None
        self.fig.myZmax = None
        self.fig.myNBins = 100
        self.fig.myZoomIsOn = False

        self.create_main_frame()
        self.create_status_bar()

        #self.on_draw()

    #-------------------
    #  Public methods --
    #-------------------

    def create_main_frame(self):
        self.main_frame = QtGui.QWidget()

        #Get canvas for figure
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = MyNavigationToolbar(self.canvas, self.main_frame)
        self.mpl_toolbar.setParentWidget(self)
                
        # Other GUI controls
        self.but_draw  = QtGui.QPushButton("&Draw")
        self.but_quit  = QtGui.QPushButton("&Quit")
        self.cbox_grid = QtGui.QCheckBox("Show &Grid")
        self.cbox_grid.setChecked(False)
        self.cbox_log  = QtGui.QCheckBox("&Log")
        self.cbox_log.setChecked(False)

        #self.cboxXIsOn = QtGui.QCheckBox("X min, max:")
        #self.cboxYIsOn = QtGui.QCheckBox("Y min, max:")
        self.cboxZIsOn = QtGui.QCheckBox("A min, max:")

        #self.editXmin  = QtGui.QLineEdit(self.stringOrNone(self.fig.myXmin))
        #self.editXmax  = QtGui.QLineEdit(self.stringOrNone(self.fig.myXmax))
        #self.editYmin  = QtGui.QLineEdit(self.stringOrNone(self.fig.myYmin))
        #self.editYmax  = QtGui.QLineEdit(self.stringOrNone(self.fig.myYmax))
        self.editZmin  = QtGui.QLineEdit(self.stringOrNone(self.fig.myZmin))
        self.editZmax  = QtGui.QLineEdit(self.stringOrNone(self.fig.myZmax))

        width = 60
        #self.editXmin.setMaximumWidth(width)
        #self.editXmax.setMaximumWidth(width)
        #self.editYmin.setMaximumWidth(width)
        #self.editYmax.setMaximumWidth(width)
        self.editZmin.setMaximumWidth(width)
        self.editZmax.setMaximumWidth(width)

        #self.editXmin.setValidator(QtGui.QIntValidator(0,100000,self))
        #self.editXmax.setValidator(QtGui.QIntValidator(0,100000,self)) 
        #self.editYmin.setValidator(QtGui.QIntValidator(0,100000,self))
        #self.editYmax.setValidator(QtGui.QIntValidator(0,100000,self)) 
        self.editZmin.setValidator(QtGui.QIntValidator(-100000,100000,self))
        self.editZmax.setValidator(QtGui.QIntValidator(-100000,100000,self))
 
        self.connect(self.but_draw,  QtCore.SIGNAL('clicked()'),         self.processDraw)
        self.connect(self.but_quit,  QtCore.SIGNAL('clicked()'),         self.processQuit)
        self.connect(self.cbox_grid, QtCore.SIGNAL('stateChanged(int)'), self.processDraw)
        self.connect(self.cbox_log,  QtCore.SIGNAL('stateChanged(int)'), self.processDraw)

        #self.connect(self.cboxXIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)
        #self.connect(self.cboxYIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)
        self.connect(self.cboxZIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)

        #self.connect(self.editXmin, QtCore.SIGNAL('editingFinished ()'), self.processEditXmin)
        #self.connect(self.editXmax, QtCore.SIGNAL('editingFinished ()'), self.processEditXmax)
        #self.connect(self.editYmin, QtCore.SIGNAL('editingFinished ()'), self.processEditYmin)
        #self.connect(self.editYmax, QtCore.SIGNAL('editingFinished ()'), self.processEditYmax)
        self.connect(self.editZmin, QtCore.SIGNAL('editingFinished ()'), self.processEditZmin)
        self.connect(self.editZmax, QtCore.SIGNAL('editingFinished ()'), self.processEditZmax)

        # Layout with box sizers
        # 
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.but_draw)
        hbox.addWidget(self.cbox_grid)
        hbox.addStretch(1)
        hbox.addWidget(self.but_quit)
        #hbox.setAlignment(w, QtCore.Qt.AlignVCenter)

        #hboxX = QtGui.QHBoxLayout()
        #hboxX.addWidget(self.cboxXIsOn)
        #hboxX.addWidget(self.editXmin)
        #hboxX.addWidget(self.editXmax)
        #hboxX.addStretch(1)

        #hboxY = QtGui.QHBoxLayout()
        #hboxY.addWidget(self.cboxYIsOn)
        #hboxY.addWidget(self.editYmin)
        #hboxY.addWidget(self.editYmax)
        #hboxY.addStretch(1)

        hboxZ = QtGui.QHBoxLayout()
        hboxZ.addWidget(self.cboxZIsOn)
        hboxZ.addWidget(self.editZmin)
        hboxZ.addWidget(self.editZmax)
        hboxZ.addWidget(self.cbox_log)
        hboxZ.addStretch(1)

        vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        vbox.addWidget(self.canvas)        # <=== Add figure 
        vbox.addWidget(self.mpl_toolbar)   # <=== Add toolbar
        
        #vbox.addLayout(hboxX)              # <=== Add buttons etc.
        #vbox.addLayout(hboxY)
        vbox.addLayout(hboxZ)
        vbox.addLayout(hbox)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

        self.setEditFieldColors()    
        self.setEditFieldValues()

    
    def create_status_bar(self):
        self.status_text = QtGui.QLabel("Status bar info is here")
        self.statusBar().addWidget(self.status_text, 1)


    def set_image_array(self,arr):
        self.arr = arr
        self.on_draw(self.fig.myXmin, self.fig.myXmax, self.fig.myYmin, self.fig.myYmax, self.fig.myZmin, self.fig.myZmax)


    def on_draw(self, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
        """Redraws the figure"""

        if xmin == None or xmax == None or ymin == None or ymax == None :
            self.arrwin  = self.arr
            self.range   = None # original image range in pixels
        else :
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)

            print 'xmin, xmax, ymin, ymax =', xmin, xmax, ymin, ymax
            self.arrwin =  self.arr[ymin:ymax,xmin:xmax]
            self.range  = [xmin, xmax, ymax, ymin]


        if self.cbox_log.isChecked() : self.arr2d = np.log(self.arrwin)
        else :                         self.arr2d =        self.arrwin

        self.fig.clear()        

        gs = gridspec.GridSpec(20, 20)
        zmin = self.intOrNone(zmin)
        zmax = self.intOrNone(zmax)

        h1Range = (zmin,zmax)
        if zmin==None and zmax==None : h1Range = None

        print 'h1Range = ', h1Range

        #self.fig.myaxesH = self.fig.add_subplot(212)
        self.fig.myaxesH = self.fig.add_subplot(gs[15:19,:])
        
        self.fig.myaxesH.hist(self.arrwin.flatten(), bins=self.fig.myNBins, range=h1Range)#, range=(Amin, Amax)) 
        Nmin, Nmax = self.fig.myaxesH.get_ylim() 
        yticks = np.arange(Nmin, Nmax, int((Nmax-Nmin)/4))
        if len(yticks)<2 : yticks = [Nmin,Nmax]
        self.fig.myaxesH.set_yticks( yticks )

        zmin,zmax      = self.fig.myaxesH.get_xlim() 
        coltickslocs   = self.fig.myaxesH.get_xticks()
        self.fig.myaxesH.set_xticklabels('')
        #coltickslabels = self.fig.myaxesH.get_xticklabels()
        print 'colticks =', coltickslocs#, coltickslabels
 
        self.fig.myZmin, self.fig.myZmax = zmin, zmax
        self.setEditFieldValues()

        self.fig.myaxesI = self.fig.add_subplot(gs[0:14,:])
        self.fig.myaxesI.grid(self.cbox_grid.isChecked())
        self.fig.myaxesImage = self.fig.myaxesI.imshow(self.arr2d, origin='upper', interpolation='nearest', extent=self.range, aspect='auto')
        self.fig.myaxesImage.set_clim(zmin,zmax)
        #self.fig.mycolbar = self.fig.colorbar(self.fig.myaxesImage, fraction=0.15, pad=0.1, shrink=1.0, aspect=15, orientation=1, ticks=coltickslocs) #orientation=1,
        self.fig.myaxesC = self.fig.add_subplot(gs[19,:])
        self.fig.mycolbar = self.fig.colorbar(self.fig.myaxesImage, cax=self.fig.myaxesC, orientation=1, ticks=coltickslocs) #orientation=1,
        #self.fig.mycolbar.set_clim(zmin,zmax)

        self.fig.canvas.mpl_connect('button_press_event',   self.processMouseButtonPress) 
        self.fig.canvas.mpl_connect('button_release_event', self.processMouseButtonRelease) 
        self.canvas.draw()


    def setEditFieldValues(self) :
        #self.editXmin.setText( str(self.intOrNone(self.fig.myXmin)) )
        #self.editXmax.setText( str(self.intOrNone(self.fig.myXmax)) )

        #self.editYmin.setText( str(self.intOrNone(self.fig.myYmin)) )
        #self.editYmax.setText( str(self.intOrNone(self.fig.myYmax)) ) 

        self.editZmin.setText( str(self.intOrNone(self.fig.myZmin)) )
        self.editZmax.setText( str(self.intOrNone(self.fig.myZmax)) )

        self.setEditFieldColors()

       
    def setEditFieldColors(self) :
        
        #if self.cboxXIsOn.isChecked(): self.styleSheet = self.styleSheetWhite
        #else                         : self.styleSheet = self.styleSheetGrey
        #self.editXmin.setStyleSheet('Text-align:left;' + self.styleSheet)
        #self.editXmax.setStyleSheet('Text-align:left;' + self.styleSheet)

        #if self.cboxYIsOn.isChecked(): self.styleSheet = self.styleSheetWhite
        #else                         : self.styleSheet = self.styleSheetGrey
        #self.editYmin.setStyleSheet('Text-align:left;' + self.styleSheet)
        #self.editYmax.setStyleSheet('Text-align:left;' + self.styleSheet)

        if self.cboxZIsOn.isChecked(): self.styleSheet = self.styleSheetWhite
        else                         : self.styleSheet = self.styleSheetGrey
        self.editZmin.setStyleSheet('Text-align:left;' + self.styleSheet)
        self.editZmax.setStyleSheet('Text-align:left;' + self.styleSheet)

        #self.editXmin.setReadOnly( not self.cboxXIsOn.isChecked() )
        #self.editXmax.setReadOnly( not self.cboxXIsOn.isChecked() )

        #self.editYmin.setReadOnly( not self.cboxYIsOn.isChecked() )
        #self.editYmax.setReadOnly( not self.cboxYIsOn.isChecked() )

        self.editZmin.setReadOnly( not self.cboxZIsOn.isChecked() )
        self.editZmax.setReadOnly( not self.cboxZIsOn.isChecked() )


    def processDraw(self) :
        #fig = event.canvas.figure
        fig = self.fig
        self.on_draw(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax, fig.myZmin, fig.myZmax)


    def processMouseButtonPress(self, event) :
        print 'MouseButtonPress'
        self.fig = event.canvas.figure

        if event.inaxes == self.fig.mycolbar.ax : self.mousePressOnColorBar (event)
        if event.inaxes == self.fig.myaxesI     : self.mousePressOnImage    (event)
        if event.inaxes == self.fig.myaxesH     : self.mousePressOnHistogram(event)


    def mousePressOnImage(self, event) :
        fig = self.fig
        if event.inaxes == fig.myaxesI :
            print 'PressOnImage'
           #print 'event.xdata, event.ydata =', event.xdata, event.ydata
            self.xpress = event.xdata
            self.ypress = event.ydata


    def mousePressOnHistogram(self, event) :
        print 'PressOnHistogram'
        lims = self.fig.myaxesH.get_xlim()
        self.setColorLimits(event, lims[0], lims[1], event.xdata)


    def mousePressOnColorBar(self, event) :
        print 'PressOnColorBar'
        lims = self.fig.myaxesImage.get_clim()
        colmin = lims[0]
        colmax = lims[1]
        range = colmax - colmin
        value = colmin + event.xdata * range
        self.setColorLimits(event, colmin, colmax, value)


    def setColorLimits(self, event, colmin, colmax, value) :

        print colmin, colmax, value

        # left button
        if event.button is 1 :
            if value > colmin and value < colmax :
                colmin = value
                print "New mininum: ", colmin

        # middle button
        elif event.button is 2 :
            #colmin, colmax = self.getImageAmpLimitsFromWindowParameters()
            print 'Reset fig' # ,fig.number #, fig.nwin 
            colmin = None
            colmax = None

        # right button
        elif event.button is 3 :
            if value > colmin and value < colmax :
                colmax = value
                print "New maximum: ", colmax

        self.fig.myZmin = colmin
        self.fig.myZmax = colmax

        self.processDraw()


    def processMouseButtonRelease(self, event) :
        print 'MouseButtonRelease'

        fig         = event.canvas.figure # or plt.gcf()
        #figNum      = fig.number 
        self.fig    = fig
        axes        = event.inaxes # fig.gca() 
                
        if event.inaxes == fig.myaxesI and event.button == 1 : # Left button

            if fig.myZoomIsOn :
                self.xrelease = event.xdata
                self.yrelease = event.ydata
                fig.myXmin = int(min(self.xpress, self.xrelease))
                fig.myXmax = int(max(self.xpress, self.xrelease))  
                fig.myYmin = int(min(self.ypress, self.yrelease))
                fig.myYmax = int(max(self.ypress, self.yrelease))

                print ' Xmin, Xmax, Ymin, Ymax =', fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax
                self.on_draw(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax, fig.myZmin, fig.myZmax)

        if event.button == 2 : #or event.button == 3 : # middle or right button
            fig.myXmin = None
            fig.myXmax = None
            fig.myYmin = None
            fig.myYmax = None
            fig.myZmin = None
            fig.myZmax = None
            self.on_draw()

            #plt.draw() # redraw the current figure
            #fig.myZoomIsOn = False

        self.setEditFieldValues()


    def processCBoxes(self):
        self.setEditFieldColors()


    def stringOrNone(self,value):
        if value == None : return 'None'
        else             : return str(value)


    def intOrNone(self,value):
        if value == None : return None
        else             : return int(value)


    #def processEditXmin(self):
    #    self.fig.myXmin = self.editXmin.displayText()


    #def processEditXmax(self):
    #    self.fig.myXmax = self.editXmax.displayText()


    #def processEditYmin(self):
    #    self.fig.myYmin = self.editYmin.displayText()


    #def processEditYmax(self):
    #    self.fig.myYmax = self.editYmax.displayText()


    def processEditZmin(self):
        self.fig.myZmin = self.editZmin.displayText()


    def processEditZmax(self):
        self.fig.myZmax = self.editZmax.displayText()
 

    def processQuit(self):
        print 'Quit'
        self.close()


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        print 'Close application'
           
#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    fig = Figure((5.0, 10.0), dpi=100, facecolor='w',edgecolor='w',frameon=True)

    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)

    #ex  = ImageWithGUI_NTB(None, fig, arr)
    ex  = ImageWithGUI_NTB(None, fig)
    ex.move(QtCore.QPoint(50,50))
    ex.set_image_array(arr)

    ex.show()
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
