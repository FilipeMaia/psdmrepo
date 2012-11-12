#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgSpeWidget...
#
#------------------------------------------------------------------------

"""Plots image and spectrum for 2d numpy array.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 

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

#import matplotlib
#matplotlib.use('GTKAgg') # forse Agg rendering to a GTK canvas (backend)
#matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)

#from   matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from   matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from PyQt4 import QtGui, QtCore

#---------------------
#  Class definition --
#---------------------

class ImgSpeWidget (QtGui.QWidget) :
    """Plots image and spectrum for 2d numpy array."""

    def __init__(self, parent=None, arr=None):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('Matplotlib image embadded in Qt widget')
        self.arr = arr        
        self.fig = plt.figure(num=1, figsize=(5,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)
       #self.fig = Figure(          figsize=(5,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)

        #-----------------------------------
        #self.canvas = FigureCanvas(self.fig)
        self.canvas = self.fig.canvas
        vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        vbox.addWidget(self.canvas)        # <=== Add figure 
        self.setLayout(vbox)
        #-----------------------------------
        self.initParameters()
        if self.arr != None : self.on_draw()


    def initParameters(self):
        self.setFrame()
        self.fig.myXmin = None
        self.fig.myXmax = None
        self.fig.myYmin = None
        self.fig.myYmax = None
        self.fig.myZmin = None
        self.fig.myZmax = None
        self.fig.myNBins = 100
        self.fig.myGridIsOn  = False
        self.fig.myLogIsOn   = False
        self.fig.myZoomIsOn  = False
        self.fig.ntbZoomIsOn = False

        self.xpress = 0
        self.ypress = 0
        self.xpressabs = 0
        self.ypressabs = 0


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def getCanvas(self):
        return self.canvas


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        #print 'ImgSpeWidget: closeEvent'
        pass


    def set_image_array(self,arr):
        self.arr = arr
        self.processDraw()


    def processDraw(self) :
        #fig = event.canvas.figure
        fig = self.fig
        self.on_draw(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax, fig.myZmin, fig.myZmax)


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

            #print 'xmin, xmax, ymin, ymax =', xmin, xmax, ymin, ymax
            self.arrwin =  self.arr[ymin:ymax,xmin:xmax]
            self.range  = [xmin, xmax, ymax, ymin]

        self.arr2d = self.arrwin

        #if self.fig.myLogIsOn : self.arr2d = np.log(self.arrwin)
        #else :                  self.arr2d =        self.arrwin

        self.fig.clear()        

        gs = gridspec.GridSpec(20, 20)
        zmin = self.intOrNone(zmin)
        zmax = self.intOrNone(zmax)

        h1Range = (zmin,zmax)
        if zmin==None and zmax==None : h1Range = None

        #print 'h1Range = ', h1Range

        #self.fig.myaxesH = self.fig.add_subplot(212)
        self.fig.myaxesH = self.fig.add_subplot(gs[15:19,:])
        
        self.fig.myaxesH.hist(self.arrwin.flatten(), bins=self.fig.myNBins, range=h1Range)#, range=(Amin, Amax)) 
        self.fig.myaxesH.grid(self.fig.myGridIsOn)
        Nmin, Nmax = self.fig.myaxesH.get_ylim() 
        yticks = np.arange(Nmin, Nmax, int((Nmax-Nmin)/4))
        if len(yticks)<2 : yticks = [Nmin,Nmax]
        self.fig.myaxesH.set_yticks( yticks )

        zmin,zmax      = self.fig.myaxesH.get_xlim() 
        coltickslocs   = self.fig.myaxesH.get_xticks()
        self.fig.myaxesH.set_xticklabels('')
        #coltickslabels = self.fig.myaxesH.get_xticklabels()
        #print 'colticks =', coltickslocs#, coltickslabels
 
        self.fig.myZmin, self.fig.myZmax = zmin, zmax
        #self.setEditFieldValues()

        self.fig.myaxesI = self.fig.add_subplot(gs[0:14,:])
        self.fig.myaxesI.grid(self.fig.myGridIsOn)
        self.fig.myaxesImage = self.fig.myaxesI.imshow(self.arr2d, origin='upper', interpolation='nearest', extent=self.range, aspect='auto')

        self.fig.myaxesImage.set_clim(zmin,zmax)
        #self.fig.mycolbar = self.fig.colorbar(self.fig.myaxesImage, fraction=0.15, pad=0.1, shrink=1.0, aspect=15, orientation=1, ticks=coltickslocs) #orientation=1,
        self.fig.myaxesC = self.fig.add_subplot(gs[19,:])
        self.fig.mycolbar = self.fig.colorbar(self.fig.myaxesImage, cax=self.fig.myaxesC, orientation=1, ticks=coltickslocs) #orientation=1,
        #self.fig.mycolbar.set_clim(zmin,zmax)

        self.canvas.mpl_connect('button_press_event',   self.processMouseButtonPress) 
        self.canvas.mpl_connect('button_release_event', self.processMouseButtonRelease) 
        self.canvas.mpl_connect('motion_notify_event',  self.processMouseMotion)

        self.canvas.draw()
        #print 'End of on_draw'

    def processMouseMotion(self, event) :

        fig = self.fig

        if fig.ntbZoomIsOn : return

        if event.inaxes == fig.myaxesI and fig.myZoomIsOn :
            #print 'processMouseMotion',
            ##print 'event.xdata, event.ydata =', event.xdata, event.ydata
            #print 'event.x, event.y =', event.x, event.y
            height = self.canvas.figure.bbox.height
            self.xmotion    = event.xdata
            self.ymotion    = event.ydata
            x0 = self.xpressabs
            x1 = event.x
            y0 = height - self.ypressabs
            y1 = height - event.y
            w  = x1 - x0
            h  = y1 - y0
            rect = [x0, y0, w, h]
            self.fig.canvas.drawRectangle( rect )            

    def processMouseButtonPress(self, event) :
        #print 'MouseButtonPress'
        self.fig = event.canvas.figure

        if event.inaxes == self.fig.mycolbar.ax : self.mousePressOnColorBar (event)
        if event.inaxes == self.fig.myaxesI     : self.mousePressOnImage    (event)
        if event.inaxes == self.fig.myaxesH     : self.mousePressOnHistogram(event)


    def mousePressOnImage(self, event) :
        if event.inaxes == self.fig.myaxesI :
           #print 'PressOnImage'
           #print 'event.xdata, event.ydata =', event.xdata, event.ydata
            self.xpress    = event.xdata
            self.ypress    = event.ydata
            self.xpressabs = event.x
            self.ypressabs = event.y
            self.fig.myZoomIsOn = True

    def mousePressOnHistogram(self, event) :
        #print 'PressOnHistogram'
        lims = self.fig.myaxesH.get_xlim()
        self.setColorLimits(event, lims[0], lims[1], event.xdata)


    def mousePressOnColorBar(self, event) :
        #print 'PressOnColorBar'
        lims = self.fig.myaxesImage.get_clim()
        colmin = lims[0]
        colmax = lims[1]
        range = colmax - colmin
        value = colmin + event.xdata * range
        self.setColorLimits(event, colmin, colmax, value)


    def setColorLimits(self, event, colmin, colmax, value) :

        #print colmin, colmax, value

        # left button
        if event.button is 1 :
            if value > colmin and value < colmax :
                colmin = value
                #print "New mininum: ", colmin

        # middle button
        elif event.button is 2 :
            #colmin, colmax = self.getImageAmpLimitsFromWindowParameters()
            #print 'Reset fig' # ,fig.number #, fig.nwin 
            colmin = None
            colmax = None

        # right button
        elif event.button is 3 :
            if value > colmin and value < colmax :
                colmax = value
                #print "New maximum: ", colmax

        self.fig.myZmin = colmin
        self.fig.myZmax = colmax

        self.processDraw()


    def processMouseButtonRelease(self, event) :
        #print 'MouseButtonRelease'

        self.fig = fig = event.canvas.figure # or plt.gcf()
        #figNum  = fig.number 
        #axes     = event.inaxes # fig.gca() 
                
        if event.inaxes == fig.myaxesI and event.button == 1 and fig.myZoomIsOn :

            self.xrelease = event.xdata
            self.yrelease = event.ydata
            if   abs(self.xrelease-self.xpress)<2 : pass
            elif abs(self.yrelease-self.ypress)<2 : pass
            else :
                fig.myXmin = int(min(self.xpress, self.xrelease))
                fig.myXmax = int(max(self.xpress, self.xrelease))  
                fig.myYmin = int(min(self.ypress, self.yrelease))
                fig.myYmax = int(max(self.ypress, self.yrelease))

                #print ' Xmin, Xmax, Ymin, Ymax =', fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax
                #self.on_draw(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax, fig.myZmin, fig.myZmax)
                self.processDraw()

        elif event.button == 2 : # middle or right button
            if event.inaxes == fig.myaxesI : 
                fig.myXmin = None
                fig.myXmax = None
                fig.myYmin = None
                fig.myYmax = None
                self.processDraw()
                #self.on_draw()

            elif event.inaxes == fig.myaxesH or event.inaxes == self.fig.mycolbar.ax :
                fig.myZmin = None
                fig.myZmax = None
                self.processDraw()
                #self.on_draw()

            #plt.draw() # redraw the current figure
        fig.myZoomIsOn = False

        #self.setEditFieldValues()


    def stringOrNone(self,value):
        if value == None : return 'None'
        else             : return str(value)


    def intOrNone(self,value):
        if value == None : return None
        else             : return int(value)


#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)
    return arr

#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    #w = ImgSpeWidget(None, arr)
    w = ImgSpeWidget(None)
    w.move(QtCore.QPoint(50,50))
    w.set_image_array( get_array2d_for_test() )
    w.show()    

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
