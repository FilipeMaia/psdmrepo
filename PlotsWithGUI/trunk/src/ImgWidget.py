#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgWidget...
#
#------------------------------------------------------------------------

"""Plots for any 'image' record in the EventeDisplay project.

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

import matplotlib
#matplotlib.use('GTKAgg') # forse Agg rendering to a GTK canvas (backend)
#matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)

#from   matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from   matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import ImgDrawOnTop   as idont
import ImgDrawOutside as idout

from PyQt4 import QtGui, QtCore
#import ImgConfigParameters as icp
#---------------------
#  Class definition --
#---------------------

class ImgWidget (QtGui.QWidget, idont.ImgDrawOnTop, idout.ImgDrawOutside) :
    """Plots for any 'image' record in the EventeDisplay project."""

    def __init__(self, icp=None, fig=None, arr=None):
        QtGui.QWidget.__init__(self, parent=None)
        idont.ImgDrawOnTop  .__init__(self, icp)
        idout.ImgDrawOutside.__init__(self, icp)
        
        self.setWindowTitle('Matplotlib image embadded in Qt widget')
        self.arr         = arr        
        self.fig         = fig
        self.icp         = icp
        self.icp.wimg    = self

        #-----------------------------------
        #self.canvas = FigureCanvas(self.fig)
        self.canvas = self.fig.canvas
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.canvas)        # Wrap figure canvas in widget 
        self.setLayout(vbox)
        #-----------------------------------

        self.initParameters()


    def initParameters(self):
        self.setFrame()

        self.fig.myXmin = None
        self.fig.myXmax = None
        self.fig.myYmin = None
        self.fig.myYmax = None
        self.fig.myZmin = None
        self.fig.myZmax = None

        self.fig.myZoomIsOn = False
        self.selectRectIsOn = False

        self.mouseLeftIsPrresed = False
        self.mouseMidlIsPrresed = False
        self.mouseRighIsPrresed = False

        self.xpress    = 0
        self.ypress    = 0
        self.xpressabs = 0
        self.ypressabs = 0

        # Initialization of signals for draw_on_top() and draw_outside()
        # This can not be done in ImgControl.__init__ because the figure is initialized later...
        self.get_control().set_signal_info()


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False) # False = SET THE FRAME INVISIBLE !!


    def getCanvas(self):
        return self.canvas


    def get_canvas(self):
        return self.canvas


    def get_control(self):
        return self.icp.control


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())
        pass


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

        self.arr2d = self.arrwin

        zmin = self.intOrNone(zmin)
        zmax = self.intOrNone(zmax)
        #h1Range = (zmin,zmax)
        #if zmin==None or zmax==None :  h1Range = None
        #print 'h1Range = ', h1Range

        self.fig.myZmin, self.fig.myZmax = zmin, zmax

        self.fig.clear()        

        #gs = gridspec.GridSpec(20, 20)
        #self.fig.myaxesSIm = self.fig.add_subplot(gs[0:18,:])
        #self.fig.myaxesSIm = self.fig.add_subplot(gs[:,:])
        self.fig.myaxesSIm = self.fig.add_subplot(111)

        self.grid_onoff(self.icp.gridIsOn)

        self.fig.myaxesImg = self.fig.myaxesSIm.imshow(self.arr2d, origin='upper', interpolation='nearest', extent=self.range, aspect='auto')
        self.fig.myaxesImg.set_clim(zmin,zmax)

        # pad=0.005, fraction=0.10, aspect=12, shrink=1, orientation=2, aspect = 8, ticks=coltickslocs
        self.fig.mycolbar  = self.fig.colorbar(self.fig.myaxesImg, pad=0.004, fraction=0.1, aspect=40)

        #self.fig.myaxesSCB = self.fig.add_subplot(gs[19,:])
        #self.fig.mycolbar  = self.fig.colorbar(self.fig.myaxesImg, cax=self.fig.myaxesSCB, orientation=1)#, ticks=coltickslocs) 
        #self.fig.mycolbar.set_clim(zmin,zmax)

        self.connectImgWithMouse()

        self.draw_on_top()
        self.draw_outside()
        self.canvas.draw()


    def grid_onoff(self, status=False):
        print 'Set grid status =', status
        self.fig.myaxesSIm.grid(status)        
        self.canvas.draw()


    def connectImgWithMouse(self):
        'connect to all the events we need'        
        self.cidpress   = self.canvas.mpl_connect('button_press_event',   self.processMouseButtonPress) 
        self.cidrelease = self.canvas.mpl_connect('button_release_event', self.processMouseButtonRelease) 
        self.cidmotion  = self.canvas.mpl_connect('motion_notify_event',  self.processMouseMotion)


    def disconnectImgFromMouse(self):
        'disconnect all the stored connection ids'
        self.canvas.mpl_disconnect(self.cidpress) 
        self.canvas.mpl_disconnect(self.cidrelease) 
        self.canvas.mpl_disconnect(self.cidmotion)
 

    def processMouseMotion(self, event) :
        fig = self.fig
        if event.inaxes == fig.myaxesSIm :
            #print 'processMouseMotion',
            #print 'event.xdata, event.ydata =', event.xdata, event.ydata
            #print 'event.x, event.y =', event.x, event.y

            #if self.selectRectIsOn and self.mouseLBottonIsPrresed :            

                #self.addRectangle(event) ### ?????????????????????
            pass


    def processDraw(self) :
        #fig = event.canvas.figure
        fig = self.fig
        self.on_draw(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax, fig.myZmin, fig.myZmax)


    def processMouseButtonPress(self, event) :
        print 'ImgWidget: MouseButtonPress'
        self.fig = event.canvas.figure
        if event.button is 1 : self.mouseLeftIsPrresed = True
        if event.button is 2 : self.mouseMidlIsPrresed = True
        if event.button is 3 : self.mouseRighIsPrresed = True

        if event.inaxes == self.fig.mycolbar.ax : self.mousePressOnColorBar (event)
        if event.inaxes == self.fig.myaxesSIm   : self.mousePressOnImage    (event)


    def processMouseButtonRelease(self, event) :
        print 'ImgWidget: MouseButtonRelease'
        self.fig = event.canvas.figure
        self.mouseLeftIsPrresed = False
        self.mouseMidlIsPrresed = False
        self.mouseRighIsPrresed = False
                
        if event.inaxes == self.fig.mycolbar.ax : pass # self.mouseReleaseOnColorBar (event)
        if event.inaxes == self.fig.myaxesSIm   : self.mouseReleaseOnImage  (event)


    def mousePressOnImage(self, event) :
        print 'ImgWidget: mousePressOnImage'

        self.xpress    = event.xdata
        self.ypress    = event.ydata
        self.xpressabs = event.x
        self.ypressabs = event.y

        self.get_control().signal_mouse_on_image_press(event) # send signal in control module


    def mouseReleaseOnImage(self, event) :
        print 'ImgWidget: mouseReleaseOnImage'

        self.xrelease = event.xdata
        self.yrelease = event.ydata

        self.get_control().signal_mouse_on_image_release(event) # send signal in control module
        #self.zoomImageOnRelease(event) 


    def mousePressOnColorBar(self, event) :
        print 'ImgWidget: mousePressOnColorBar'

        lims = self.fig.myaxesImg.get_clim()
        colmin = lims[0]
        colmax = lims[1]
        range = colmax - colmin
        if self.fig.mycolbar.orientation == 'vertical' : value = colmin + event.ydata * range
        else                                           : value = colmin + event.xdata * range         
        self.setColorLimits(event, colmin, colmax, value)


    def setColorLimits(self, event, colmin, colmax, value) :

        print 'ImgWidget: setColorLimits(...) colmin, colmax, value =', colmin, colmax, value

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


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        print 'Close application'
           

    def stringOrNone(self,value):
        if value == None : return 'None'
        else             : return str(value)


    def intOrNone(self,value):
        if value == None : return None
        else             : return int(value)


#---------------------
# CURRENTLY IS NOT USED
    def zoomImageOnRelease(self, event) :
        fig  = self.fig = event.canvas.figure # or plt.gcf()
        #axes = event.inaxes        # fig.gca() 

        if event.button == 1 : # Left button

            if self.selectRectIsOn == True : # THIS SHOULD BE SET FROM EXTERNAL MENU

                fig.myXmin = int(min(self.xpress, self.xrelease))
                fig.myXmax = int(max(self.xpress, self.xrelease))  
                fig.myYmin = int(min(self.ypress, self.yrelease))
                fig.myYmax = int(max(self.ypress, self.yrelease))
                print ' Xmin, Xmax, Ymin, Ymax =', fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax
                self.drawRectangle( fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax )

            if fig.myZoomIsOn :              # THIS SHOULD BE SET FROM EXTERNAL MENU
                self.on_draw(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax, fig.myZmin, fig.myZmax)

        if event.button == 2 : #or event.button == 3 : # middle or right button
            fig.myXmin = None
            fig.myXmax = None
            fig.myYmin = None
            fig.myYmax = None
            fig.myZmin = None
            fig.myZmax = None
            self.on_draw()

#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    print 'ImgWidget: get_array2d_for_test()'
    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)
    return arr

#-----------------------------

def main():

# IN ORDER TO RUN NEEDS IN matplotlib.use('Qt4Agg') UNCOMMENTED ON THE TOP !!!!

    import ImgConfigParameters as gicp
    icp = gicp.giconfpars.addImgConfigPars( None )

    app = QtGui.QApplication(sys.argv)

    #fig = Figure(          figsize=(5,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)
    fig = plt.figure(num=1, figsize=(5,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)

    #w = ImgWidget(None, fig, arr)
    w = ImgWidget(icp, fig)
    w.move(QtCore.QPoint(50,50))
    w.set_image_array( get_array2d_for_test() )
    w.show()    

    app.exec_()
        
#-----------------------------
#  For test
#
if __name__ == "__main__" :
    main()

#-----------------------------
