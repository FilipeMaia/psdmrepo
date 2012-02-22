#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgExplorer...
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
matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
import matplotlib.pyplot as plt


#from matplotlib.figure import Figure
from PyQt4 import QtGui, QtCore
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
#-----------------------------
# Imports for other modules --
#-----------------------------

import ImgFigureManager     as imgfm
import ImgWidget            as imgwidg
import ImgGUI               as imggui
import ImgControl           as ic
#import MyNavigationToolbar  as mytb

#---------------------
#  Class definition --
#---------------------

#class ImgExplorer (QtGui.QMainWindow) :
class ImgExplorer (QtGui.QWidget, ic.ImgControl) :
    """Plots for any 'image' record in the EventeDisplay project."""


    def __init__(self, parent=None, arr=None):
        #QtGui.QMainWindow.__init__(self, parent)
        QtGui.QWidget.__init__(self, parent)
        ic.ImgControl.__init__(self)

        #self.icp.control = self # is set in ImgControl

        self.setGeometry(10, 10, 880, 1000) 
        self.setWindowTitle('GUI with plot')
        self.setFrame()

        #self.fig = Figure((5, 10), dpi=100, facecolor='w',edgecolor='w',frameon=True)
        #self.fig = plt.figure(num=None, figsize=(5,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)
        self.fig  = imgfm.ifm.get_figure(figsize=(10,10), type='maxspace', icp=self.icp)
        self.wimg = imgwidg.ImgWidget(self.icp, self.fig, arr)
        self.wgui = imggui.ImgGUI(self.icp)

        # Create the navigation toolbar, tied to the canvas
        #self.mpl_toolbar = mytb.MyNavigationToolbar(self.wimg.getCanvas(), self)
 
        #---------------------

        vbox = QtGui.QVBoxLayout()                 # <=== Begin to combine layout 
        #vbox.addWidget(self.wimg)                 # <=== Add figure as QWidget
        vbox.addWidget(self.wimg.getCanvas())      # <=== Add figure as FigureCanvas (saves useful space)
        #vbox.addWidget(self.mpl_toolbar)           # <=== Add toolbar
        vbox.addWidget(self.wgui)                  # <=== Add gui         
        self.setLayout(vbox)

        #---------------------
        #self.main_frame = QtGui.QWidget()
        #self.main_frame.setLayout(vbox)
        #self.setCentralWidget(self.main_frame)
        #---------------------


    def set_image_array(self,arr):
        self.wimg.set_image_array(arr)


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def getPosition(self):
        self.position = (self.pos().x(), self.pos().y())
        print 'self.position=', self.position #WORKS
        return self.position


    def getSize(self):
        self.width = self.size().width()        
        self.height= self.size().height()
        print 'self.size=', self.width, self.height # WORKS
        return (self.width, self.height)


    def moveEvent(self, e):
        #print 'moveEvent' 
        #self.getPosition()
        pass


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        #self.getSize()
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        print 'Close application'
        self.signal_quit()
        #self.wimg  .close()

#-----------------------------
# For test

    def get_array2d_for_test(self) :
        return getRandomWithRing2DArray()

#-----------------------------
# Test
#-----------------------------

def getRandom2DArray(shape=(100,200), mu=200, sigma=25) :
    arr = mu + sigma*np.random.standard_normal(size=shape)
    #arr.shape = shape
    return arr

#-----------------------------

def getUniform2DArray(shape=(100,200), value=0, dtype=np.float) :
    if value == 0 : return np.zeros(shape,dtype)
    else          : return value * np.ones(shape,dtype)

#-----------------------------

def getSmouth2DArray(shape=(100,200), dtype=None) :
    sizex, sizey = shape
    arr = np.arange(sizex*sizey, dtype)
    arr.shape = shape
    return arr

#-----------------------------

def getRing2DArray(shape=(200,200), ring=(1,50,5), center=None, dtype=np.float, modulation=None) :

    arr = np.zeros(shape, dtype) # make empty array
    a1, r1, s1 = ring            # get amplitude, radius, and sigma of the ring

    sizex, sizey = shape
    if center == None : x0, y0 = sizex/2, sizey/2
    else              : x0, y0 = center

    x = np.arange(0,sizex,1,dtype) # np.linspace(0,200,201)
    y = np.arange(0,sizey,1,dtype) # np.linspace(0,100,101)
    X, Y = np.meshgrid(x-x0, y-y0)        
    R   = np.sqrt(X*X+Y*Y)

    if modulation == None :
        amp = 1
    else :                         # apply modulation along phi
        T   = np.arctan2(Y,X)
        NS, NC = modulation
        amp = np.sin(NS*T) * np.cos(NC*T)
    arr += a1 * amp * amp * gaussian(R, r1, s1)

    return arr

#-----------------------------

def getRandomWithRing2DArray(shape=(500,500)) :
    sizex, sizey = shape
    mycenter = (sizex/2, sizey/2)
    Radius   = min(sizex, sizey)/3
    arr  = getRandom2DArray(shape, mu=10, sigma=3)
    arr += getRing2DArray(shape, ring=(100,Radius,5), center=mycenter, modulation=(9,1))
    return arr

#-----------------------------

import math
def gaussian(r,r0,sigma) :
    factor = 1/ (math.sqrt(2) * sigma)
    rr = factor*(r-r0)
    return np.exp(-rr*rr)

#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    #w  = ImgExplorer(None, arr)
    w  = ImgExplorer(None)
    w.move(QtCore.QPoint(10,10))
    w.set_image_array( getRandomWithRing2DArray() )
    #w.set_image_array( getRandom2DArray(shape=(500,500)) )
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
