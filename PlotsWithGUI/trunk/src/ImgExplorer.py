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
import MyNavigationToolbar  as mytb
import ImgWidget            as imgwidg
import ImgGUI               as imggui

#---------------------
#  Class definition --
#---------------------

#class ImgExplorer (QtGui.QMainWindow) :
class ImgExplorer (QtGui.QWidget) :
    """Plots for any 'image' record in the EventeDisplay project."""


    def __init__(self, parent=None, arr=None):
        #QtGui.QMainWindow.__init__(self, parent)
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('GUI with plot')
        self.setFrame()

        #self.fig = Figure((5.0, 10.0), dpi=100, facecolor='w',edgecolor='w',frameon=True)
        #self.fig = plt.figure(num=None, figsize=(5,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)
        self.fig       = imgfm.ifm.get_figure(figsize=(5,10), type='type1')
        self.widgimage = imgwidg.ImgWidget(parent, self.fig, arr)
        self.widimggui = imggui.ImgGUI(None)

        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = mytb.MyNavigationToolbar(self.widgimage.getCanvas(), self)
 
        #---------------------

        vbox = QtGui.QVBoxLayout()                      # <=== Begin to combine layout 
        #vbox.addWidget(self.widgimage)                 # <=== Add figure as QWidget
        vbox.addWidget(self.mpl_toolbar)                # <=== Add toolbar
        vbox.addWidget(self.widgimage.getCanvas())      # <=== Add figure as FigureCanvas (saves useful space)
        vbox.addWidget(self.widimggui)                  # <=== Add gui         
        self.setLayout(vbox)

        #---------------------
        #self.main_frame = QtGui.QWidget()
        #self.main_frame.setLayout(vbox)
        #self.setCentralWidget(self.main_frame)
        #---------------------


    def set_image_array(self,arr):
        self.widgimage.set_image_array(arr)


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())
        pass


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        print 'Close application'
        #self.widgimage  .close()

#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)
    return arr


def main():

    app = QtGui.QApplication(sys.argv)

    #w  = ImgExplorer(None, arr)
    w  = ImgExplorer(None)
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
