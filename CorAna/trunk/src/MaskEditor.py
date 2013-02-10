#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module MaskEditor...
#
#------------------------------------------------------------------------

"""Mask editor for 2d array.

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
#import matplotlib.pyplot as plt


#from matplotlib.figure import Figure
from PyQt4 import QtGui, QtCore
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
#-----------------------------
# Imports for other modules --
#-----------------------------

import PlotImgSpeWidget         as imgwidg
import PlotImgSpeButtons        as imgbuts
import MaskEditorButtons        as mebuts

from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------

#class MaskEditor (QtGui.QMainWindow) :
class MaskEditor (QtGui.QWidget) :
    """Mask editor for 2d array"""


    def __init__(self, parent=None, arr=None, ofname='./fig.png', title='Mask editor for 2d array'):
        #QtGui.QMainWindow.__init__(self, parent)
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(20, 40, 500, 550)
        self.setWindowTitle(title)
        self.setFrame()

        self.widgimage   = imgwidg.PlotImgSpeWidget(parent, arr)
        self.widgbuts    = imgbuts.PlotImgSpeButtons(self, self.widgimage, ofname)
        self.widgmebuts  = mebuts .MaskEditorButtons(self, self.widgimage, ofname)
 
        #---------------------

        hbox = QtGui.QHBoxLayout()      
        hbox.addWidget(self.widgmebuts)
        hbox.addWidget(self.widgimage.getCanvas())
        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.widgbuts)
        self.setLayout(vbox)

        #self.show()
        #---------------------
        #self.main_frame = QtGui.QWidget()
        #self.main_frame.setLayout(vbox)
        #self.setCentralWidget(self.main_frame)
        #---------------------


    def set_image_array(self,arr,title='Mask editor for 2d array'):
        self.widgimage.set_image_array(arr)
        self.setWindowTitle(title)


    def set_image_array_new(self,arr,title='Mask editor for 2d array'):
        self.widgimage.set_image_array_new(arr)
        self.setWindowTitle(title)


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

        try    : self.widgimage.close()
        except : pass

        try    : self.widgbuts.close()
        except : pass

        try    : self.widgmebuts.close()
        except : pass

        #try    : del cp.plotimgspe # suicide... of object #1
        #except : pass

        #try    : del cp.plotimgspe_g # suicide... of object #2
        #except : pass

        #print 'Close application'


#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    rows, cols = 1300, 1340
    arr = mu + sigma*np.random.standard_normal(size=rows*cols)
    #arr = 100*np.random.standard_exponential(size=2400)
    #arr = np.arange(2400)
    arr.shape = (rows,cols)
    return arr


def main():

    app = QtGui.QApplication(sys.argv)

    w = MaskEditor(None, get_array2d_for_test())
    #w = MaskEditor(None)
    #w.set_image_array( get_array2d_for_test() )
    w.move(QtCore.QPoint(300,10))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
