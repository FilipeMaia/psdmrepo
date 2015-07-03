#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotImgSpe...
#
#------------------------------------------------------------------------

"""Plots image and spectrum for 2d array.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import random
import numpy as np

# For self-run debugging:
#if __name__ == "__main__" :
import matplotlib
#matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
if matplotlib.get_backend() != 'Qt4Agg' : matplotlib.use('Qt4Agg')

#import matplotlib.pyplot as plt


#from matplotlib.figure import Figure
from PyQt4 import QtGui, QtCore
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
#-----------------------------
# Imports for other modules --
#-----------------------------

#import ImgSpeNavToolBar     as imgtb
import PlotImgSpeWidget         as imgwidg
import PlotImgSpeButtons        as imgbuts

from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------

#class PlotImgSpe (QtGui.QMainWindow) :
class PlotImgSpe (QtGui.QWidget) :
    """Plots image and spectrum for 2d array"""


    def __init__(self, parent=None, arr=None, ifname='', ofname='./fig.png', title='Plot 2d array', orient=0, y_is_flip=False ):
        #QtGui.QMainWindow.__init__(self, parent)
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(20, 40, 700, 800)
        self.setWindowTitle(title)
        self.setFrame()

        if arr is not None : self.arr = arr
        elif ifname != ''  : self.arr = gu.get_array_from_file(ifname)
        else               : self.arr = get_array2d_for_test()

        self.ext_ref = None

        self.widgimage   = imgwidg.PlotImgSpeWidget(parent, arr, orient, y_is_flip)
        self.widgbuts    = imgbuts.PlotImgSpeButtons(self, self.widgimage, ifname, ofname)
        #self.mpl_toolbar = imgtb.ImgSpeNavToolBar(self.widgimage, self)
 
        #---------------------

        vbox = QtGui.QVBoxLayout()                      # <=== Begin to combine layout 
        #vbox.addWidget(self.widgimage)                 # <=== Add figure as QWidget
        vbox.addWidget(self.widgimage.getCanvas())      # <=== Add figure as FigureCanvas 
        #vbox.addWidget(self.mpl_toolbar)                # <=== Add toolbar
        vbox.addWidget(self.widgbuts)                   # <=== Add buttons         
        self.setLayout(vbox)
        #self.show()
        #---------------------
        #self.main_frame = QtGui.QWidget()
        #self.main_frame.setLayout(vbox)
        #self.setCentralWidget(self.main_frame)
        #---------------------


    def set_image_array(self,arr,title='Plot 2d array'):
        self.widgimage.set_image_array(arr)
        self.setWindowTitle(title)


    def set_image_array_new(self,arr,title='Plot 2d array', orient=0, y_is_flip=False):
        self.widgimage.set_image_array_new(arr, orient, y_is_flip)
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

        #try    : self.mpl_toolbar.close()
        #except : pass

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

    #w  = PlotImgSpe(None, get_array2d_for_test())
    w  = PlotImgSpe(None)
    w.set_image_array( get_array2d_for_test() )
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
