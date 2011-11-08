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
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

#--------------------------------
#  Imports for test ONLY
#--------------------------------

#import random
import numpy as np

#import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from PyQt4 import QtGui, QtCore

#---------------------
#  Class definition --
#---------------------

class MyNavigationToolbar ( NavigationToolbar ) :
    """Get full control on navigation toolbar buttons"""

    def __init__(self, canvas, parent=None):
        #print 'MyNavigationToolbar.__init__'
        NavigationToolbar.__init__( self, canvas, parent )

        self.parentWidget = parent
        #self.canvas = canvas

    def home(self, *args) :
        print 'Home is clicked'
        #fig = self.canvas.figure
        #fig.myXmin = None
        #fig.myXmax = None
        #fig.myYmin = None
        #fig.myYmax = None
        NavigationToolbar.home(self)
        #self.parentWidget.on_draw()


    def zoom(self, *args) :
        print 'Zoom is clicked'
        NavigationToolbar.zoom(self)
        #fig = self.canvas.figure
        #fig.myZoomIsOn = True
        #if fig.myZoomIsOn : fig.myZoomIsOn = False
        #else              : fig.myZoomIsOn = True

        #self.canvas.draw()
        #axes = fig.gca() 
        #bounds = axes.viewLim.bounds
        #fig.myXmin = bounds[0]
        #fig.myXmax = bounds[0] + bounds[2] 
        #fig.myYmin = bounds[1] + bounds[3]
        #fig.myYmax = bounds[1]
        #print 'zoom: Xmin, Xmax, Ymin, Ymax =', fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax


    def back(self, *args) :
        print 'Back is clicked'
        NavigationToolbar.back(self)


    def forward(self, *args):
        print 'Forward is clicked'
        NavigationToolbar.forward(self)


    def pan(self,*args):
        print 'Pan is clicked'
        NavigationToolbar.pan(self)


    def edit_parameters(self):
        print 'Edit parameters'
        NavigationToolbar.edit_parameters(self)


    def configure_subplots(self):
        print 'Configure subplots'
        NavigationToolbar.configure_subplots(self)


    def save_figure(self, *args):
        print 'Save is clicked'
        NavigationToolbar.save_figure(self)


#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    arr = np.arange(2400)
    arr.shape = (40,60)

    fig = Figure((5.0, 10.0), dpi=100, facecolor='w',edgecolor='w',frameon=True)
    axes   = fig.add_subplot(111)
    imAxes = axes.imshow(arr, origin='upper', interpolation='nearest', aspect='auto') # extent=range
    canvas = FigureCanvas(fig)
    mytb   = MyNavigationToolbar(canvas, None)

    vbox = QtGui.QVBoxLayout()
    vbox.addWidget(canvas)
    vbox.addWidget(mytb)

    w = QtGui.QWidget()
    w.setLayout(vbox)
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
 
#-----------------------------

if __name__ == "__main__" :
    main()
    sys.exit ('Exit test') 
    #sys.exit ( "Module is not supposed to be run as main module..." )

#-----------------------------
