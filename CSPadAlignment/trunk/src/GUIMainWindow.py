#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMainWindow...
#
#------------------------------------------------------------------------

"""Module GUIMainWindow for CSPadAlignment package

CSPadAlignment package is intended to check quality of the CSPad alignment
using image of wires illuminated by flat field.
Shadow of wires are compared with a set of straight lines, which can be
interactively adjusted.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$
#------------------------------
#!/usr/bin/env python
#----------------------------------

#--------------------------------
#  Imports of modules --
#--------------------------------
import sys
import os
import numpy as np

#import matplotlib
#matplotlib.use('GTKAgg') # forse Agg rendering to a GTK canvas (backend)
#matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
from PyQt4 import QtGui, QtCore

from   matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from   matplotlib.figure import Figure
import matplotlib.pyplot   as plt

import GUIVertical         as guiv
import GUIHorizontal       as guih
import MyNavigationToolbar as myntb
import GraphicalWindow     as grawin
import MessageWindow       as meswin
import ImageParameters     as imp
import Draw                as drw


#---------------------
#  Class definition --
#---------------------

class GUIMainWindow (QtGui.QMainWindow) :
    """Plots for any 'image' record in the EventeDisplay project."""

    def __init__(self, parent=None, fig=None, arr=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('CSpad Alignment GUI')
        self.setGeometry(100, 0, 930, 1050)
        
        imp.impars.gmw = self

        self.arr = arr        
        self.fig = fig
        self.canvas = fig.canvas

        self.create_main_frame()
        self.create_status_bar()

        self.graphWindow = grawin.GraphicalWindow(self.fig, self.arr) # Graphic window specific stuff
        #self.draw_graphical_window()


    #-------------------
    #  Public methods --
    #-------------------

    def draw_graphical_window(self):
        self.graphWindow.on_draw()

 
    def create_main_frame(self):
        self.main_frame = QtGui.QWidget()                               # Graphic window specific stuff
        self.canvas.setParent(self.main_frame)                          # Graphic window specific stuff

        self.wMessageWindow = meswin.MessageWindow()
        self.mpl_toolbar    = myntb.MyNavigationToolbar(self.canvas, self.main_frame)
        self.wGUIHorizontal = guih.GUIHorizontal()
        self.wGUIVertical   = guiv.GUIVertical()

        splitterV1 = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitterV1.addWidget(self.mpl_toolbar)
        splitterV1.addWidget(self.canvas)

        splitterH1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitterH1.addWidget(self.wGUIVertical)
        splitterH1.addWidget(splitterV1)

        splitterV2 = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitterV2.addWidget(self.wGUIHorizontal)
        splitterV2.addWidget(splitterH1)

        splitterV3 = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitterV3.addWidget(splitterV2)
        splitterV3.addWidget(self.wMessageWindow)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(splitterV3)
        
        self.main_frame.setLayout(self.vbox)
        self.setCentralWidget(self.main_frame)


    def create_status_bar(self):
        self.status_text = QtGui.QLabel("Status info is here")
        self.statusBar().addWidget(self.status_text, 1)


    def getGraphicalWindow(self):
        return self.graphWindow


    def getMessageWindow(self):
        return self.wMessageWindow


    def getVerticalWindow(self):
        return self.wGUIVertical


    def getHorizontalWindow(self):
        return self.wGUIHorizontal


    def set_image_array(self, arr):
        self.graphWindow.set_image_array(arr)


    def on_quit(self):
        print 'on_quit: Quit'
        self.close() # call closeEvent(...)


    def closeEvent(self, event):
        self.wGUIHorizontal.close()
        self.wGUIVertical.close()
        self.wMessageWindow.close()
        self.canvas.close()
        self.mpl_toolbar.close()

        drw.draw.close_fig() # close all figures
        print 'closeEvent : Quit GUIMainWindow'
          
#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    mu, sigma = 2000, 200
    arr = mu + sigma*np.random.standard_normal(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)

    fig = Figure((10.0, 10.0), dpi=100, facecolor='w',edgecolor='w',frameon=True)
    fig.canvas = FigureCanvas(fig)   # Graphic window specific stuff

    ex  = GUIMainWindow(None, fig, arr)
    #ex.set_image_array(arr)

    ex.show()
    app.exec_()
        
#-----------------------------
if __name__ == "__main__" :
    main()
#-----------------------------
