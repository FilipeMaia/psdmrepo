#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotG2...
#
#------------------------------------------------------------------------

"""Plot for G2 arrays.

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
if __name__ == "__main__" :
    import matplotlib
    matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)

import matplotlib.pyplot as plt

#from matplotlib.figure import Figure
from PyQt4 import QtGui, QtCore
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
#-----------------------------
# Imports for other modules --
#-----------------------------

import PlotG2Widget         as imgwidg
import PlotG2Buttons        as imgbuts

from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------

class PlotG2 (QtGui.QWidget) :
    """Plot for G2 arrays"""

    def __init__(self, parent=None, arrays=None, ofname='./fig_g2.png', title=''):
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(20, 40, 800, 600)
        self.setWindowTitle('Plot for G2 arrays')
        self.setFrame()

        self.widgimage = imgwidg.PlotG2Widget(parent, arrays, title=title)
        self.widgbuts  = imgbuts.PlotG2Buttons(self, self.widgimage, ofname)
 
        #---------------------
        vbox = QtGui.QVBoxLayout()                      # <=== Begin to combine layout 
        #vbox.addWidget(self.widgimage)                 # <=== Add figure as QWidget
        vbox.addWidget(self.widgimage.getCanvas())      # <=== Add figure as FigureCanvas 
        vbox.addWidget(self.widgbuts)                   # <=== Add buttons         
        self.setLayout(vbox)
        #---------------------
        cp.plotg2_is_on = True


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

        cp.plotg2_is_on = False

        try    : del cp.plot_g2 # suicide..., but what to do in order to close it properly?
        except : pass

        #print 'Close application'

    def set_array(self, arr, title='') :
        self.widgimage.set_array(arr, title)

#-----------------------------
# Test
#-----------------------------

def get_arrays_for_test() :
    rows, cols = 31, 20 # for q and tau
    mu, sigma = 1., 0.2
    arr_g2 = mu + sigma*np.random.standard_normal( size = rows*cols )
    arr_g2.shape = (rows,cols)
    arr_tau = np.arange(rows)
    arr_q   = np.arange(cols)

    print_array(arr_g2,  'arr_g2')
    print_array(arr_q,   'arr_q')
    print_array(arr_tau, 'arr_tau')

    return arr_g2, arr_tau, arr_q

def print_array(arr, msg='') :
    print '\n' + msg + ':\n', arr
    print 'shape:', arr.shape

def main():
    app = QtGui.QApplication(sys.argv)
    w = PlotG2(arrays=get_arrays_for_test(), ofname='./fig_g2.png', title='My title')
    w.move(QtCore.QPoint(50,50))
    w.show()
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
