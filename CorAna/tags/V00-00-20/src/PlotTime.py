#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotTime...
#
#------------------------------------------------------------------------

"""Plot for time records.

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

import PlotTimeWidget   as imgwidg
import PlotArrayButtons as imgbuts

from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------

#class PlotTime (QtGui.QMainWindow) :
class PlotTime (QtGui.QWidget) :
    """Plot for time records"""


    def __init__(self, parent=None, ifname=None, ofname='./fig.png'):
        #QtGui.QMainWindow.__init__(self, parent)
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(20, 40, 800, 400)
        self.setWindowTitle('Plot for time records')
        self.setFrame()

        self.widgimage   = imgwidg.PlotTimeWidget(parent, ifname)
        self.widgbuts    = imgbuts.PlotArrayButtons(self, self.widgimage, ofname)
 
        #---------------------

        vbox = QtGui.QVBoxLayout()                      # <=== Begin to combine layout 
        #vbox.addWidget(self.widgimage)                 # <=== Add figure as QWidget
        vbox.addWidget(self.widgimage.getCanvas())      # <=== Add figure as FigureCanvas 
        vbox.addWidget(self.widgbuts)                   # <=== Add buttons         
        self.setLayout(vbox)

        #---------------------

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

        #try    : del cp.plottime
        #except : pass

        #print 'Close application'


#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    #arr = mu + sigma*np.random.standard_normal(size=2400)
    arr = 100*np.random.standard_exponential(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)
    return arr


def main():

    app = QtGui.QApplication(sys.argv)
    fname = 'work/cora-xcsi0112-r0015-data-scan-tstamp-list.txt'
    fname = '/reg/neh/home1/dubrovin/LCLS/PSANA-V01/work-1/t1-xcsi0112-r0015-data-scan-tstamp-list.txt'
    w  = PlotTime(None, fname, './fig.png')
    w.move(QtCore.QPoint(50,50))
    w.show()
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
