#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotArray...
#
#------------------------------------------------------------------------

"""Plot for array.

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

import PlotArrayWidget         as imgwidg
import PlotArrayButtons        as imgbuts

from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------

class PlotArray (QtGui.QWidget) :
    """Plot for array"""

    def __init__(self, parent=None, arr=None, ofname='./fig.png', title='', help_msg=None):
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(20, 40, 800, 600)
        self.setWindowTitle('Plot for array')
        self.setFrame()

        self.widgimage = imgwidg.PlotArrayWidget(parent, arr, title=title)
        self.widgbuts  = imgbuts.PlotArrayButtons(self, self.widgimage, ofname, help_msg)
 
        #---------------------
        vbox = QtGui.QVBoxLayout()                      # <=== Begin to combine layout 
        #vbox.addWidget(self.widgimage)                 # <=== Add figure as QWidget
        vbox.addWidget(self.widgimage.getCanvas())      # <=== Add figure as FigureCanvas 
        vbox.addWidget(self.widgbuts)                   # <=== Add buttons         
        self.setLayout(vbox)
        #---------------------
        cp.plotarray_is_on = True


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

        cp.plotarray_is_on = False

        #try    : del cp.plotarray
        #except : pass

        #print 'Close application'

    def set_array(self, arr, title='') :
        self.widgimage.set_array(arr, title)

#-----------------------------
# Test
#-----------------------------

def get_array_for_test() :
    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=500)
    #arr = 100*np.random.standard_exponential(size=500)
    #arr = np.arange(2400)
    #arr.shape = (40,60)
    return arr


def main():
    app = QtGui.QApplication(sys.argv)
    w = PlotArray(arr=get_array_for_test(), ofname='./fig.png')
    w.move(QtCore.QPoint(50,50))
    w.show()
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
