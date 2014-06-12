#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotGraph...
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
#if __name__ == "__main__" :
#    import matplotlib
#    matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)

import matplotlib
if matplotlib.get_backend() != 'Qt4Agg' : matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

#print 'Backend:', matplotlib.get_backend()

#from matplotlib.figure import Figure
from PyQt4 import QtGui, QtCore
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
#-----------------------------
# Imports for other modules --
#-----------------------------

import PlotGraphWidget      as imgwidg
#import PlotG2Buttons        as imgbuts
import PlotArrayButtons     as imgbuts

from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------

class PlotGraph (QtGui.QWidget) :
    """Plot for graphic arrays"""

    def __init__(self, parent=None, arrays=None, ofname='./fig_graphs.png', title='', axlabs=('','')):
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(20, 40, 800, 600)
        self.setWindowTitle('Plot for G2 arrays')
        self.setFrame()

        self.widgimage = imgwidg.PlotGraphWidget(parent, arrays, title=title, axlabs=axlabs)
        #self.widgbuts  = imgbuts.PlotG2Buttons(self, self.widgimage, ofname)
        self.widgbuts  = imgbuts.PlotArrayButtons(self, self.widgimage, ofname)#, help_msg)
  
        #---------------------
        vbox = QtGui.QVBoxLayout()                      # <=== Begin to combine layout 
        #vbox.addWidget(self.widgimage)                 # <=== Add figure as QWidget
        vbox.addWidget(self.widgimage.getCanvas())      # <=== Add figure as FigureCanvas 
        vbox.addWidget(self.widgbuts)                   # <=== Add buttons         
        self.setLayout(vbox)
        #---------------------
        cp.plot_graph_is_on = True


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

        cp.plot_graph_is_on = False

        try    : del cp.plot_graph # suicide..., but what to do in order to close it properly?
        except : pass

        #print 'Close application'

    def set_array(self, arr, title='') :
        self.widgimage.set_array(arr, title)

#-----------------------------
# Test
#-----------------------------

def get_arrays_for_test() :
    rows, cols = 100, 50 # for q and tau
    mu, sigma = 1., 0.2
    arrsy = mu + sigma*np.random.standard_normal( size = rows*cols )
    arrsy.shape = (rows,cols)
    arr_x = np.arange(cols)
    arr_n = np.arange(rows)

    print_array(arrsy, 'arrsy')
    print_array(arr_x, 'arr_x')
    print_array(arr_n, 'arr_n')

    return arrsy, arr_x, arr_n         # MULTI-graphics
    #return arrsy[0,:], arr_x, None    # ONE-graphic

def print_array(arr, msg='') :
    print '\n' + msg + ':\n', arr
    print 'shape:', arr.shape

def main():
    app = QtGui.QApplication(sys.argv)
    w = PlotGraph(arrays=get_arrays_for_test(), ofname='./fig_gr.png', title='My title', axlabs=( r'$\tau$[sec] ', r'$g_{2}$' ))
    w.move(QtCore.QPoint(50,50))
    w.show()
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
