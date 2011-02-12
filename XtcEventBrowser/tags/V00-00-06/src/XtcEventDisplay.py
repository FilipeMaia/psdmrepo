#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module XtcEventDisplay...
#
#------------------------------------------------------------------------

"""MatPlotLib canvas for XtcEventDisplay 

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Ingrid Ofte
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys, os, random
import numpy as np

from PyQt4 import QtGui, QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------



#------------------------
# Exported definitions --
#------------------------

# these definitions are visible to the clients

#---------------------
#  Class definition --
#--------------------- 
class MplCanvas ( FigureCanvas ) :
    """MatPlotLib Canvas and QWidget
    
    """
    
   #--------------------
   #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor.

        @param parent   parent widget, if any
        """

        fig = Figure(figsize=(10,10), dpi=100)
        self.axes = fig.add_subplot(111)
        self.axes.hold(False)

        FigureCanvas.__init__(self,fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)

        FigureCanvas.updateGeometry(self)


    def plot_on(self, array ):
        self.axes.plot(array[0],array[1],"r")
        print "Have now plotted array", array
        return

    #-------------------
    #  Public methods --
    #-------------------


    #--------------------------------
    #  Static/class public methods --
    #--------------------------------

    #--------------------
    #  Private methods --
    #--------------------


class XtcEventDisplay ( QtGui.QWidget ) :
    """MatPlotLib widget

    """
    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None) :
        """Constructor.
        
        """
        QtGui.QWidget.__init__(self, parent)
        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setStyleSheet("QWidget {background-color: #FFFFFF }")
            
        self.setWindowTitle('Xtc Event Diplay')
        self.setWindowIcon(QtGui.QIcon('XtcEventBrowser/src/lclsLogo.gif'))


        #l = QtGui.QVBoxLayout(self)
        #mycanvas = MplCanvas(self)
        #l.addWidget(mycanvas)

        #array = np.array( [ [0,1,2,3,4], [6,5,4,3,2]] )
        #mycanvas.plot_on(array)
        
        

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
