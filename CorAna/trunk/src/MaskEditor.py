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

import GlobalUtils              as gu
from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------

#class MaskEditor (QtGui.QMainWindow) :
class MaskEditor (QtGui.QWidget) :
    """Mask editor for 2d array"""


    def __init__(self, parent=None, arr=None, xyc=None, ifname='', ofname='./fig.png', mfname='./roi-mask',
                 title='Mask editor', lw=1, col='b', picker=8, verb=False, ccd_rot=None, updown=None, fexmod=False):
        """List of input parameters:
        @param parent  parent window is used to open other window moved w.r.t. parent.
        @param arr     2D array for image. If None then image will be taken from file or generated as random.
        @param xyc     (x,y) coordinate of the (beam) center which is used to create Wedges. If None, then center of image will be used.
        @param ifname  path to the text file with 2D array for image.
        @param ofname  default path to save plot of the graphical window.
        @param mfname  default path-prefix for newly created files with mask and shaping objects.
        @param title   Initial title of the window.
        @param ccd_rot Orientation of the frame in N*90 degrees 0,90,180,270.
        @param updown  (mirror) flip of the y coordinate True/False. 
        """
 
        #QtGui.QMainWindow.__init__(self, parent)
        QtGui.QWidget.__init__(self, parent)
        #self.setGeometry(20, 40, 500, 550)
        self.setGeometry(20, 40, 900, 900)
        self.setWindowTitle(title)
        self.setFrame()

        if  arr is not None : self.arr = arr
        elif ifname != ''   : self.arr = gu.get_image_array_from_file(ifname)
        else                : self.arr = get_array2d_for_test()

        if self.arr is None : self.arr = get_array2d_for_test()

        self.ifname = ifname
        self.title  = title
        ccd_rot_n90 = int(cp.ccd_orient.value())
        if ccd_rot is not None : ccd_rot_n90 = ccd_rot

        y_is_flip = cp.y_is_flip.value() # True
        if updown is not None : y_is_flip = updown

        self.widgimage   = imgwidg.PlotImgSpeWidget(parent, self.arr, ccd_rot_n90, y_is_flip)
        self.widgbuts    = imgbuts.PlotImgSpeButtons(self, self.widgimage, ifname, ofname, help_msg=self.help_message())
        self.widgmebuts  = mebuts .MaskEditorButtons(self, self.widgimage, ifname, ofname, mfname, xyc, lw, col, picker, verb, ccd_rot_n90, y_is_flip, fexmod)
 
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


    def set_image_array(self,arr,title=None):
        self.widgimage.set_image_array(arr)
        if title is not None : self.setWindowTitle(title)
        else             : self.setWindowTitle(self.title)


    def set_image_array_new(self,arr,title=None):
        self.widgimage.set_image_array_new(arr)
        if title is not None : self.setWindowTitle(title)
        else             : self.setWindowTitle(self.title)


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

    def help_message(self):
        msg  = 'Mouse control buttons for Mask Editor' + \
               '\n' + '='*37 + \
               '\n' + \
               '\nForms' + \
               '\n' + '='*5 + \
               '\n"Rectangle", "Wedge", "Circle", "Line", and "Polygon" forms are used to compose ROI or inversed mask on image. ' + \
               'Selected forms will inverse choosen regions in composition of the mask. ' + \
               'Forms can be added, moved/edited, selected, or removed using appropriate mode, as explained below.' + \
               '\n' + \
               '\n' + \
               '\nModes' + \
               '\n' + '='*5 + \
               '\n"Zoom" - zoom-in image and set limits for spectrum:' + \
               '\nZoom-in image: left mouse button click-drug-release for desired region of image.' + \
               '\nSet limits for spectrum: left/right mouse button click on desired min/max limit on spectrum.' + \
               '\nReset to full size: middle mouse button click or "Reset" button.' + \
               '\n' + \
               '\n"Add" - add new form to compose the mask on image: ' + \
               '\nIn the "Add" mode for Rectangle, Wedge, Circle, and Line use left mouse button click-drug-release. ' + \
               'For Polygon use left mouse button click to add each next vertex and right mouse button click for the last vertex. '+  \
               'Polygon form will be closed automatically after the last right mouse button click.' + \
               '\n' + \
               '\n"Move" - move/edit form: ' + \
               '\nIn the Move mode choose the form, then use mouse button click-drug-release on form boarder.' + \
               'to move a single vertex or entire form for left or right button, respectively.' + \
               '\n' + \
               '\n"Select" - select/deselect form(s) for inverse mask region(s): \nIn the Select mode choose the form, ' + \
               'then use left mouse button click on form boarder. Form color is changed at selection/deselection.' + \
               '\n' + \
               '\n"Remove" - removes form from image:' + \
               '\nIn the Remove mode choose the form, then use any mouse button click on form boarder.' + \
               '\nAs an alternative option, currently active form can be removed by the middle mouse button click on its boarder.' + \
               '\n' + \
               '\nI/O' + \
               '\n' + '='*5 + \
               '\n"Load Image"  - loads image for display from file.' + \
               '\n"Load Forms"  - loads forms of masked regions from file.' + \
               '\n"Save Forms"  - saves forms of masked regions in text file.' + \
               '\n"Save Mask"   - saves mask as a 2D array of ones and zeros in text file.' + \
               '\n"Save Inv-M"  - saves inversed-mask as a 2D array of ones and zeros in text file.' + \
               '\n"Print Forms" - prints parameters of currently available forms.' + \
               '\n"Clear Forms" - removes all forms from image.' + \
               '\n'

        return msg

#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    rows, cols = 1300, 1340
    #arr = mu + sigma*np.random.standard_normal(size=rows*cols)
    #arr = 100*np.random.standard_exponential(size=2400)
    arr = np.arange(rows*cols)*0.001
    arr.shape = (rows,cols)
    return arr


def main():

    app = QtGui.QApplication(sys.argv)
    w = MaskEditor(None, get_array2d_for_test(), xyc=(600,700), ccd_rot=90, updown=False)
    #w = MaskEditor(None, get_array2d_for_test(), xyc=(600,700))
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
