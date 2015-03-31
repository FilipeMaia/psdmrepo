
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIROIMask...
#
#------------------------------------------------------------------------

"""Renders the main GUI for the CalibManager.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import numpy as np

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

#from CalibManager.Frame   import Frame
from CalibManager.Logger  import logger
from GUIMaskEditor        import * 

#---------------------
#  Class definition --
#---------------------
#class GUIROIMask ( QtGui.QWidget ) :
class GUIROIMask ( Frame ) : 
    """QWidger wrapping ROI mask processing.

    @see BaseClass
    @see OtherClass
    """
    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIROIMask'
        QtGui.QWidget.__init__(self, parent)
        #Frame.__init__(self, parent, mlw=1)

        self.setGeometry(10, 25, 900, 400)
        self.setWindowTitle('ROI Mask')

        self.win = GUIMaskEditor(self)
        #self.lab_status = QtGui.QLabel('Status: ')

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addWidget(self.win)
        self.vbox.addStretch(1)
        #self.vbox.addWidget(self.lab_status)

        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addStretch(1)
        self.hbox.addLayout(self.vbox)
        self.hbox.addStretch(1)

        self.setLayout(self.hbox)
        
        self.showToolTips()
        self.setStyle()

        #self.setStatus(0)
        cp.guiroimask = self
        self.move(10,25)
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        pass
        #self.setToolTip('ROI mask wrapping widget') 

    def setStyle(self):

        self.setMinimumSize(800,400)
        #self.setMaximumWidth(800)
        #self.lab_status.setMinimumWidth(600) 

        #self.but_mask_editor.setStyleSheet(cp.styleButton)
        #self.but_mask_editor.setFixedWidth(200)
        #self.but_mask_editor.setMinimumHeight(60)
        #self.but_mask_editor.setMinimumSize(180,40)
        #self.but_roi_convert.setMinimumSize(180,40)
        #self.but_reco_image .setMinimumSize(180,40)

        #self.edi_roi_mask_nda.setReadOnly(True)

        #self.edi_geometry    .setEnabled(False)


#    def resizeEvent(self, e):
#        pass


#    def moveEvent(self, e):
#        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', self.name)

        try    : cp.maskeditor.close()
        except : pass

        cp.guiroimask = None


#    def onExit(self):
#        logger.debug('onExit', self.name)
#        self.close()


#    def setStatus(self, status_index=0, msg='Waiting for the next command'):
#        list_of_states = ['Good','Warning','Alarm']
#        if status_index == 0 : self.lab_status.setStyleSheet(cp.styleStatusGood)
#        if status_index == 1 : self.lab_status.setStyleSheet(cp.styleStatusWarning)
#        if status_index == 2 : self.lab_status.setStyleSheet(cp.styleStatusAlarm)
#        #self.lab_status.setText('Status: ' + list_of_states[status_index] + msg)
#        self.lab_status.setText(msg)

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIROIMask()
    ex.show()
    app.exec_()
#-----------------------------
