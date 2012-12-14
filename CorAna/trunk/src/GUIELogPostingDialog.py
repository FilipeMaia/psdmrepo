#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIELogPostingDialog...
#
#------------------------------------------------------------------------

"""Send message to ELog"""

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

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------
from Logger import logger
from ConfigParametersCorAna import confpars as cp
from GUIELogPostingFields import *

#---------------------
#  Class definition --
#---------------------


class GUIELogPostingDialog(QtGui.QDialog) :
    def __init__(self, parent=None, fname=None):
        QtGui.QDialog.__init__(self,parent)
        self.setGeometry(20, 40, 500, 200)
        self.setWindowTitle('Send message to ELog')
        self.setFrame()
 
        #self.setModal(True)
        self.widg_pars = GUIELogPostingFields(self,att_fname=fname)
        self.but_canc  = QtGui.QPushButton('&Cancel') 
        self.but_send  = QtGui.QPushButton('&Send to ELog') 

        self.hbox  = QtGui.QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_canc)
        self.hbox.addWidget(self.but_send)

        self.vbox  = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.widg_pars)
        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)

        self.connect( self.but_canc, QtCore.SIGNAL('clicked()'), self.onCancel )
        self.connect( self.but_send, QtCore.SIGNAL('clicked()'), self.onSend )

        #self.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)

        self.setStyle()
        self.but_send.setFocus(False)
        self.but_canc.setFocus(False)

#-----------------------------  

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setMinimumWidth(400)
        self.setStyleSheet(cp.styleBkgd)
        self.but_canc.setStyleSheet(cp.styleButton)
        self.but_send.setStyleSheet(cp.styleButton)
 
    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        print 'closeEvent'
        try    : self.widg_pars.close()
        except : pass

    def onCancel(self):
        logger.debug('onCancel', __name__)
        #print 'onCancel'
        self.reject()
        #self.close()

    def onSend(self):
        logger.debug('onSend', __name__)
        #print 'onSend'
        self.widg_pars.updateConfigPars()
        list_of_fields = self.widg_pars.getListOfFields()

        logger.info('onSend: Send to ELod the massege with parameters:', __name__)        
        for (label, edi, par, val_loc) in list_of_fields :
            msg = str(label.text()) + ' ' + val_loc
            logger.info(msg, __name__)        
            print msg       


        self.accept()
        #self.close()
 
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIELogPostingDialog ()
    #w.show()
    resp=w.exec_()
    print 'resp=',resp
    print 'QtGui.QDialog.Rejected: ', QtGui.QDialog.Rejected
    print 'QtGui.QDialog.Accepted: ', QtGui.QDialog.Accepted

    #app.exec_()

#-----------------------------
