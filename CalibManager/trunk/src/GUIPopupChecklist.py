#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIPopupChecklist...
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
from ConfigParametersForApp import cp

#---------------------
#  Class definition --
#---------------------


class GUIPopupChecklist(QtGui.QDialog) :

    def __init__(self, parent=None, list_of_items_in=[]):
        QtGui.QDialog.__init__(self,parent)
        #self.setGeometry(20, 40, 500, 200)
        self.setWindowTitle('Send message to ELog')
        self.setFrame()
 
        #self.setModal(True)

        self.vbox = QtGui.QVBoxLayout()


        self.list_of_items = []

        for k,v in list_of_items_in :
        
            cbx = QtGui.QCheckBox(k) 
            if v : cbx.setCheckState(QtCore.Qt.Checked)
            self.connect( cbx, QtCore.SIGNAL('stateChanged(int)'), self.onCBox)

            self.list_of_items.append([cbx, k, v]) 

            self.vbox.addWidget(cbx)


        self.but_cancel = QtGui.QPushButton('&Cancel') 
        self.but_apply  = QtGui.QPushButton('&Apply') 
        cp.setIcons()
        self.but_cancel.setIcon(cp.icon_button_cancel)
        self.but_apply .setIcon(cp.icon_button_ok)
        
        self.connect( self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel )
        self.connect( self.but_apply,  QtCore.SIGNAL('clicked()'), self.onApply )

        self.hbox = QtGui.QHBoxLayout()
        #self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_cancel)
        self.hbox.addWidget(self.but_apply)
        self.hbox.addStretch(1)

        #self.vbox.addWidget(self.widg_pars)
        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)

        self.but_cancel.setFocusPolicy(QtCore.Qt.NoFocus)
        #self.but_apply.setFocusPolicy(QtCore.Qt.NoFocus)
        #self.but_apply.setFocus()

        self.setStyle()
        self.showToolTips()

#-----------------------------  

    def showToolTips(self):
        self.but_apply.setToolTip('Mouse click on this button or Alt-S \nor "Enter" submits message to ELog')
        self.but_cancel.setToolTip('Mouse click on this button \nor Alt-C cancels submission...')
        #self.cbx_cntl.setToolTip('Lock/unlock top row \nof control buttons')
        
    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setFixedWidth(500)
        self.setStyleSheet(cp.styleBkgd)
        self.but_cancel.setStyleSheet(cp.styleButton)
        self.but_apply.setStyleSheet(cp.styleButton)
 
    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        #print 'closeEvent'
        #try    : self.widg_pars.close()
        #except : pass

    def onCBox(self, state):

        for cbx,k,v in self.list_of_items :
            if cbx.hasFocus() :
                msg = 'onCBox: k:%s, hasFocus: %s, isChecked: %s, state %s'%( k, cbx.hasFocus(), cbx.isChecked(), state)
                lstate = cbx.isChecked()
                print msg
                #logger.info(msg, __name__)
        
        # str(self.cbx_cntl.checkState())
        # cbx_cntl.isChecked()
        #logger.info('onCBox: control lock state: ', __name__)

    def onCancel(self):
        logger.debug('onCancel', __name__)
        self.reject()
        #self.close()

    def onApply(self):
        logger.info('onApply', __name__)  
        self.accept()
 
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)

    list_of_items = [['A',True], ['B', False], ['C', True], ['D', False]]
    
    w = GUIPopupChecklist (None, list_of_items)
    #w.show()
    resp=w.exec_()
    print 'resp=',resp
    print 'QtGui.QDialog.Rejected: ', QtGui.QDialog.Rejected
    print 'QtGui.QDialog.Accepted: ', QtGui.QDialog.Accepted

    #app.exec_()

#-----------------------------
