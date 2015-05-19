#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIPopupRadioList...
#
#------------------------------------------------------------------------

"""Send message to ELog"""

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

class GUIPopupRadioList(QtGui.QDialog) :
    """Gets list of item for checkbox GUI in format [['name1',false], ['name2',true], ..., ['nameN',false]], 
    and modify this list in popup dialog gui.
    """

    def __init__(self, parent=None, dict_of_pars={}, win_title='Load action', do_confirm=True):
        QtGui.QDialog.__init__(self,parent)
        #self.setGeometry(20, 40, 500, 200)
        self.setWindowTitle(win_title)
        self.setFrame()
 
        #self.setModal(True)
        self.dict_of_pars = dict_of_pars
        self.do_confirm   = do_confirm

        self.vbox = QtGui.QVBoxLayout()

        self.make_radio_buttons()

        self.but_cancel = QtGui.QPushButton('&Cancel') 
        self.but_apply  = QtGui.QPushButton('&Apply') 
        cp.setIcons()
        self.but_cancel.setIcon(cp.icon_button_cancel)
        self.but_apply .setIcon(cp.icon_button_ok)
        
        self.connect( self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel )
        self.connect( self.but_apply,  QtCore.SIGNAL('clicked()'), self.onApply )

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.but_cancel)
        self.hbox.addWidget(self.but_apply)
        self.hbox.addStretch(1)

        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)

        self.but_cancel.setFocusPolicy(QtCore.Qt.NoFocus)

        self.setStyle()
        self.showToolTips()

#-----------------------------  

    def make_radio_buttons(self) :
        self.list_of_rad = []
        self.rad_grp = QtGui.QButtonGroup()

        pattern = self.dict_of_pars['checked']

        for name in self.dict_of_pars['list']:
            rad = QtGui.QRadioButton(name) 
            self.list_of_rad.append(rad)
            self.vbox.addWidget(rad)
            self.connect(rad, QtCore.SIGNAL('clicked()'), self.onRadioButton)
            #if name == pattern : rad.setChecked(True)  

#-----------------------------  

    def showToolTips(self):
        self.but_apply.setToolTip('Apply changes to the list')
        self.but_cancel.setToolTip('Use default list')
        
    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        #self.setFixedWidth(200)
        self.setMinimumWidth(200)
        self.setStyleSheet(cp.styleBkgd)
        self.but_cancel.setStyleSheet(cp.styleButton)
        self.but_apply.setStyleSheet(cp.styleButton)
        self.but_cancel.setVisible(self.do_confirm) 
        self.but_apply .setVisible(self.do_confirm) 
        
 
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

    #def event(self, event):
    #    print 'Event happens...:', event

    
    def onRadioButton(self):
        if not self.do_confirm :
            self.applySelection()


        #for rad in self.list_of_rad :
        #    if rad.isChecked() :
        #        msg = 'Selected button: %s' % str(rad.text())
        #        logger.info(msg, __name__)
        #        break;


    def applySelection(self):
        for rad in self.list_of_rad :
            if rad.isChecked() :
                name = str(rad.text()) 
                self.dict_of_pars['checked'] = name
                logger.info('Selected button: %s' % name, __name__)
                self.accept()
                break;


    def onCancel(self):
        logger.debug('onCancel', __name__)
        self.reject()


    def onApply(self):
        logger.debug('onApply', __name__)  
        self.applySelection()


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)


    dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}

    #w = GUIPopupRadioList (None, dict_of_pars, win_title='Radio buttons', do_confirm=True)
    w = GUIPopupRadioList (None, dict_of_pars, win_title='Radio buttons', do_confirm=False)
    #w.show()
    resp=w.exec_()
    print 'dict=',str(dict_of_pars)
    print 'resp=',resp
    print 'QtGui.QDialog.Rejected: ', QtGui.QDialog.Rejected
    print 'QtGui.QDialog.Accepted: ', QtGui.QDialog.Accepted

    #app.exec_()

#-----------------------------
