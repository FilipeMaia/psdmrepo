#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIPopupCheckList...
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

class GUIPopupCheckList(QtGui.QDialog) :
    """Gets list of item for checkbox GUI in format [['name1',false], ['name2',true], ..., ['nameN',false]], 
    and modify this list in popup dialog gui.
    """

    def __init__(self, parent=None, list_in_out=[], win_title='Set check boxes'):
        QtGui.QDialog.__init__(self,parent)
        #self.setGeometry(20, 40, 500, 200)
        self.setWindowTitle(win_title)
        self.setFrame()
 
        #self.setModal(True)
        self.list_in_out = list_in_out

        self.vbox = QtGui.QVBoxLayout()

        self.make_gui_checkbox()

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

    def make_gui_checkbox(self) :
        self.dict_of_items = {}
        for k,[name,state] in enumerate(self.list_in_out) :        
            cbx = QtGui.QCheckBox(name) 
            if state : cbx.setCheckState(QtCore.Qt.Checked)
            else     : cbx.setCheckState(QtCore.Qt.Unchecked)
            self.connect(cbx, QtCore.SIGNAL('stateChanged(int)'), self.onCBox)
            self.vbox.addWidget(cbx)

            self.dict_of_items[cbx] = [k,name,state] 

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

    
    def onCBox(self, tristate):
        for cbx in self.dict_of_items.keys() :
            if cbx.hasFocus() :
                k,name,state = self.dict_of_items[cbx]
                state_new = cbx.isChecked()
                msg = 'onCBox: Checkbox #%d:%s - state is changed to %s, tristate=%s'%(k, name, state_new, tristate)
                #print msg
                logger.debug(msg, __name__)
                self.dict_of_items[cbx] = [k,name,state_new]


    def onCancel(self):
        logger.debug('onCancel', __name__)
        self.reject()


    def onApply(self):
        logger.debug('onApply', __name__)  
        self.fill_output_list()
        self.accept()


    def fill_output_list(self):
        """Fills output list"""
        for cbx,[k,name,state] in self.dict_of_items.iteritems() :
            self.list_in_out[k] = [name,state]

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)

    list_in = [['CSPAD1',True], ['CSPAD2x21', False], ['pNCCD1', True], ['Opal1', False], \
               ['CSPAD2',True], ['CSPAD2x22', False], ['pNCCD2', True], ['Opal2', False]]

    for name,state in list_in : print  '%s checkbox is in state %s' % (name.ljust(10), state) 
    
    w = GUIPopupCheckList (None, list_in)
    #w.show()
    resp=w.exec_()
    print 'resp=',resp
    print 'QtGui.QDialog.Rejected: ', QtGui.QDialog.Rejected
    print 'QtGui.QDialog.Accepted: ', QtGui.QDialog.Accepted

    for name,state in list_in : print  '%s checkbox is in state %s' % (name.ljust(10), state) 

    #app.exec_()

#-----------------------------
