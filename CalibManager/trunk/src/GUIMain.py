
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMain...
#
#------------------------------------------------------------------------

"""Renders the main GUI for the CalibManager.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id:$

@author Mikhail S. Dubrovin
"""


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
import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersForApp import cp
from Logger                 import logger
from GUIButtonBar           import *
from GUILogger              import *
from GUITabs                import *
from GUIInsExpDirDet        import *
#from GUICalibDirTree        import *

#---------------------
#  Class definition --
#---------------------
class GUIMain ( QtGui.QWidget ) :
    """Main GUI for calibration management project.

    @see BaseClass
    @see OtherClass
    """
    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIMain'
        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        cp.setIcons()

        self.setGeometry(10, 25, 800, 700)
        self.setWindowTitle('Calibration Manager')
        self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.setFrame()
 
        #self.guitree   = GUICalibDirTree()
        self.guitabs   = GUITabs(self) # QtGui.QTextEdit()
        self.guilogger = GUILogger(show_buttons=False)
        self.guiinsexpdirdet = GUIInsExpDirDet(self)

        self.vsplit = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.vsplit.addWidget(self.guitabs)
        self.vsplit.addWidget(self.guilogger)
        #self.vsplit.moveSplitter(0,200)
        

        self.vbox = QtGui.QVBoxLayout() 
        #self.vbox.addWidget(self.guibuttonbar)
        self.vbox.addWidget(self.guiinsexpdirdet)
        #self.vbox.addLayout(self.hboxB) 
        #self.vbox.addStretch(1)
        self.vbox.addWidget(self.vsplit) 

        self.setLayout(self.vbox)

        self.showToolTips()
        self.setStyle()

        self.move(10,25)
        cp.guimain = self
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def printStyleInfo(self):
        qstyle     = self.style()
        qpalette   = qstyle.standardPalette()
        qcolor_bkg = qpalette.color(1)
        #r,g,b,alp  = qcolor_bkg.getRgb()
        msg = 'Background color: r,g,b,alpha = %d,%d,%d,%d' % ( qcolor_bkg.getRgb() )
        logger.debug(msg)


    def showToolTips(self):
        pass
        #self.butStop.setToolTip('Not implemented yet...')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):
        pass
        self.setMinimumSize(800,700)
        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
        #self.vsplit.setMinimumHeight(700)
        
        #self.        setStyleSheet(cp.styleBkgd)
        #self.butSave.setStyleSheet(cp.styleButton)
        #self.butExit.setStyleSheet(cp.styleButton)
        #self.butELog.setStyleSheet(cp.styleButton)
        #self.butFile.setStyleSheet(cp.styleButton)

        #self.butELog    .setVisible(False)
        #self.butFBrowser.setVisible(False)

        #self.butSave.setText('')
        #self.butExit.setText('')
        #self.butExit.setFlat(True)

        #self.vsplit.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Ignored)


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass

    def closeEvent(self, event):
        logger.info('closeEvent', self.name)

        try    : self.guiinsexpdirdet.close() 
        except : pass
        
        try    : self.guitabs.close()
        except : pass
        
        try    : cp.guilogger.close()
        except : pass

        self.onSave()


    def onSave(self):
        cp.close()
        if cp.save_log_at_exit.value() : logger.saveLogInFile(fnm.log_file())
        #logger.saveLogTotalInFile( fnm.log_file_total() )


#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
    #def mousePressEvent(self, event):
    #    print 'event.x, event.y, event.button =', str(event.x()), str(event.y()), str(event.button())         

    #def mouseReleaseEvent(self, event):
    #    print 'event.x, event.y, event.button =', str(event.x()), str(event.y()), str(event.button())                

#http://doc.qt.nokia.com/4.6/qt.html#Key-enum
    def keyPressEvent(self, event):
        #print 'event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
            #self.close()
            self.SHowIsOn = False    
            pass

        if event.key() == QtCore.Qt.Key_B:
            #print 'event.key() = %s' % (QtCore.Qt.Key_B)
            pass

        if event.key() == QtCore.Qt.Key_Return:
            #print 'event.key() = Return'
            pass

        if event.key() == QtCore.Qt.Key_Home:
            #print 'event.key() = Home'
            pass

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIMain()
    ex.show()
    app.exec_()
#-----------------------------
