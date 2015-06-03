
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUITabs...
#
#------------------------------------------------------------------------

"""GUI for tabs.

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

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersForApp import cp

#from GUILogger            import *
#from FileNameManager      import fnm
from Logger               import logger
from GUIConfig            import * 
from GUIDark              import * 
from GUIROIMask           import * 
from GUIFileManager       import * 
from GUIGeometry          import * 

#---------------------
#  Class definition --
#---------------------
class GUITabs ( QtGui.QWidget ) :
    """GUI for tabs support in GUIMain.

    @see BaseClass
    @see OtherClass
    """

    list_of_tabs = [ 'Dark'
                    ,'Gain'
                    ,'ROI'
                    ,'Geometry'
                    ,'File Manager'
                    ,'Configuration'
                    ]


    orientation = 'H'
    #orientation = 'V'


    def __init__ (self, parent=None, app=None) :

        self.name = 'GUITabs'
        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        cp.setIcons()
 
        self.setGeometry(10, 25, 400, 600)
        self.setWindowTitle('Calibration Manager')
        self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.setFrame()

        #self.butELog    .setIcon(cp.icon_mail_forward)
        #self.butFile    .setIcon(cp.icon_save)
        #self.butExit    .setIcon(cp.icon_exit)
        #self.butLogger  .setIcon(cp.icon_logger)
        #self.butFBrowser.setIcon(cp.icon_browser)
        #self.butSave    .setIcon(cp.icon_save_cfg)
        #self.butStop    .setIcon(cp.icon_stop)

        self.gui_win = None

        self.hboxW = QtGui.QHBoxLayout()

        self.makeTabBar()
        self.guiSelector()

        if self.orientation == 'H' : self.box = QtGui.QVBoxLayout(self) 
        else :                       self.box = QtGui.QHBoxLayout(self) 

        self.box.addWidget(self.tab_bar)
        self.box.addLayout(self.hboxW)
        #self.box.addStretch(1)

        self.setLayout(self.box)

        self.showToolTips()
        self.setStyle()
        gu.printStyleInfo(self)

        cp.guitabs = self
        self.move(10,25)
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        pass
        #self.butExit.setToolTip('Close all windows and \nexit this program') 


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def setStyle(self):
        pass
        #self.adjustSize()
        #self.        setStyleSheet(cp.styleBkgd)
        #self.butSave.setStyleSheet(cp.styleButton)
        #self.butExit.setStyleSheet(cp.styleButton)
        #self.butELog.setStyleSheet(cp.styleButton)
        #self.butFile.setStyleSheet(cp.styleButton)

        #self.butELog    .setVisible(False)
        #self.butFBrowser.setVisible(False)
        #self.butExit.setText('')
        #self.butExit.setFlat(True)

        self.setMinimumHeight(250)
        #self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
 

    def makeTabBar(self,mode=None) :
        #if mode is not None : self.tab_bar.close()
        self.tab_bar = QtGui.QTabBar()

        #len(self.list_of_tabs)
        for tab_name in self.list_of_tabs :
            tab_ind = self.tab_bar.addTab( tab_name )
            self.tab_bar.setTabTextColor(tab_ind, QtGui.QColor('blue')) #gray, red, grayblue

        if self.orientation == 'H' :
            self.tab_bar.setShape(QtGui.QTabBar.RoundedNorth)
        else :
            self.tab_bar.setShape(QtGui.QTabBar.RoundedWest)

        self.setTabByName(cp.current_tab.value())
            
        self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.onTabBar)


    def setTabByName(self, tab_name) :
        try    :
            tab_index = self.list_of_tabs.index(tab_name)
        except :
            tab_index = 0
            cp.current_tab.setValue(self.list_of_tabs[tab_index])
        logger.info(' makeTabBarr - set tab: %s' % tab_name, __name__)
        self.tab_bar.setCurrentIndex(tab_index)


    def guiSelector(self):

        try    : self.gui_win.close()
        except : pass

        try    : del self.gui_win
        except : pass

        self.gui_win = None

        if   cp.current_tab.value() == self.list_of_tabs[0] :
            self.gui_win = GUIDark(self)
            #self.gui_win = GUIFiles(self)
            #self.setStatus(0, 'Status: processing for pedestals')
            
        elif cp.current_tab.value() == self.list_of_tabs[1] :
            self.gui_win = QtGui.QTextEdit('') # Gain is not implemented.') # GUIDark(self)

        elif cp.current_tab.value() == self.list_of_tabs[2] :
            self.gui_win = GUIROIMask(self)

        elif cp.current_tab.value() == self.list_of_tabs[3] :
            self.gui_win = GUIGeometry(self)

        elif cp.current_tab.value() == self.list_of_tabs[4] :
            #self.gui_win = QtGui.QTextEdit('File manager is not implemented.')
            self.gui_win = GUIFileManager(self)

        elif cp.current_tab.value() == self.list_of_tabs[5] :
            self.gui_win = GUIConfig(self)
            #self.setStatus(0, 'Status: processing for data')

        self.hboxW.addWidget(self.gui_win)
        self.gui_win.setVisible(True)


    def onTabBar(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        cp.current_tab.setValue( tab_name )
        msg = 'Selected tab: %i - %s' % (tab_ind, tab_name)
        logger.info(msg, __name__)
        self.guiSelector()


    def resizeEvent(self, e):
        self.frame.setGeometry(self.rect())
        #logger.debug('resizeEvent', self.name) 
        #print 'GUITabs resizeEvent: %s' % str(self.size())


    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass


    def closeEvent(self, event):
        logger.info('closeEvent', self.name)

        try    : self.gui_win.close()
        except : pass

        #try    : del self.gui_win
        #except : pass


    def onExit(self):
        logger.debug('onExit', self.name)
        self.close()

#-----------------------------
#-----------------------------
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUITabs()
    ex.move(QtCore.QPoint(50,50))
    ex.show()
    app.exec_()
#-----------------------------
