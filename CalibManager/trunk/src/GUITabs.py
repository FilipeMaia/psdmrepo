
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
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersForApp import cp

from GUILogger            import *
from Logger               import logger
from FileNameManager      import fnm
from GUIConfig            import * 
from GUIDark              import * 


#from GUIDark              import * 
#from GUIFiles             import *
#from GUISetupInfo         import *
#from GUIAnaSettings       import *
#from GUISystemSettings    import *
#from GUIIntensityMonitors import *
#from GUIRun               import *
#from GUIViewResults       import *
#from GUIFileBrowser       import *
#from GUIELogPostingDialog import *

#---------------------
#  Class definition --
#---------------------
class GUITabs ( QtGui.QWidget ) :
    """Main GUI for calibration management project.

    @see BaseClass
    @see OtherClass
    """

    list_of_tabs = [ 'Dark'
                    ,'Gain'
                    ,'ROI'
                    ,'Geometry'
                    ,'Configuration'
                    ]


    orientation = 'H'
    #orientation = 'V'


    def __init__ (self, parent=None, app=None) :

        self.name = 'GUITabs'
        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        cp.setIcons()
 
        self.setGeometry(10, 25, 750, 500)
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

        self.hboxW = QtGui.QHBoxLayout() 

        self.vboxW = QtGui.QVBoxLayout() 
        self.vboxW.addStretch(1)
        self.vboxW.addLayout(self.hboxW) 
        self.vboxW.addStretch(1)

        self.hboxWW= QtGui.QHBoxLayout() 
        self.hboxWW.addStretch(1)
        self.hboxWW.addLayout(self.vboxW) 
        self.hboxWW.addStretch(1)

        self.makeTabBar()
        self.guiSelector()

        if self.orientation == 'H' : self.box = QtGui.QVBoxLayout() 
        else :                       self.box = QtGui.QHBoxLayout() 

        self.box.addWidget(self.tab_bar)
        self.box.addLayout(self.hboxWW)
        self.box.addStretch(1)

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

        #self.butSave.setText('')
        #self.butExit.setText('')
        #self.butExit.setFlat(True)


    def makeTabBar(self,mode=None) :
        #if mode != None : self.tab_bar.close()
        self.tab_bar = QtGui.QTabBar()


        #len(self.list_of_tabs)
        for tab_name in self.list_of_tabs :
            tab_ind = self.tab_bar.addTab( tab_name )
            self.tab_bar.setTabTextColor(tab_ind, QtGui.QColor('blue')) #gray, red, grayblue

        #Uses self.list_file_types
        #self.ind_tab_0 = self.tab_bar.addTab( self.list_of_tabs[0] )
        #self.ind_tab_1 = self.tab_bar.addTab( self.list_of_tabs[1] )
        #self.ind_tab_2 = self.tab_bar.addTab( self.list_of_tabs[2] )
        #self.ind_tab_3 = self.tab_bar.addTab( self.list_of_tabs[3] )
        #self.ind_tab_4 = self.tab_bar.addTab( self.list_of_tabs[4] )

        #self.tab_bar.setTabTextColor(self.ind_tab_0, QtGui.QColor('blue')) #gray, red, grayblue
        #self.tab_bar.setTabTextColor(self.ind_tab_1, QtGui.QColor('blue'))
        #self.tab_bar.setTabTextColor(self.ind_tab_2, QtGui.QColor('blue'))
        #self.tab_bar.setTabTextColor(self.ind_tab_3, QtGui.QColor('blue'))
        #self.tab_bar.setTabTextColor(self.ind_tab_4, QtGui.QColor('blue'))

        if self.orientation == 'H' :
            self.tab_bar.setShape(QtGui.QTabBar.RoundedNorth)
        else :
            self.tab_bar.setShape(QtGui.QTabBar.RoundedWest)

        #self.tab_bar.setTabEnabled(1, False)
        #self.tab_bar.setTabEnabled(3, False)

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

        if   cp.current_tab.value() == self.list_of_tabs[0] :
            self.gui_win = GUIDark(self)
            #self.gui_win = GUIFiles(self)
            #self.setStatus(0, 'Status: processing for pedestals')
            
        elif cp.current_tab.value() == self.list_of_tabs[1] :
            self.gui_win = QtGui.QTextEdit() # GUIDark(self)

        elif cp.current_tab.value() == self.list_of_tabs[2] :
            self.gui_win = QtGui.QTextEdit()

        elif cp.current_tab.value() == self.list_of_tabs[3] :
            self.gui_win = QtGui.QTextEdit()

        elif cp.current_tab.value() == self.list_of_tabs[4] :
            self.gui_win = GUIConfig(self)
            #self.setStatus(0, 'Status: processing for data')

        #self.gui_win.setMinimumWidth(500)
        #self.gui_win.setMinimumHeight(300)
        #self.gui_win.setMinimumSize(500,400)
        self.gui_win.setMinimumSize(750,400)
        #self.gui_win.setBaseSize(700,500)
        #self.gui_win.adjustSize()

        self.hboxW.addWidget(self.gui_win)

        #min_height = self.gui_win.minimumHeight() 
        #self.setFixedHeight(min_height + 90)
        #self.setMinimumHeight(min_height + 90)


    def onTabBar(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        cp.current_tab.setValue( tab_name )
        msg = 'Selected tab: %i - %s' % (tab_ind, tab_name)
        logger.info(msg, __name__)
        self.guiSelector()


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
    ex.show()
    app.exec_()
#-----------------------------
