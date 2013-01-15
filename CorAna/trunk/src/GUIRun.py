#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRun...
#
#------------------------------------------------------------------------

"""GUI control of the entire file processing procedure"""

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

from ConfigParametersCorAna import confpars as cp
from GUIConfigParameters    import *
from GUIRunInfo             import *
from GUIRunSplit            import *
from GUIRunProc             import *
from GUIRunMerge            import *
from Logger                 import logger
from BatchJobPedestals      import bjpeds

#---------------------
#  Class definition --
#---------------------
class GUIRun ( QtGui.QWidget ) :
    """GUI control of the entire file processing procedure"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(1, 1, 800, 600)
        self.setWindowTitle('Run control')
        self.setFrame()

        self.lab_title  = QtGui.QLabel     ('Run control')
        #self.but_save   = QtGui.QPushButton('&Save')
        
        self.hboxW = QtGui.QHBoxLayout()
        #self.hboxB = QtGui.QHBoxLayout()
        #self.hboxB.addWidget(self.lab_status)
        #self.hboxB.addStretch(1)     
        #self.hboxB.addWidget(self.but_save)

        self.list_run_types = ['Info', 'Split', 'Process', 'Merge', 'Auto']
        self.makeTabBar()
        self.guiSelector()

        self.vbox = QtGui.QVBoxLayout()   
        self.vbox.addWidget(self.lab_title)
        self.vbox.addWidget(self.tab_bar)
        self.vbox.addLayout(self.hboxW)
        #self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)

        #self.connect( self.but_save, QtCore.SIGNAL('clicked()'), self.onSave )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        msg = 'Edit field'
        #self.but_save  .setToolTip('Save all current configuration parameters.')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.          setStyleSheet (cp.styleBkgd)

        self.lab_title.setStyleSheet (cp.styleTitleBold)
        self.lab_title.setAlignment(QtCore.Qt.AlignCenter)
        #self.setMinimumWidth (600)
        #self.setMaximumWidth (700)
        #self.setMinimumHeight(300)
        #self.setMaximumHeight(400)
        #self.setFixedWidth (700)
        #self.setFixedHeight(400)
        #self.setFixedHeight(330)
        #self.setFixedSize(550,350)
        self.setFixedSize(750,500)
        
    def makeTabBar(self,mode=None) :
        #if mode != None : self.tab_bar.close()
        self.tab_bar = QtGui.QTabBar()

        #Uses self.list_run_types
        self.ind_tab_info  = self.tab_bar.addTab( self.list_run_types[0] )
        self.ind_tab_split = self.tab_bar.addTab( self.list_run_types[1] )
        self.ind_tab_proc  = self.tab_bar.addTab( self.list_run_types[2] )
        self.ind_tab_merge = self.tab_bar.addTab( self.list_run_types[3] )
        self.ind_tab_auto  = self.tab_bar.addTab( self.list_run_types[4] )

        self.tab_bar.setTabTextColor(self.ind_tab_info , QtGui.QColor('green'))
        self.tab_bar.setTabTextColor(self.ind_tab_split, QtGui.QColor('red'))
        self.tab_bar.setTabTextColor(self.ind_tab_proc , QtGui.QColor('gray'))
        self.tab_bar.setTabTextColor(self.ind_tab_merge, QtGui.QColor('blue'))
        self.tab_bar.setTabTextColor(self.ind_tab_auto , QtGui.QColor('black'))
        self.tab_bar.setShape(QtGui.QTabBar.RoundedNorth)

        #self.tab_bar.setTabEnabled(1, False)
        #self.tab_bar.setTabEnabled(2, False)
        #self.tab_bar.setTabEnabled(3, False)
        #self.tab_bar.setTabEnabled(4, False)
        
        try    :
            tab_index = self.list_run_types.index(cp.current_run_tab.value())
        except :
            tab_index = 3
            cp.current_run_tab.setValue(self.list_run_types[tab_index])
        self.tab_bar.setCurrentIndex(tab_index)

        logger.info(' make_tab_bar - set mode: ' + cp.current_run_tab.value(), __name__)

        self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.onTabBar)


    def guiSelector(self):

        try    : self.gui_win.close()
        except : pass

        try    : del self.gui_win
        except : pass

        if cp.current_run_tab.value() == self.list_run_types[0] :
            self.gui_win = GUIRunInfo(self)
            #self.setStatus(0, 'Status: run info')
            
        if cp.current_run_tab.value() == self.list_run_types[1] :
            self.gui_win = GUIRunSplit(self)
            #self.setStatus(0, 'Status: split')

        if cp.current_run_tab.value() == self.list_run_types[2] :
            self.gui_win = GUIRunProc(self)
            #self.gui_win = QtGui.QLineEdit( 'Empty' )
            #self.setStatus(0, 'Status: processing for correlations')

        if cp.current_run_tab.value() == self.list_run_types[3] :
            self.gui_win = GUIRunMerge(self)
            #self.gui_win = QtGui.QLineEdit( 'Empty' )
            #self.setStatus(0, 'Status: merging')

        if cp.current_run_tab.value() == self.list_run_types[4] :
        #    self.gui_win = GUIRunAuto(self)
            self.gui_win = QtGui.QLineEdit( 'Empty' )
            #self.setStatus(0, 'Status: auto run')

        #self.gui_win.setFixedHeight(180)
        self.gui_win.setFixedHeight(400)
        self.hboxW.addWidget(self.gui_win)

    def onTabBar(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        cp.current_run_tab.setValue( tab_name )
        logger.info(' ---> selected tab: ' + str(tab_ind) + ' - open GUI to work with: ' + tab_name, __name__)
        self.guiSelector()

    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent: new pos:' + str(self.position), __name__)
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

        try    : cp.guimain.butFiles.setStyleSheet(cp.styleButton)
        except : pass

        try    : self.gui_win.close()
        except : pass

        try    : self.tab_bar.close()
        except : pass
        
        try    : del cp.guirun # GUIRun
        except : pass # silently ignore

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def onSave(self):
        logger.debug('onSave', __name__)
        cp.saveParametersInFile( cp.fname_cp.value() )

#    def setStatus(self, status_index=0, msg=''):
#        list_of_states = ['Good','Warning','Alarm']
#        if status_index == 0 : self.lab_status.setStyleSheet(cp.styleStatusGood)
#        if status_index == 1 : self.lab_status.setStyleSheet(cp.styleStatusWarning)
#        if status_index == 2 : self.lab_status.setStyleSheet(cp.styleStatusAlarm)
        #self.lab_status.setText('Status: ' + list_of_states[status_index] + msg)
#        self.lab_status.setText(msg)


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIRun ()
    widget.show()
    app.exec_()

#-----------------------------
