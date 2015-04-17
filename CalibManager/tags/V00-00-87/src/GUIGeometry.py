#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIGeometry...
#
#------------------------------------------------------------------------

"""GUI sets path to files"""

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
from GUIMetrology           import *
from GUIAlignment           import *
from Logger                 import logger

#from BatchJobPedestals      import bjpeds

#---------------------
#  Class definition --
#---------------------
class GUIGeometry ( QtGui.QWidget ) :
    """GUI with tabs for calibration of geometry"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(1, 1, 600, 200)
        self.setWindowTitle('Geometry Calibration')
        self.setFrame()

        self.lab_title  = QtGui.QLabel     ('Geometry Calibration')
        self.lab_status = QtGui.QLabel     ('Status: ')
        self.but_close  = QtGui.QPushButton('&Close') 
        self.but_save   = QtGui.QPushButton('&Save') 
        self.but_show   = QtGui.QPushButton('Show &Image') 

        self.hboxW = QtGui.QHBoxLayout()
        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.lab_status)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.but_close)
        self.hboxB.addWidget(self.but_save)
        self.hboxB.addWidget(self.but_show )

        self.list_tab_titles = [ 'Metrology'
                                ,'Alignment'
                               ]
        self.makeTabBar()
        self.guiSelector()

        self.vbox = QtGui.QVBoxLayout()   
        #cp.guiworkresdirs = GUIWorkResDirs()
        #self.vbox.addWidget(cp.guiworkresdirs)
        self.vbox.addWidget(self.lab_title)
        self.vbox.addWidget(self.tab_bar)
        self.vbox.addLayout(self.hboxW)
        self.vbox.addStretch(1)     
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)

        self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose )
        self.connect( self.but_save,  QtCore.SIGNAL('clicked()'), self.onSave )
        self.connect( self.but_show,  QtCore.SIGNAL('clicked()'), self.onShow )

        self.showToolTips()
        self.setStyle()

        cp.guigeometry = self

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #msg = 'Edit field'
        self.but_close .setToolTip('Close this window.')
        self.but_save  .setToolTip('Save all current configuration parameters.')
        self.but_show  .setToolTip('Show ...')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        self.          setStyleSheet (cp.styleBkgd)
        self.but_close.setStyleSheet (cp.styleButton)
        self.but_save .setStyleSheet (cp.styleButton)
        self.but_show .setStyleSheet (cp.styleButton)

        self.lab_title.setStyleSheet (cp.styleTitleBold)
        self.lab_title .setAlignment(QtCore.Qt.AlignCenter)
        #self.setMinimumWidth (600)
        #self.setMaximumWidth (700)
        #self.setMinimumHeight(300)
        #self.setMaximumHeight(400)
        #self.setFixedWidth (700)
        #self.setFixedHeight(400)
        #self.setFixedHeight(330)
        #self.setFixedSize(550,350)
        #self.setFixedSize(600,360)
        self.setMinimumSize(600,360)
        #self.setMinimumSize(750,760)

        #self.lab_status.setVisible(False)
        self.but_close .setVisible(False)
        self.but_save  .setVisible(False)
        self.but_show  .setVisible(False)


    def makeTabBar(self,mode=None) :
        #if mode is not None : self.tab_bar.close()
        self.tab_bar = QtGui.QTabBar()

        #Uses self.list_tab_titles
        self.ind_tab_0 = self.tab_bar.addTab( self.list_tab_titles[0] )
        self.ind_tab_1 = self.tab_bar.addTab( self.list_tab_titles[1] )

        self.tab_bar.setTabTextColor(self.ind_tab_0, QtGui.QColor('magenta'))
        self.tab_bar.setTabTextColor(self.ind_tab_1, QtGui.QColor('magenta'))
        self.tab_bar.setShape(QtGui.QTabBar.RoundedNorth)

        #self.tab_bar.setTabEnabled(1, False)
        #self.tab_bar.setTabEnabled(2, False)
        #self.tab_bar.setTabEnabled(3, False)
        #self.tab_bar.setTabEnabled(4, False)
        
        try    :
            tab_index = self.list_tab_titles.index(cp.current_geometry_tab.value())
        except :
            tab_index = 1
            cp.current_geometry_tab.setValue(self.list_tab_titles[tab_index])
        self.tab_bar.setCurrentIndex(tab_index)

        logger.debug(' make_tab_bar - set mode: ' + cp.current_geometry_tab.value(), __name__)

        self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.onTabBar)


    def guiSelector(self):

        try    : self.gui_win.close()
        except : pass

        try    : del self.gui_win
        except : pass

        if cp.current_geometry_tab.value() == self.list_tab_titles[0] :
            self.gui_win = GUIMetrology(self)
            self.setStatus(0, 'Status: work with %s' % self.list_tab_titles[0])
            self.gui_win.setFixedHeight(200)
            
        if cp.current_geometry_tab.value() == self.list_tab_titles[1] :
            #self.gui_win = QtGui.QTextEdit('') # GUIConfigFile(self)
            self.gui_win = GUIAlignment(self)
            self.setStatus(0, 'Status: work with %s' % self.list_tab_titles[1])
            self.gui_win.setFixedHeight(300)

        #self.gui_win.setFixedHeight(180)
        #self.gui_win.setFixedHeight(600)
        self.hboxW.addWidget(self.gui_win)
        self.gui_win.setVisible(True)


    def onTabBar(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        cp.current_geometry_tab.setValue( tab_name )
        logger.info(' ---> selected tab: ' + str(tab_ind) + ' - open GUI to work with: ' + tab_name, __name__)
        self.guiSelector()


    def setParent(self,parent) :
        self.parent = parent


    def resizeEvent(self, e):
        logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())
        #print __name__ + ' config: self.size():', self.size()
        #self.setMinimumSize( self.size().width(), self.size().height()-40 )


    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent: new pos:' + str(self.position), __name__)
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

        #try    : cp.guimain.butFiles.setStyleSheet(cp.styleButton)
        #except : pass

        try    : self.gui_win.close()
        except : pass

        try    : self.tab_bar.close()
        except : pass
        
        #try    : del cp.guifiles # GUIGeometry
        #except : pass # silently ignore

        cp.guigeometry = None


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onSave(self):
        logger.debug('onSave', __name__)
        cp.saveParametersInFile( cp.fname_cp.value() )


    def onShow(self):
        logger.debug('onShow - is not implemented yet...', __name__)


    def setStatus(self, status_index=0, msg=''):
        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : self.lab_status.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : self.lab_status.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : self.lab_status.setStyleSheet(cp.styleStatusAlarm)

        #self.lab_status.setText('Status: ' + list_of_states[status_index] + msg)
        self.lab_status.setText(msg)


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIGeometry ()
    widget.show()
    app.exec_()

#-----------------------------
