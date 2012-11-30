#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIFiles...
#
#------------------------------------------------------------------------

"""GUI sets path to files"""

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
from GUIDark                import *
from GUIFlatField           import *
from GUIBlamish             import *
from GUIWorkResDirs         import *
from Logger                 import logger
from BatchJobPedestals      import bjpeds

#---------------------
#  Class definition --
#---------------------
class GUIFiles ( QtGui.QWidget ) :
    """GUI sets path to files"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(1, 1, 800, 300)
        self.setWindowTitle('Files')
        self.setFrame()

        self.lab_title  = QtGui.QLabel     ('Files')
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

        self.list_file_types = ['Dark run', 'Flat field', 'Blamish', 'Data', 'Conf.pars', 'Work/Results']
        self.makeTabBar()
        self.guiSelector()

        self.vbox = QtGui.QVBoxLayout()   
        #cp.guiworkresdirs = GUIWorkResDirs()
        #self.vbox.addWidget(cp.guiworkresdirs)
        self.vbox.addWidget(self.lab_title)
        self.vbox.addWidget(self.tab_bar)
        self.vbox.addLayout(self.hboxW)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)

        self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose )
        self.connect( self.but_save,  QtCore.SIGNAL('clicked()'), self.onSave )
        self.connect( self.but_show,  QtCore.SIGNAL('clicked()'), self.onShow )

        self.showToolTips()
        self.setStyle()

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
        #self.frame.setVisible(False)

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
        self.setFixedHeight(330)
        
    def makeTabBar(self,mode=None) :
        #if mode != None : self.tab_bar.close()
        self.tab_bar = QtGui.QTabBar()

        #Uses self.list_file_types
        self.ind_tab_dark = self.tab_bar.addTab( self.list_file_types[0] )
        self.ind_tab_flat = self.tab_bar.addTab( self.list_file_types[1] )
        self.ind_tab_blam = self.tab_bar.addTab( self.list_file_types[2] )
        self.ind_tab_data = self.tab_bar.addTab( self.list_file_types[3] )
        self.ind_tab_conf = self.tab_bar.addTab( self.list_file_types[4] )
        self.ind_tab_work = self.tab_bar.addTab( self.list_file_types[5] )

        self.tab_bar.setTabTextColor(self.ind_tab_dark, QtGui.QColor('green'))
        self.tab_bar.setTabTextColor(self.ind_tab_flat, QtGui.QColor('red'))
        self.tab_bar.setTabTextColor(self.ind_tab_blam, QtGui.QColor('gray'))
        self.tab_bar.setTabTextColor(self.ind_tab_data, QtGui.QColor('blue'))
        self.tab_bar.setTabTextColor(self.ind_tab_conf, QtGui.QColor('magenta'))
        self.tab_bar.setTabTextColor(self.ind_tab_work, QtGui.QColor('gray'))
        self.tab_bar.setShape(QtGui.QTabBar.RoundedNorth)

        #self.tab_bar.setTabEnabled(1, False)
        self.tab_bar.setTabEnabled(3, False)
        
        logger.info(' make_tab_bar - set mode: ' + cp.ana_type.value(), __name__)

        try    :
            tab_index = self.list_file_types.index(cp.current_file_tab.value())
        except :
            tab_index = 3
            cp.current_file_tab.setValue(self.list_file_types[tab_index])
        self.tab_bar.setCurrentIndex(tab_index)

        self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.onTabBar)


    def guiSelector(self):

        try    : self.gui_win.close()
        except : pass

        try    : del self.gui_win
        except : pass

        if cp.current_file_tab.value() == self.list_file_types[0] :
            self.gui_win = GUIDark(self)
            self.setStatus(0, 'Status: processing for pedestals')
            
        if cp.current_file_tab.value() == self.list_file_types[1] :
            self.gui_win = GUIFlatField(self)
            self.setStatus(0, 'Status: set file for flat field')

        if cp.current_file_tab.value() == self.list_file_types[2] :
            self.gui_win = GUIBlamish(self)
            self.setStatus(0, 'Status: set file for blamish mask')

#        if cp.current_file_tab.value() == self.list_file_types[3] :
#            self.gui_win = GUIData(self)
#            self.setStatus(0, 'Status: processing for data')

        if cp.current_file_tab.value() == self.list_file_types[4] :
            self.gui_win = GUIConfigParameters(self)
            self.setStatus(0, 'Status: set file for config. pars.')

        if cp.current_file_tab.value() == self.list_file_types[5] :
            self.gui_win = GUIWorkResDirs(self)
            self.setStatus(0, 'Status: set work and result dirs.')

        self.gui_win.setFixedHeight(180)
        self.hboxW.addWidget(self.gui_win)

    def onTabBar(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        cp.current_file_tab.setValue( tab_name )
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
        
        try    : del cp.guifiles # GUIFiles
        except : pass # silently ignore

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def onSave(self):
        logger.debug('onSave', __name__)
        cp.saveParametersInFile( cp.fname_cp.value() )

    def onShow(self):
        logger.debug('onShow - is not implemented yet...', __name__)


    def on_off_gui_dark(self,but):
        logger.debug('on_off_gui_dark', __name__)
        self.tab_bar.setCurrentIndex(0)
        if bjpeds.status_for_pedestals() : but.setStyleSheet(cp.styleButtonGood)
        else                             : but.setStyleSheet(cp.styleButtonBad)

#        try :
#            cp.guidark.close()
#            but.setStyleSheet(cp.styleButtonBad)
#        except : # AttributeError: #NameError 
#            cp.guidark = GUIDark()
#            cp.guidark.setParent(self)
#            cp.guidark.move(self.pos().__add__(QtCore.QPoint(20,82))) # open window with offset w.r.t. parent
#            cp.guidark.show()
#            but.setStyleSheet(cp.styleButtonGood)


    def on_off_gui_flat(self,but):
        logger.debug('on_off_gui_flat', __name__)
        self.tab_bar.setCurrentIndex(1)


    def on_off_gui_data(self,but):
        logger.debug('on_off_gui_data', __name__)
        self.tab_bar.setCurrentIndex(2)


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
    widget = GUIFiles ()
    widget.show()
    app.exec_()

#-----------------------------
