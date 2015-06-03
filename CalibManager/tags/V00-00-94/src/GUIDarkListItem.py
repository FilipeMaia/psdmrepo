# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDarkListItem ...
#
#------------------------------------------------------------------------

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
import unicodedata

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)
#from time import time

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersForApp import cp
from Logger                 import logger
import GlobalUtils          as     gu
from FileNameManager        import fnm

from GUIDarkListItemRun     import *
from GUIDarkListItemAdd     import *
#from GUIAnyFilesStatusTable import *

#---------------------
#  Class definition --
#---------------------
class GUIDarkListItem ( QtGui.QWidget ) :
    """GUI sets the source dark run number, validity range, and starts calibration of pedestals"""

    char_expand    = cp.char_expand.strip()
    char_shrink    = cp.char_shrink.strip()
    #char_expand    = ' v'
    #char_expand    = cp.char_expand
    #char_expand    = u'\u25BC' # solid down-head triangle
    #char_shrink    = u'\u25B2' # solid up-head triangle
    #char_shrink    = u'\u25B6' # solid right-head triangle
    #char_expand    = u'\u25BD' # open down-head triangle
    #char_shrink    = u'\u25B7' # open right-head triangle
    #char_expand    = 'V' # solid down-head triangle
    #char_shrink    = '>' # solid right-head triangle

    def __init__ ( self, parent=None, str_run_num='0000', run_type='Type N/A', comment='', xtc_in_dir=True) :

        #self.t0_sec = time()

        QtGui.QWidget.__init__(self, parent)

        self.parent = parent

        #self.setGeometry(100, 100, 740, 77)
        self.setGeometry(100, 100, 740, 35)
        self.setWindowTitle('GUI Dark List Item')
        #try : self.setWindowIcon(cp.icon_help)
        #except : pass
        self.setFrame()

        self.list_of_runs    = None

        self.str_run_num = str_run_num # cp.str_run_number

        #self.str_run_number = cp.str_run_number # cp.str_run_number
        #self.calib_dir      = cp.calib_dir
        #self.det_name       = cp.det_name
        
        self.but_expand_shrink = QtGui.QPushButton(self.char_expand)

        self.gui_add = None
        self.gui_run = GUIDarkListItemRun(self, str_run_num, run_type, comment, xtc_in_dir)
        self.hboxTT = QtGui.QHBoxLayout()
        self.hboxTT.addSpacing(5)     
        self.hboxTT.addWidget(self.but_expand_shrink)
        self.hboxTT.addWidget(self.gui_run)
        self.hboxTT.addStretch(1)     

        self.hboxW = QtGui.QHBoxLayout()
        #self.hboxWW = QtGui.QHBoxLayout()
        ##self.hboxWW.addStretch(1)     
        #self.hboxWW.addSpacing(5)     
        #self.hboxWW.addLayout(self.hboxW)
        #self.hboxWW.addStretch(1)     

        self.vbox = QtGui.QVBoxLayout()
        #self.vbox.addWidget(self.gui_run)
        self.vbox.addLayout(self.hboxTT)
        self.vbox.addLayout(self.hboxW)
        self.vbox.addStretch(1)     

        self.setLayout(self.vbox)

        self.connect( self.but_expand_shrink, QtCore.SIGNAL('clicked()'), self.onButExpandShrink  )
 
        self.showToolTips()

        self.setStyle()

        #print '\nConsumed time (sec) =', time()-self.t0_sec


    def showToolTips(self):
        self.but_expand_shrink.setToolTip('Expand/shrink additional information space for this run.')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def setFieldsEnabled(self, is_enabled=True):
        logger.debug('Set fields enabled: %s' %  is_enabled, __name__)
        #self.lab_rnum .setEnabled(is_enabled)
        self.but_expand_shrink.setEnabled(is_enabled)
        self.gui_run.setFieldsEnabled(is_enabled)
        

    def setStyle(self):
        self.setMinimumSize(740,35)
        self.setStyleSheet (cp.styleBkgd)
        #self.setStyleSheet(cp.styleGreenish)
        #self.setStyleSheet (cp.styleYellowish)

        self.but_expand_shrink.setFixedSize(20,22)
        #self.but_expand_shrink.setStyleSheet(cp.styleButtonGood)
        self.but_expand_shrink.setStyleSheet(cp.stylePink)

        #self.lab_from.setStyleSheet(cp.styleLabel)
        #self.lab_rnum .setFixedWidth(80)        
        #self.edi_from .setAlignment (QtCore.Qt.AlignRight)
        #if self.edi_from.isReadOnly() : self.edi_from.setStyleSheet (cp.styleEditInfo)

        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())
        #self.box_txt.setGeometry(self.contentsRect())
        #print 'GUIDarkListItem resizeEvent: %s' % str(self.size())

        
    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        #self.saveLogTotalInFile() # It will be saved at closing of GUIMain

        try    : self.gui_add.close()
        except : pass

        try    : self.gui_run.close()
        except : pass


    def onClose(self):
        logger.info('onClose', __name__)
        self.close()


    def setStatusMessage(self):
        if cp.guistatus is None : return
        #cp.guistatus.setStatusMessage(msg)


    def onButExpandShrink(self):
        #logger.debug('onButExpandShrink', __name__)
        but = self.but_expand_shrink

        if but.text() == self.char_expand :
            #print 'Click on expand button'
            but.setText(self.char_shrink)
            self.onClickExpand()
            self.parent.onItemExpand(self)

        elif but.text() == self.char_shrink :
            #print 'Click on shrink button'
            but.setText(self.char_expand)
            self.onClickShrink()
            self.parent.onItemShrink(self)

        self.updateButtons()
        

    def onClickExpand(self):
        logger.debug('onClickExpand', __name__)
        #self.gui_add = QtGui.QLabel('Additional information')  
        self.gui_add = GUIDarkListItemAdd(self, self.str_run_num) 
        self.hboxW.addWidget(self.gui_add)

        self.setFixedHeight( self.height() + self.gui_add.height() )

        #self.gui_add.setMinimumHeight( self.height() + self.gui_add.height() )
        #self.hboxW.addStretch(1)     
        #self.gui_add.setStyleSheet(cp.styleYellowish)
        #self.gui_run.setStyleSheet(cp.styleYellowish)



    def onClickShrink(self):
        logger.debug('onClickShrink', __name__)

        #self.gui_add.setVisible(False)
        self.hboxW.removeWidget(self.gui_add)

        try    : self.gui_add.close()
        except : pass

        try    : del self.gui_add
        except : pass

        self.gui_add = None

        self.gui_run.setStyleSheet(cp.styleBkgd)
        self.setFixedHeight(35)
        #self.resize(740,35)
        #self.updateGeometry()


    def getHeight(self):
        logger.debug('getHeight', __name__)
        h = self.gui_run.height()
        if self.gui_add is not None :
            h += self.gui_add.getHeight()
        return h + 10


    def updateButtons(self, str_run_type='', comment='', xtc_in_dir=True) :
        #logger.info('update', __name__)
        self.gui_run.updateButtons(str_run_type, comment, xtc_in_dir)


    def getStrRunNum(self) :
        return self.str_run_num
       
    def getRunNum(self) :
        return int(self.str_run_num)
       
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIDarkListItem(parent=None, str_run_num='0016')
    w.setFieldsEnabled(True)
    w.show()
    app.exec_()

#-----------------------------
