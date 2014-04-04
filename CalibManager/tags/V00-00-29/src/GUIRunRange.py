
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRunRange...
#
#------------------------------------------------------------------------

"""Run range setting GUI

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
#  Imports
#--------------------------------
import sys
import os
from PyQt4 import QtGui, QtCore

from Logger import logger
from ConfigParametersForApp import cp

#---------------------
#  Class definition --
#---------------------
class GUIRunRange ( QtGui.QWidget ) :
    """Run range setting GUI
    @see BaseClass
    @see OtherClass
    """

    def __init__ (self, parent=None) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(10, 25, 250, 40)
        self.setWindowTitle('Run range setting GUI')

        self.setFrame()

        self.setParams()
 
        self.lab_from       = QtGui.QLabel('valid from run')
        self.lab_to         = QtGui.QLabel('to')
        self.edi_from       = QtGui.QLineEdit  ( self.str_run_from )
        self.edi_to         = QtGui.QLineEdit  ( self.str_run_to )

        self.edi_from.setValidator(QtGui.QIntValidator(0,9999,self))
        self.edi_to  .setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]\\d{0,3}|end$"),self))
 
        self.hboxC = QtGui.QHBoxLayout() 
        self.hboxC.addWidget( self.lab_from )
        self.hboxC.addWidget( self.edi_from )
        self.hboxC.addWidget( self.lab_to )
        self.hboxC.addWidget( self.edi_to )
        self.hboxC.addStretch(1)     

        self.vboxW = QtGui.QVBoxLayout() 
        self.vboxW.addStretch(1)
        self.vboxW.addLayout( self.hboxC ) 
        self.vboxW.addStretch(1)
        
        self.setLayout(self.vboxW)

        self.connect( self.edi_from,   QtCore.SIGNAL('editingFinished()'), self.onEdiFrom )
        self.connect( self.edi_to,     QtCore.SIGNAL('editingFinished()'), self.onEdiTo )
  
        self.showToolTips()
        self.setStyle()

        cp.guirunrange = self


    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        self.edi_from  .setToolTip('Enter run number in range [0,9999]')
        self.edi_to    .setToolTip('Enter run number in range [0,9999] or "end"')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        self.          setStyleSheet(cp.styleBkgd)
        
        self.setMinimumSize(225,32)
        #self.setFixedHeight(40)
        self.setContentsMargins (QtCore.QMargins(0,-9,0,-9))

        self.edi_from.setFixedWidth(40)
        self.edi_to  .setFixedWidth(40)

        self.lab_from  .setStyleSheet(cp.styleLabel)
        self.lab_to    .setStyleSheet(cp.styleLabel)
 
        self.setStyleButtons()


    def setStyleButtons(self):
        pass


    def setParams(self) :
        self.str_run_from = '0'
        self.str_run_to   = 'end'


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        #self.frame.setGeometry(self.rect())
        #print 'GUIRunRange resizeEvent: %s' % str(self.size())
        pass


    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        cp.guirunrange = None


    def onEdiFrom(self):
        logger.debug('onEdiFrom', __name__ )
        self.str_run_from = str( self.edi_from.displayText() )        
        msg = 'Set the run validity range from %s' % self.str_run_from
        logger.info(msg, __name__ )


    def onEdiTo(self):
        logger.debug('onEdiTo', __name__ )
        self.str_run_to = str( self.edi_to.displayText() )        
        msg = 'Set the run validity range up to %s' % self.str_run_to
        logger.info(msg, __name__ )


    def setFieldsEnable(self, is_enable=True):
        """Interface method enabling/disabling the edit fields"""
        if is_enable :
            self.edi_from.setStyleSheet(cp.styleEdit)
            self.edi_to  .setStyleSheet(cp.styleEdit)
        else :
            self.edi_from.setStyleSheet(cp.styleEditInfo)
            self.edi_to  .setStyleSheet(cp.styleEditInfo)

        self.edi_from.setEnabled(is_enable) 
        self.edi_to  .setEnabled(is_enable) 


    def resetFields(self) :
        """Interface method resetting the run range fields to default"""
        self.setParams()
        self.edi_from  .setText(self.str_run_from)
        self.edi_to    .setText(self.str_run_to)
        self.setStyleButtons()


    def getRunRange(self) :
        """Interface method returning run range string, for example '123-end' """
        return self.str_run_from + '-' + self.str_run_to

#-----------------------------

if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIRunRange()
    ex.move(10,25)
    ex.show()
    app.exec_()

#-----------------------------
