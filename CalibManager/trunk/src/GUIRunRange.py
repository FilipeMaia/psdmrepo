
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

    def __init__ (self, parent=None, str_run_from=None, str_run_to=None, txt_from='valid from', txt_to='to') :

        QtGui.QWidget.__init__(self, parent)

        if txt_from == '' :
            self.setGeometry(10, 25, 140, 40)
            self.use_lab_from = False
        else :
            self.setGeometry(10, 25, 200, 40)
            self.use_lab_from = True

        self.setWindowTitle('Run range setting GUI')

        self.setFrame()

        self.setParams(str_run_from, str_run_to)

        self.txt_from = txt_from
        if self.use_lab_from : self.lab_from = QtGui.QLabel(txt_from)
        self.lab_to         = QtGui.QLabel(txt_to)
        self.edi_from       = QtGui.QLineEdit  ( self.str_run_from )
        self.edi_to         = QtGui.QLineEdit  ( self.str_run_to )

        self.edi_from.setValidator(QtGui.QIntValidator(0,9999,self))
        self.edi_to  .setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[1-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-9][0-9][0-9][0-9]|end$"),self))
        #self.edi_to  .setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]\\d{0,3}|end$"),self))
 
        self.hboxC = QtGui.QHBoxLayout()
        self.hboxC.addStretch(1)     
        if self.use_lab_from : self.hboxC.addWidget( self.lab_from )
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

        # cp.guirunrange = self # DO NOT REGISTER THIS OBJECT! There may be many instances in the list of runs...


    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        self.edi_from  .setToolTip('Enter run number in range [0,9999]')
        self.edi_to    .setToolTip('Enter run number in range [1,9999] or "end"')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        self.          setStyleSheet(cp.styleBkgd)

        if self.use_lab_from :
            self.setMinimumSize(200,32)
        else :
            self.setMinimumSize(100,32)

        #self.setFixedHeight(40)
        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))

        self.edi_from.setFixedWidth(40)
        self.edi_to  .setFixedWidth(40)

        self.edi_from .setAlignment (QtCore.Qt.AlignRight)
        self.edi_to   .setAlignment (QtCore.Qt.AlignRight)

        if self.use_lab_from : self.lab_from  .setStyleSheet(cp.styleLabel)
        self.lab_to.setStyleSheet(cp.styleLabel)
 
        self.setStyleButtons()


    def statusButtonsIsGood(self):
        if self.str_run_to == 'end' : return True

        if int(self.str_run_from) > int(self.str_run_to) :
            msg  = 'Begin run number %s exceeds the end run number %s' % (self.str_run_from, self.str_run_to)
            msg += '\nRUN RANGE SEQUENCE SHOULD BE FIXED !!!!!!!!'
            logger.warning(msg, __name__ )            
            return False

        return True


    def setStyleButtons(self):
        if self.statusButtonsIsGood() :
            self.edi_from.setStyleSheet(cp.styleEdit)
            self.edi_to  .setStyleSheet(cp.styleEdit)
        else :
            self.edi_from.setStyleSheet(cp.styleEditBad)
            self.edi_to  .setStyleSheet(cp.styleEditBad)


    def setParams(self, str_run_from, str_run_to) :
        self.str_run_from = str_run_from if str_run_from is not None else '0'
        self.str_run_to   = str_run_to   if str_run_to is not None else 'end'


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
        # cp.guirunrange = None 


#    def run( self ) :
#        self.emitFieldIsChangedSignal()


    def emitFieldIsChangedSignal(self,msg) :
        self.emit(QtCore.SIGNAL('update(QString)'), msg)
        #print msg

  
    def onEdiFrom(self):
        logger.debug('onEdiFrom', __name__ )
        txt = str( self.edi_from.displayText() )        
        if txt == self.str_run_from : return # if text has not changed
        self.str_run_from = txt
        msg = 'Set the run validity range from %s' % self.str_run_from
        logger.info(msg, __name__ )
        self.setStyleButtons()
        self.emitFieldIsChangedSignal('from:%s'%self.str_run_from)


    def onEdiTo(self):
        logger.debug('onEdiTo', __name__ )
        txt = str( self.edi_to.displayText() )        
        if txt == self.str_run_to : return # if text has not changed
        self.str_run_to = txt
        msg = 'Set the run validity range up to %s' % self.str_run_to
        logger.info(msg, __name__ )
        self.setStyleButtons()
        self.emitFieldIsChangedSignal('to:%s'%self.str_run_to)


    def setFieldsEnable(self, is_enabled=True):
        """Interface method enabling/disabling the edit fields"""
        if is_enabled :
            self.setStyleButtons()
            #self.edi_from.setStyleSheet(cp.styleEdit)
            #self.edi_to  .setStyleSheet(cp.styleEdit)
        else :
            self.edi_from.setStyleSheet(cp.styleEditInfo)
            self.edi_to  .setStyleSheet(cp.styleEditInfo)

        self.edi_from.setEnabled(is_enabled) 
        self.edi_to  .setEnabled(is_enabled) 

        self.edi_from .setReadOnly(not is_enabled)
        self.edi_to   .setReadOnly(not is_enabled)


    def resetFields(self) :
        """Interface method resetting the run range fields to default"""
        self.setParams()
        self.edi_from  .setText(self.str_run_from)
        self.edi_to    .setText(self.str_run_to)
        self.setStyleButtons()


    def getRunRange(self) :
        """Interface method returning run range string, for example '123-end' """
        if self.statusButtonsIsGood() :
            return '%d-%s' % ( int(self.str_run_from),
                                   self.str_run_to.lstrip('0') )
        else :
            return '%d-%d' % ( int(self.str_run_from), int(self.str_run_from) )

        #return self.str_run_from + '-' + self.str_run_to

#-----------------------------

if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIRunRange(None,'0','end','')
    #ex  = GUIRunRange(None,'0','end')
    ex.move(10,25)
    ex.show()
    app.exec_()

#-----------------------------
