
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRange...
#
#------------------------------------------------------------------------

"""Range setting GUI

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
class GUIRange ( QtGui.QWidget ) :
    """Range setting GUI
    @see BaseClass
    @see OtherClass
    """

    def __init__ (self, parent=None, str_from=None, str_to=None, txt_from='valid from', txt_to='to') :

        QtGui.QWidget.__init__(self, None)
        self.parent = parent

        if txt_from == '' :
            self.setGeometry(10, 25, 140, 40)
            self.use_lab_from = False
        else :
            self.setGeometry(10, 25, 200, 40)
            self.use_lab_from = True

        self.setWindowTitle('Range setting GUI')

        self.setFrame()

        self.setParams(str_from, str_to)

        self.txt_from = txt_from
        if self.use_lab_from : self.lab_from = QtGui.QLabel(txt_from)
        self.lab_to         = QtGui.QLabel(txt_to)
        self.edi_from       = QtGui.QLineEdit  ( self.str_from )
        self.edi_to         = QtGui.QLineEdit  ( self.str_to )

        self.setEdiValidators()

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

        # cp.guirange = self # DO NOT REGISTER THIS OBJECT! There may be many instances in the list of runs...


    #-------------------
    # Private methods --
    #-------------------

    def setEdiValidators(self):
        self.edi_from.setValidator(QtGui.QIntValidator(0,9999,self))
        self.edi_to  .setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[1-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-9][0-9][0-9][0-9]|end$"),self))
        #self.edi_to  .setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]\\d{0,3}|end$"),self))


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
        if self.str_to == 'end' : return True

        if int(self.str_from) > int(self.str_to) :
            msg  = 'Begin number %s exceeds the end number %s' % (self.str_from, self.str_to)
            msg += '\nRANGE SEQUENCE SHOULD BE FIXED !!!!!!!!'
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


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        #self.frame.setGeometry(self.rect())
        #print 'GUIRange resizeEvent: %s' % str(self.size())
        pass


    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        # cp.guirange = None 


#    def run( self ) :
#        self.emitFieldIsChangedSignal()


    def emitFieldIsChangedSignal(self,msg) :
        self.emit(QtCore.SIGNAL('update(QString)'), msg)
        #print msg

  
    def onEdiFrom(self):
        logger.debug('onEdiFrom', __name__ )
        txt = str( self.edi_from.text() )
        if txt == self.str_from : return # if text has not changed
        self.str_from = txt
        msg = 'Set the range from "%s"' % self.str_from
        logger.info(msg, __name__ )
        self.setStyleButtons()
        self.emitFieldIsChangedSignal('from:%s'%self.str_from)


    def onEdiTo(self):
        logger.debug('onEdiTo', __name__ )
        txt = str( self.edi_to.text() )
        if txt == self.str_to : return # if text has not changed
        self.str_to = txt
        msg = 'Set the range up to "%s"' % self.str_to
        logger.info(msg, __name__ )
        self.setStyleButtons()
        self.emitFieldIsChangedSignal('to:%s'%self.str_to)


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


    def setParams(self, str_from=None, str_to=None) :
        self.str_from = str_from if str_from is not None else '0'
        self.str_to   = str_to   if str_to is not None else 'end'


    def resetFields(self, str_from=None, str_to=None) :
        """Interface method resetting the range fields to default"""
        self.setParams(str_from, str_to)
        self.setFields()


    def setFields(self) :
        self.edi_from.setText(self.str_from)
        self.edi_to  .setText(self.str_to)
        self.setStyleButtons()


    def getRange(self) :
        """Interface method returning range string, for example '123-end' """
        if self.statusButtonsIsGood() :
            return '%d-%s' % ( int(self.str_from),
                                   self.str_to.lstrip('0') )
        else :
            return '%d-%d' % ( int(self.str_from), int(self.str_from) )

        #return self.str_from + '-' + self.str_to

#-----------------------------

if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIRange(None,'0','end','')
    #ex  = GUIRange(None,'0','end')
    ex.move(10,25)
    ex.show()
    app.exec_()

#-----------------------------
