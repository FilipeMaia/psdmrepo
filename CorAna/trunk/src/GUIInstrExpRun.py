#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIInstrExpRun...
#
#------------------------------------------------------------------------

"""GUI sets the instrument, experiment, and run number for signal and dark data"""

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

import ConfigParametersCorAna as cp

#---------------------
#  Class definition --
#---------------------
class GUIInstrExpRun ( QtGui.QWidget ) :
    """GUI sets the instrument, experiment, and run number for signal and dark data"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 500, 50)
        self.setWindowTitle('Instrument Experiment Run')
        self.setFrame()

        self.list_of_instr  = ['XCS', 'AMO', 'CXI', 'MEC', 'SXR', 'XPP']
        self.char_expand    = u' \u25BE' # down-head triangle
        self.instr_dir      = cp.confpars.instr_dir.value()
        self.instr_name     = cp.confpars.instr_name.value()
        self.exp_name       = cp.confpars.exp_name.value()
        self.run_number     = cp.confpars.str_run_number.value()
        self.run_number_dark= cp.confpars.str_run_number_dark.value()

        self.titInstr    = QtGui.QLabel('Instr:')
        self.butInstr    = QtGui.QPushButton( self.instr_name + self.char_expand )
        self.butExp      = QtGui.QPushButton( "Exp:" )
        self.ediExp      = QtGui.QLineEdit  ( self.exp_name )  
        self.butRun      = QtGui.QPushButton( "Run:" )
        self.ediRun      = QtGui.QLineEdit  ( self.run_number )  
        self.butRunDark  = QtGui.QPushButton( "Dark:" )
        self.ediRunDark  = QtGui.QLineEdit  ( self.run_number_dark )  

        self.ediExp    .setReadOnly( True ) 
        self.ediRun    .setReadOnly( True ) 
        self.ediRunDark.setReadOnly( True ) 

        self.popupMenuInstr = QtGui.QMenu()
        for instr in self.list_of_instr :
            self.popupMenuInstr.addAction( instr )
         

        self.butInstr   .setMaximumWidth(50)
        self.butExp     .setMaximumWidth(40)
        self.butRun     .setMaximumWidth(40)
        self.butRunDark .setMaximumWidth(40)
        self.ediRun     .setMaximumWidth(40)
        self.ediRunDark .setMaximumWidth(40)
        self.ediExp     .setMinimumWidth(70)
        self.ediExp     .setMaximumWidth(80)

        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.titInstr)
        self.hbox.addWidget(self.butInstr)
        self.hbox.addStretch(1)     
        self.hbox.addWidget(self.butExp)
        self.hbox.addWidget(self.ediExp)
        self.hbox.addStretch(1)     
        self.hbox.addWidget(self.butRun)
        self.hbox.addWidget(self.ediRun)
        self.hbox.addStretch(1)     
        self.hbox.addWidget(self.butRunDark)
        self.hbox.addWidget(self.ediRunDark)

        self.setLayout(self.hbox)

        self.connect( self.butInstr,   QtCore.SIGNAL('clicked()'),          self.processInstr  )
        self.connect( self.ediExp,     QtCore.SIGNAL('editingFinished ()'), self.processEdiExp )
        self.connect( self.butExp,     QtCore.SIGNAL('clicked()'),          self.processButExp )
        self.connect( self.ediRun,     QtCore.SIGNAL('editingFinished ()'), self.processEdiRun )
        self.connect( self.butRun,     QtCore.SIGNAL('clicked()'),          self.processButRun )
        self.connect( self.ediRunDark, QtCore.SIGNAL('editingFinished ()'), self.processEdiRunDark )
        self.connect( self.butRunDark, QtCore.SIGNAL('clicked()'),          self.processButRunDark )

        self.showToolTips()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.butInstr  .setToolTip('Select the instrument name from the pop-up menu.')
        self.butExp    .setToolTip('Select the experiment name from the directory list.')
        self.ediExp    .setToolTip('Edit the name of the experiment.\nWARNING: whatever you edit may be incorrect!\nIt is recommended to use the "Exp:" button.')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp.confpars
            del cp.confpars.guiinstrexprun# GUIInstrExpRun
        except AttributeError:
            pass # silently ignore

    def processClose(self):
        #print 'Close button'
        self.close()

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #print 'moveEvent' 
        pass
#        cp.confpars.posGUIMain = (self.pos().x(),self.pos().y())

    def processInstr(self):
        print 'processInstr'
        action_selected = self.popupMenuInstr.exec_(QtGui.QCursor.pos())
        if action_selected is None : return
        self.instr_name = action_selected.text()
        cp.confpars.instr_name.setValue( self.instr_name )
        self.butInstr.setText( self.instr_name + self.char_expand )
        print ' ---> selected instrument:', self.instr_name

    def processButExp(self):
        print 'processButExp'
        dir  = self.instr_dir + '/' + self.instr_name
        #path = QtGui.QFileDialog.getOpenFileName(self,'Select experiment',dir)
        path = str( QtGui.QFileDialog.getExistingDirectory(self,'Select experiment',dir) )
        path1, name = os.path.split(path)

        if path1==self.instr_dir and name==self.instr_name :
            print 'The experiment is unchanged: ' + self.exp_name
            return

        if path1 != dir :
            print 'WARNING! Wrong directory: ' + path1
            print 'Select the experiment being in the directory: ' + dir
            return

        print 'Set experiment: ' + name
        self.exp_name = name
        cp.confpars.exp_name.setValue(name)
        self.ediExp.setText(name)        


    def processEdiExp(self):
        print 'WARNING! Non-editable field.'

    def processButRun(self):
        print 'processButRun'

    def processEdiRun(self):
        print 'WARNING! Non-editable field.'

    def processButRunDark(self):
        print 'processButRunDark'

    def processEdiRunDark(self):
        print 'WARNING! Non-editable field.'


#    def processRead(self):
#        print 'Read'
#        cp.confpars.readParametersFromFile( self.getFileNameFromEditField() )
#        self.fnameEdit.setText( cp.confpars.fname_cp.value() )
#        #self.parent.fnameEdit.setText( cp.confpars.fname_cp.value() )
#        #self.refreshGUIWhatToDisplay()

#    def processWrite(self):
#        print 'Write'
#        cp.confpars.saveParametersInFile( self.getFileNameFromEditField() )

#    def processDefault(self):
#        print 'Set default values of configuration parameters.'
#        cp.confpars.setDefaultValues()
#        self.fnameEdit.setText( cp.confpars.fname_cp.value() )
#        #self.refreshGUIWhatToDisplay()

#    def processBrowse(self):
#        print 'Browse'
#        self.path = self.getFileNameFromEditField()
#        self.dname,self.fname = os.path.split(self.path)
#        print 'dname : %s' % (self.dname)
#        print 'fname : %s' % (self.fname)
#        self.path = QtGui.QFileDialog.getOpenFileName(self,'Open file',self.dname)
#        self.dname,self.fname = os.path.split(str(self.path))

#        if self.dname == '' or self.fname == '' :
#            print 'Input directiry name or file name is empty... use default values'  
#        else :
#            self.fnameEdit.setText(self.path)
#            cp.confpars.fname_cp.setValue(self.path)

#    def processFileEdit(self):
#        print 'FileEdit'
#        self.path = self.getFileNameFromEditField()
#        cp.confpars.fname_cp.setValue(self.path)
#        dname,fname = os.path.split(self.path)
#        print 'Set dname : %s' % (dname)
#        print 'Set fname : %s' % (fname)

#    def getFileNameFromEditField(self):
#        return str( self.fnameEdit.displayText() )

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIInstrExpRun ()
    widget.show()
    app.exec_()

#-----------------------------
