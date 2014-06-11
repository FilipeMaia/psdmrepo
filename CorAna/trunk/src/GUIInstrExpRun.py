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

from ConfigParametersCorAna import confpars as cp

#---------------------

def xtc_fname_parser_helper( part, prefix ) :    
    """In parsing the xtc file name, this function extracts the string after expected prefix, i.e. 'r0123' -> '0123'"""
    if len(part)>1 and part[0] == prefix :
        try :
            return part[1:]
        except :
            pass
    return None

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

        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Instrument Experiment Run')
        self.setFrame()
        self.setStyle()
 
        self.list_of_instr  = ['XCS', 'AMO', 'CXI', 'MEC', 'SXR', 'XPP']
        self.char_expand    = u' \u25BE' # down-head triangle
        self.instr_dir      = cp.instr_dir.value()
        self.instr_name     = cp.instr_name.value()
        self.exp_name       = cp.exp_name.value()
        self.run_number     = cp.str_run_number.value()
        self.run_number_dark= cp.str_run_number_dark.value()

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

        self.connect( self.ediExp,     QtCore.SIGNAL('editingFinished ()'), self.processEdiExp )
        self.connect( self.ediRun,     QtCore.SIGNAL('editingFinished ()'), self.processEdiRun )
        self.connect( self.ediRunDark, QtCore.SIGNAL('editingFinished ()'), self.processEdiRunDark )
        self.connect( self.butInstr,   QtCore.SIGNAL('clicked()'),          self.processInstr  )
        self.connect( self.butExp,     QtCore.SIGNAL('clicked()'),          self.processButExp )
        self.connect( self.butRun,     QtCore.SIGNAL('clicked()'),          self.processButRun )
        self.connect( self.butRunDark, QtCore.SIGNAL('clicked()'),          self.processButRunDark )

        self.showToolTips()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        msg_edi = 'WARNING: whatever you edit may be incorrect...\nIt is recommended to use the '

        self.butInstr  .setToolTip('Select the instrument name from the pop-up menu.')
        self.butExp    .setToolTip('Select the experiment name from the directory list.')
        self.butRun    .setToolTip('Sets the run number from the file name,\nselected from the list of xtc files.')
        self.butRunDark.setToolTip('Sets the dark run number from the file name,\nselected from the list of xtc files.')
        self.ediExp    .setToolTip( msg_edi + '"Exp:" button.')
        self.ediRun    .setToolTip( msg_edi + '"Run:" button.')
        self.ediRunDark.setToolTip( msg_edi + '"Dark:" button.')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setStyleSheet(cp.styleYellow)

    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp
            del cp.guiinstrexprun# GUIInstrExpRun
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
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def processEdiRun(self):
        print 'WARNING! Non-editable field.'

    def processEdiExp(self):
        print 'WARNING! Non-editable field.'

    def processEdiRunDark(self):
        print 'WARNING! Non-editable field.'


    def processInstr(self):
        print 'processInstr'
        action_selected = self.popupMenuInstr.exec_(QtGui.QCursor.pos())
        if action_selected is None : return
        self.instr_name = action_selected.text()
        cp.instr_name.setValue( self.instr_name )
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
        cp.exp_name.setValue(name)
        self.ediExp.setText(name)        


    def processButRun(self):
        print 'processButRun'

        dir  = self.instr_dir + '/' + self.instr_name + '/' + self.exp_name + '/xtc'
        path = str( QtGui.QFileDialog.getOpenFileName(self,'Select experiment',dir) )
        path1, fname = os.path.split(path) # where fname looks like: e170-r0003-s00-c00.xtc
        print 'Returned path and fname =', path1, fname

        if path1=='' and fname=='' :
            print 'The run number is unchanged: ' + self.run_number
            return

        if path1 != dir :
            print 'WARNING! Wrong directory: ' + path1
            print 'Select the run being in the directory: ' + dir
            return

        print 'Set run from file name: ' + fname
        self.parse_xtc_file_name(fname) # Parse: e170-r0003-s00-c00.xtc
        self.run_number = self._runnum
        cp.str_run_number.setValue(self.run_number)
        self.ediRun.setText(self.run_number)        


    def processButRunDark(self):
        print 'processButRunDark'

        dir  = self.instr_dir + '/' + self.instr_name + '/' + self.exp_name + '/xtc'
        path = str( QtGui.QFileDialog.getOpenFileName(self,'Select experiment',dir) )
        path1, fname = os.path.split(path) # where fname looks like: e170-r0003-s00-c00.xtc
        print 'Returned path and fname =', path1, fname

        if path1=='' and fname=='' :
            print 'The run number is unchanged: ' + self.run_number_dark
            return

        if path1 != dir :
            print 'WARNING! Wrong directory: ' + path1
            print 'Select the run being in the directory: ' + dir
            return

        print 'Set run from file name: ' + fname
        self.parse_xtc_file_name(fname) # Parse: e170-r0003-s00-c00.xtc
        self.run_number_dark = self._runnum
        cp.str_run_number.setValue(self.run_number_dark)
        self.ediRunDark.setText(self.run_number_dark)        


    def parse_xtc_file_name(self, fname):
        """Parse the file name like e170-r0003-s00-c00.xtc"""
        name, self._ext = os.path.splitext(fname) # i.e. ('e167-r0015-s00-c00', '.xtc')
        parts = name.split('-') # it gives parts = ('e167', 'r0015', 's00', 'c00')

        self._expnum = None
        self._runnum = None
        self._stream = None
        self._chunk  = None

        parts = map( xtc_fname_parser_helper, parts, ['e', 'r', 's', 'c'] )

        if None not in parts :
            self._expnum = parts[0]
            self._runnum = parts[1]
            self._stream = parts[2]
            self._chunk  = parts[3]

        print 'e,r,s,c,ext:', self._expnum, self._runnum, self._stream, self._chunk, self._ext

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIInstrExpRun ()
    widget.show()
    app.exec_()

#-----------------------------
