#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIInsExpRun...
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

from ConfigParametersForApp import cp
import GlobalUtils          as     gu
from GUILogger              import *
from FileNameManager        import fnm

#---------------------
#  Class definition --
#---------------------
class GUIInsExpRun ( QtGui.QWidget ) :
    """GUI sets the configuration parameters for instrument, experiment, and run number"""

    char_expand    = u' \u25BE' # down-head triangle

    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.instr_dir      = cp.instr_dir
        self.instr_name     = cp.instr_name
        self.exp_name       = cp.exp_name
        self.run_number     = cp.str_run_number

        self.setGeometry(100, 50, 600, 30)
        self.setWindowTitle('Instrument Experiment Run')
        self.setFrame()
 
        self.list_of_instr  = ['AMO', 'SXR', 'XPP', 'XCS', 'CXI', 'MEC']
        self.list_of_exp    = None
        self.list_of_run    = None

        self.titIns  = QtGui.QLabel('Ins:')
        self.titExp  = QtGui.QLabel('Exp:')
        self.titRun  = QtGui.QLabel('Run:')

        self.butIns  = QtGui.QPushButton( self.instr_name.value() + self.char_expand )
        self.butExp  = QtGui.QPushButton( self.exp_name.value()   + self.char_expand )
        self.butRun  = QtGui.QPushButton( self.run_number.value() + self.char_expand )

        self.ediFile = QtGui.QLineEdit  ( str(fnm.path_to_xtc_files_for_run()) )
        self.ediFile .setReadOnly( True ) 

        self.butIns .setMaximumWidth(50)
        self.butExp .setMaximumWidth(90)
        self.butRun .setMaximumWidth(70)
        self.ediFile.setMinimumWidth(330)
        
        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.titIns)
        self.hbox.addWidget(self.butIns)
        self.hbox.addStretch(1)     
        self.hbox.addWidget(self.titExp)
        self.hbox.addWidget(self.butExp)
        self.hbox.addStretch(1)     
        self.hbox.addWidget(self.titRun)
        self.hbox.addWidget(self.butRun)
        self.hbox.addWidget(self.ediFile)

        self.setLayout(self.hbox)

        #self.connect( self.ediExp,     QtCore.SIGNAL('editingFinished ()'), self.processEdiExp )
        self.connect( self.butIns,     QtCore.SIGNAL('clicked()'),          self.onButIns  )
        self.connect( self.butExp,     QtCore.SIGNAL('clicked()'),          self.onButExp )
        self.connect( self.butRun,     QtCore.SIGNAL('clicked()'),          self.onButRun )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.butIns  .setToolTip('Select the instrument name from the pop-up menu.')
        self.butExp  .setToolTip('Select the experiment name from the directory list.')
        self.butRun  .setToolTip('Sets the run number from the file name,\nselected from the list of xtc files.')
        self.ediFile .setToolTip('Use buttons to change the xtc file name.')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        #self.setStyleSheet(cp.styleYellow)
        self.titIns  .setStyleSheet (cp.styleLabel)
        self.titExp  .setStyleSheet (cp.styleLabel)
        self.titRun  .setStyleSheet (cp.styleLabel)
        self.setStyleButtons()
        

    def setStyleButtons(self):
        if self.instr_name.value() == 'None' : self.butIns.setStyleSheet(cp.styleButtonBad)
        else                                 : self.butIns.setStyleSheet(cp.styleDefault)

        if self.exp_name.value()   == 'None' : self.butExp.setStyleSheet(cp.styleButtonBad)
        else                                 : self.butExp.setStyleSheet(cp.styleDefault)

        if self.run_number.value() == 'None' : self.butRun.setStyleSheet(cp.styleButtonBad)
        else                                 : self.butRun.setStyleSheet(cp.styleDefault)

        if self.run_number.value() == 'None' : self.ediFile.setStyleSheet(cp.styleButtonBad)
        else                                 : self.ediFile.setStyleSheet(cp.styleEditInfo)
 
    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp
            del cp.guiinsexprun# GUIInsExpRun
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


    def onButIns(self):
        #print 'onButIns'
        item_selected = gu.selectFromListInPopupMenu(self.list_of_instr)
        if item_selected is None : return            # selection is cancelled
        if item_selected == self.instr_name.value() : return # selected the same item  

        self.setIns(item_selected)
        self.setExp('None')
        self.setRun('None')
        self.setStyleButtons()

    def setIns(self, txt='None'):
        self.instr_name.setValue( txt )
        self.butIns.setText( txt + self.char_expand )
        #print ' ---> selected instrument:', txt


    def onButExpV1(self):
        #print 'onButExpV1'
        dir  = self.instr_dir.value() + '/' + self.instr_name.value()

        path = str( QtGui.QFileDialog.getExistingDirectory(self,'Select experiment',dir) )
        path1, name = os.path.split(path)

        if path1==self.instr_dir.value() and name==self.instr_name.value() :
            print 'The experiment is unchanged: ' + self.exp_name.value()
            return

        if path1 != dir :
            print 'WARNING! Wrong directory: ' + path1
            print 'Select the experiment being in the directory: ' + dir
            return

        print 'Set experiment: ' + name
        self.exp_name.setValue(name)
        self.butExp.setText(name + self.char_expand)        


    def onButExp(self):
        #print 'onButExp'
        dir = self.instr_dir.value() + '/' + self.instr_name.value()
        #print 'dir =', dir
        if self.list_of_exp is None : self.list_of_exp=os.listdir(dir)
        item_selected = gu.selectFromListInPopupMenu(self.list_of_exp)
        if item_selected is None : return          # selection is cancelled
        if item_selected == self.exp_name.value() : return # selected the same item 

        self.setExp(item_selected)
        self.setRun('None')
        self.setStyleButtons()


    def setExp(self, txt='None'):
        self.exp_name.setValue(txt)
        self.butExp.setText( txt + self.char_expand)
        if txt == 'None' : self.list_of_exp = None
        #print 'Set experiment: ' + txt


    def onButRunV1(self):
        #print 'onButRunV1'
        dir = fnm.path_to_xtc_dir()
        path = str( QtGui.QFileDialog.getOpenFileName(self,'Select experiment',dir) )
        path1, fname = os.path.split(path) # where fname looks like: e170-r0003-s00-c00.xtc
        print 'Returned path and fname =', path1, fname

        if path1=='' and fname=='' :
            print 'The run number is unchanged: ' + self.run_number.value()
            return

        if path1 != dir :
            print 'WARNING! Wrong directory: ' + path1
            print 'Select the run being in the directory: ' + dir
            return

        print 'Set run from file name: ' + fname
        expnum, runnum, stream, chunk, ext = gu.parse_xtc_file_name(fname) # Parse: e170-r0003-s00-c00.xtc
        self.setRun(runnum)
        self.setStyleButtons()


    def onButRun(self):
        #print 'onButRun'
        self.list_of_run = fnm.get_list_of_xtc_files()
        #if self.list_of_run is None : self.list_of_run=os.listdir(dir)

        item_selected = gu.selectFromListInPopupMenu(self.list_of_run)
        if item_selected is None : return            # selection is cancelled
        #if item_selected == self.run_number.value() : return # selected the same item 

        fname = item_selected
        #print 'Set run from file name: ' + fname

        expnum, runnum, stream, chunk, ext = gu.parse_xtc_file_name(fname) # Parse: e170-r0003-s00-c00.xtc
        self.setRun(runnum)
        self.setStyleButtons()


    def setRun(self, txt='None'):
        self.run_number.setValue(txt)
        self.butRun.setText(txt + self.char_expand)        
        if txt == 'None' : self.list_of_run = None
        self.setFile()


    def setFile(self):
        path = str(fnm.path_to_xtc_files_for_run())
        self.ediFile.setText(path)
        logger.info('set path to xtc file: ' + path, __name__ )



#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIInsExpRun ()
    widget.show()
    app.exec_()

#-----------------------------
