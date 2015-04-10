#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIFileBrowser ...
#
#------------------------------------------------------------------------

"""GUI for File Browser"""

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
from Logger                 import logger
from FileNameManager        import fnm
import GlobalUtils          as     gu

#---------------------
#  Class definition --
#---------------------
class GUIFileBrowser ( QtGui.QWidget ) :
    """GUI File Browser"""

    def __init__ ( self, parent=None, list_of_files=['Empty list'], selected_file=None, is_editable=True) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 900, 500)
        self.setWindowTitle('GUI File Browser')
        self.setFrame()
        try : self.setWindowIcon(cp.icon_browser)
        except : pass

        self.box_txt    = QtGui.QTextEdit()
 
        self.tit_status = QtGui.QLabel('Status:')
        self.tit_file   = QtGui.QLabel('File:')
        self.but_close  = QtGui.QPushButton('Close') 
        self.but_save   = QtGui.QPushButton('Save As') 

        self.is_editable = is_editable

        self.box_file      = QtGui.QComboBox( self ) 
        self.setListOfFiles(list_of_files)

        self.hboxM = QtGui.QHBoxLayout()
        self.hboxM.addWidget( self.box_txt )

        self.hboxF = QtGui.QHBoxLayout()
        self.hboxF.addWidget(self.tit_file)
        self.hboxF.addWidget(self.box_file)

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.tit_status)
        self.hboxB.addStretch(4)     
        self.hboxB.addWidget(self.but_save)
        self.hboxB.addWidget(self.but_close)

        self.vbox  = QtGui.QVBoxLayout()
        #self.vbox.addWidget(self.tit_title)
        self.vbox.addLayout(self.hboxF)
        self.vbox.addLayout(self.hboxM)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)
        
        self.connect( self.but_save,  QtCore.SIGNAL('clicked()'), self.onSave )
        self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose )
        self.connect( self.box_file, QtCore.SIGNAL('currentIndexChanged(int)'), self.onBox  )
 
        self.startFileBrowser(selected_file)

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #self           .setToolTip('This GUI is intended for run control and monitoring.')
        self.but_close .setToolTip('Close this window.')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def setStyle(self):
        self.           setStyleSheet (cp.styleBkgd)
        self.tit_status.setStyleSheet (cp.styleTitle)
        self.tit_file  .setStyleSheet (cp.styleTitle)
        self.tit_file  .setFixedWidth (25)
        self.tit_file  .setAlignment  (QtCore.Qt.AlignRight)
        self.box_file  .setStyleSheet (cp.styleButton) 
        self.but_save  .setStyleSheet (cp.styleButton)
        self.but_close .setStyleSheet (cp.styleButton)
        self.box_txt   .setReadOnly   (not self.is_editable)
        self.box_txt   .setStyleSheet (cp.styleWhiteFixed) 


    def setListOfFiles(self, list):
        self.list_of_files  = ['Click on this box and select file from pop-up-list']
        self.list_of_files += list
        self.box_file.clear()
        self.box_file.addItems(self.list_of_files)
        self.box_file.setCurrentIndex( 0 )


    def setParent(self,parent) :
        self.parent = parent


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())


    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        #self.saveLogTotalInFile() # It will be saved at closing of GUIMain

        try    : cp.guimain.butFBrowser.setStyleSheet(cp.styleButtonBad)
        except : pass

        try    : cp.guidark.but_browse.setStyleSheet(cp.styleButtonBad)
        except : pass

        self.box_txt.close()

        try    : del cp.guifilebrowser # GUIFileBrowser
        except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onSave(self):
        logger.debug('onSave', __name__)
        path = gu.get_save_fname_through_dialog_box(self, self.fname, 'Select file to save', filter='*.txt')
        if path is None or path == '' : return
        text = str(self.box_txt.toPlainText())
        logger.info('Save in file:\n'+text, __name__)
        f=open(path,'w')
        f.write( text )
        f.close() 
 

    def onBox(self):
        self.fname = str( self.box_file.currentText() )
        logger.info('onBox - selected file: ' + self.fname, __name__)

        self.list_of_supported = 'cfg', 'log', 'txt', 'txt-tmp' 
        self.str_of_supported = ''
        for ext in self.list_of_supported : self.str_of_supported += ' ' + ext


        if self.list_of_files.index(self.fname) == 0 :
            self.setStatus(0, 'Waiting for file selection...')
            self.box_txt.setText('Click on file-box and select the file from pop-up list...')

        elif os.path.lexists(self.fname) :
            ext = os.path.splitext(self.fname)[1].lstrip('.')

            if ext in self.list_of_supported :
                self.box_txt.setText(gu.get_text_file_content(self.fname))
                self.setStatus(0, 'Status: enjoy browsing the selected file...')

            else :
                self.box_txt.setText('Sorry, but this browser supports text files with extensions:' +
                                     self.str_of_supported + '\nTry to select another file...')
                self.setStatus(1, 'Status: ' + ext + '-file is not supported...')

        else :
            self.box_txt.setText( 'Selected file is not avaliable...\nTry to select another file...')
            self.setStatus(2, 'Status: WARNING: FILE IS NOT AVAILABLE!')


    def startFileBrowser(self, selected_file=None) :
        logger.debug('Start the GUIFileBrowser.',__name__)
        self.setStatus(0, 'Waiting for file selection...')

        if selected_file is not None and selected_file in self.list_of_files :
            index = self.list_of_files.index(selected_file)
            self.box_file.setCurrentIndex( index )

        elif len(self.list_of_files) == 2 :
            self.box_file.setCurrentIndex( 1 )
            #self.onBox()      
        else :
            self.box_file.setCurrentIndex( 0 )
        #self.box_txt.setText('Click on file-box and select the file from pop-up list...')


    def appendGUILog(self, msg='...'):
        self.box_txt.append(msg)
        scrol_bar_v = self.box_txt.verticalScrollBar() # QScrollBar
        scrol_bar_v.setValue(scrol_bar_v.maximum()) 

        
    def setStatus(self, status_index=0, msg=''):
        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : self.tit_status.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : self.tit_status.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : self.tit_status.setStyleSheet(cp.styleStatusAlarm)

        #self.tit_status.setText('Status: ' + list_of_states[status_index] + msg)
        self.tit_status.setText(msg)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIFileBrowser ()
    widget.show()
    app.exec_()

#-----------------------------
