#!/usr/bin/python2.4
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module XtcBrowserMain...
#
#------------------------------------------------------------------------

"""GUI interface to xtc files

Main GUI. 

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: XtcBrowserMain 2011-01-27 14:15:00 ofte $

@author Ingrid Ofte
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 0 $"
# $Source$

#-----------------------------
# Imports for other modules --
#-----------------------------
import sys, os, random

from    PyQt4 import QtGui, QtCore
from  XtcScanner import XtcScanner
from XtcPyanaControl import XtcPyanaControl

#----------------------------------
# Local non-exported definitions --
#----------------------------------


#------------------------
# Exported definitions --
#------------------------


#---------------------
#  Class definition --
#---------------------

class XtcBrowserMain (QtGui.QMainWindow) :
    """Gui Main Window
    
    Gui Main Widget for browsing Xtc files.
    
    @see OtherClass
    """

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor.

        Description
        """
        QtGui.QMainWindow.__init__(self)
    
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setStyleSheet("QWidget {background-color: #FFFFFF }")

        self.setWindowTitle("LCLS Xtc Event Browser")
        self.setWindowIcon(QtGui.QIcon('XtcEventBrowser/src/lclsLogo.gif'))

        # list of current files
        self.filenames = []

        # make only one instance of these
        self.scanner = None
        self.pyanactrl = None
        
        self.__create_main_frame()

    def about(self):
        progname = os.path.basename(sys.argv[0])
        progversion = "0.1"
        QtGui.QMessageBox.about(self, "About %s" % os.path.basename(sys.argv[0]),
u"""%(prog)s version %(version)s
GUI interface to analysis of xtc files.

This software was developed for the LCLS project at 
SLAC National Accelerator Center. If you use all or
part of it, please give an appropriate acknowledgment.

2011   Ingrid Ofte
"""   % {"prog": progname, "version": progversion})
        
                
        
    def __create_main_frame(self):

        # Icon
        self.pic = QtGui.QLabel(self)
        self.pic.setPixmap( QtGui.QPixmap('XtcEventBrowser/src/lclsLogo.gif'))

        # menu
        self.help_menu = QtGui.QMenu('&Help', self)
        self.menuBar().addMenu(self.help_menu)
        self.help_menu.addAction('&About',self.about)

        # --- File section ---

        # Label showing currently selected files
        self.currentfiles = QtGui.QLabel(self);
        self.__update_currentfiles()

        # Button: open file browser
        self.fbrowser_button = QtGui.QPushButton("&File Browser")
        self.connect(self.fbrowser_button, QtCore.SIGNAL('clicked()'), self.__file_browser )

        # Button: clear file list
        self.fclear_button = QtGui.QPushButton("&Clear File List")
        self.connect(self.fclear_button, QtCore.SIGNAL('clicked()'), self.__clear_file_list )

        # Line edit: enter file name
        self.lineedit = QtGui.QLineEdit("")
        self.lineedit.setMinimumWidth(200)
        self.connect(self.lineedit, QtCore.SIGNAL('returnPressed()'), self.__add_file )

        # Button: add file from line edit
        self.addfile_button = QtGui.QPushButton("&Add")
        self.connect(self.addfile_button, QtCore.SIGNAL('clicked()'), self.__add_file )
             

        # --- Scan section --- 
        self.scan_button = QtGui.QPushButton("&Scan File(s)")
        self.connect(self.scan_button, QtCore.SIGNAL('clicked()'), self.__scan_files )
        
        self.qscan_button = QtGui.QPushButton("&Quick Scan")
        self.connect(self.qscan_button, QtCore.SIGNAL('clicked()'), self.__scan_files_quick )

        # Quit application
        self.quit_button = QtGui.QPushButton("&Quit")
        self.connect(self.quit_button, QtCore.SIGNAL('clicked()'), QtGui.qApp, QtCore.SLOT('quit()') )
        
        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setFocus()
        

        # holds checkboxes, pyana configuration and pyana run-button
        #self.det_selector = QtGui.QVBoxLayout()
        
        ### layout ###
        
        # header
        h0 = QtGui.QHBoxLayout()
        h0.addWidget( self.pic )
        h0.setAlignment( self.pic, QtCore.Qt.AlignLeft )

        # files
        fgroup = QtGui.QGroupBox("File section")
        

        v1 = QtGui.QVBoxLayout()
        v1.addWidget( self.fbrowser_button )
        v1.addWidget( self.fclear_button )

        v2 = QtGui.QVBoxLayout()
        v2.addWidget( self.currentfiles )
        v2.setAlignment( self.currentfiles, QtCore.Qt.AlignTop )

        h1 = QtGui.QHBoxLayout()
        h1.addLayout(v1)
        h1.addLayout(v2)

        h2 = QtGui.QHBoxLayout()
        h2.addWidget( self.lineedit )
        h2.addWidget( self.addfile_button )

        H1 = QtGui.QVBoxLayout()
        H1.addLayout(h1)
        H1.addLayout(h2)
        fgroup.setLayout(H1)
        
        # action
        v3 = QtGui.QVBoxLayout()
        v3.addWidget( self.qscan_button )
        v3.setAlignment(self.qscan_button, QtCore.Qt.AlignLeft )
        v3.addWidget( self.scan_button )
        v3.setAlignment(self.scan_button, QtCore.Qt.AlignLeft )

        h4 = QtGui.QHBoxLayout()
        h4.addLayout(v3)
        
        # Pyana
        #h5 = QtGui.QHBoxLayout()
        #h5.addLayout( self.det_selector )

        # Quit
        h6 = QtGui.QHBoxLayout()
        h6.addWidget( self.quit_button )
        h6.setAlignment( self.quit_button, QtCore.Qt.AlignRight )

        l = QtGui.QVBoxLayout(self.main_widget)
        l.addLayout(h0)
        #l.addLayout(h1)
        #l.addLayout(h2)
        l.addWidget(fgroup)
        
        l.addLayout(h4)
        #l.addLayout(h5)
        #l.addLayout(self.det_layout)
        l.addLayout(h6)

        self.setCentralWidget(self.main_widget)



    #-------------------
    #  Public methods --
    #-------------------

    def add_file(self, filename):
        self.filenames.append(filename)
        # add the last file opened to the line dialog
        self.lineedit.setText( str(filename) )
        self.__update_currentfiles()

    #--------------------
    #  Private methods --
    #--------------------

    def __file_browser(self):
        """Opens a Qt File Dialog

        Opens a Qt File dialog which allows user
        to select one or more xtc files. The file names
        are added to a list holding current files.
        """
        selectedfiles = QtGui.QFileDialog.getOpenFileNames( \
            self, "Select File","/reg/d/psdm/","xtc files (*.xtc)")
        
        # convert QStringList to python list of strings
        file = ''
        for file in selectedfiles :
            if self.filenames.count( str(file) )==0 :
                self.filenames.append( str(file) )

        # add the last file opened to the line dialog
        self.lineedit.setText( str(file) )
        self.__update_currentfiles()


        


    def __add_file(self):
        """Add a file to list of files
        
        Add a file to list of files. Input from lineedit
        """
        if self.filenames.count( str(self.lineedit.text()))==0:
            self.filenames.append(str(self.lineedit.text()))
            self.__update_currentfiles()
            
    def __clear_file_list(self):
        """Empty the file list
        
        """
        self.filenames = []
        self.lineedit.setText("")
        self.__update_currentfiles()

        self.checks = []
        self.checkboxes = []
        
            
    def __update_currentfiles(self):
        """Update status text (list of files)
        """
        status = "Currently selected file(s):       (%d)\n " % len(self.filenames )

        for filename in self.filenames :
            addline = filename+"\n"
            status+=addline
                
        self.currentfiles.setText(status)


    def __scan_files(self):
        """Scan xtc files

        Run XtcScanner to scan the files.
        When scan is done, open a new Gui Widget
        to configure pyana / plotting
        """
        if self.scanner is None:
            self.scanner = XtcScanner()
                
        print self.filenames
        self.scanner.setFiles(self.filenames)
        self.scanner.setOption({'ndatagrams':-1}) # all
        self.scanner.scan()

        #self.__add_selector()
        if self.pyanactrl is None : 
            self.pyanactrl = XtcPyanaControl()
            self.pyanactrl.add_selector( self.scanner.devices )
            self.pyanactrl.set_files(self.filenames)

    def __scan_files_quick(self):
        """Quick scan of xtc files

        Run XtcScanner to scan the first 1000 datagrams of the file(s).
        When scan is done, open a new Gui Widget
        to configure pyana / plotting
        """
        if self.scanner is None:
            self.scanner = XtcScanner()

        print self.filenames
        self.scanner.setFiles(self.filenames)
        self.scanner.setOption({'ndatagrams':1000})
        self.scanner.scan()

        #self.__add_selector()
        if self.pyanactrl is None : 
            self.pyanactrl = XtcPyanaControl()
            self.pyanactrl.add_selector( self.scanner.devices )
            self.pyanactrl.set_files(self.filenames)
        





#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    qApp = QtGui.QApplication(sys.argv)
    mainw = XtcBrowserMain()
    mainw.show()
    sys.exit(qApp.exec_())

    

