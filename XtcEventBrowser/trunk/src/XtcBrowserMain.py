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

import XtcEventDisplay as display

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt

from xbplotter import draw_on

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

        QtCore.pyqtRemoveInputHook()
        # to avoid a problems with raw_input()
        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setStyleSheet("QWidget {background-color: #FFFFFF }")

        self.setWindowTitle("LCLS Xtc Event Browser")
        self.setWindowIcon(QtGui.QIcon('XtcEventBrowser/src/lclsLogo.gif'))

        self.filenames = []
        # list of current files

        # keep reference to these objects at all times, they know a lot...
        self.scanner = None
        self.pyanactrl = None
        
        self.create_main_frame()
        print "Welcome to Xtc Event Display!"

        
    def create_main_frame(self):

        # Icon
        self.pic = QtGui.QLabel(self)
        self.pic.setPixmap( QtGui.QPixmap('XtcEventBrowser/src/lclsLogo.gif'))

        # menu
        self.help_menu = QtGui.QMenu('&Help', self)
        self.menuBar().addMenu(self.help_menu)
        self.help_menu.addAction('&Documentation',self.documentation)
        self.help_menu.addAction('&About',self.about)

        # --- Scan section --- 
        self.scan_button = QtGui.QPushButton("&Scan File(s)")
        self.connect(self.scan_button, QtCore.SIGNAL('clicked()'), self.scan_files )
        self.scan_button.setDisabled(True)
        self.scan_label = QtGui.QLabel(self.scan_button)
        self.scan_label.setText("Scan all events")

        self.scan_enable_button = QtGui.QPushButton("&Enable")
        self.connect(self.scan_enable_button, QtCore.SIGNAL('clicked()'), self.scan_enable )
        
        self.qscan_button = QtGui.QPushButton("&Quick Scan")
        self.qscan_button.setDisabled(True)
        self.connect(self.qscan_button, QtCore.SIGNAL('clicked()'), self.scan_files_quick )
        self.nev_qscan = 200
        self.qscan_edit = QtGui.QLineEdit(str(self.nev_qscan))
        self.qscan_edit.setMaximumWidth(80)
        self.connect(self.qscan_edit, QtCore.SIGNAL('returnPressed()'), self.change_nev_qscan )

        self.qscan_label = QtGui.QLabel(self.qscan_button)
        self.qscan_label.setText("Scan the first %d events   " % self.nev_qscan)

        self.fileinfo = QtGui.QLabel(self)

        # --- File section ---

        # Label showing currently selected files
        self.currentfiles = QtGui.QLabel(self)
        self.update_currentfiles()

        # Button: open file browser
        self.fbrowser_button = QtGui.QPushButton("&File Browser...")
        self.connect(self.fbrowser_button, QtCore.SIGNAL('clicked()'), self.file_browser )

        # Button: clear file list
        self.fclear_button = QtGui.QPushButton("&Clear File List")
        self.connect(self.fclear_button, QtCore.SIGNAL('clicked()'), self.clear_file_list )

        # Line edit: enter file name
        self.lineedit = QtGui.QLineEdit("")
        self.lineedit.setMinimumWidth(200)
        self.connect(self.lineedit, QtCore.SIGNAL('returnPressed()'), self.add_file_from_lineedit )

        # Button: add file from line edit
        self.addfile_button = QtGui.QPushButton("&Add")
        self.connect(self.addfile_button, QtCore.SIGNAL('clicked()'), self.add_file_from_lineedit )
             

        # ---- Test section -------
        
        # Test matplotlib widget
        self.mpl_button = QtGui.QPushButton("&MatPlotLib")
        self.connect(self.mpl_button, QtCore.SIGNAL('clicked()'), self.makeplot )
        self.mpl2_button = QtGui.QPushButton("&MatPlotLib2")
        self.connect(self.mpl2_button, QtCore.SIGNAL('clicked()'), self.makeplot2 )

        # Quit application
        self.quit_button = QtGui.QPushButton("&Quit")
        #self.connect(self.quit_button, QtCore.SIGNAL('clicked()'), QtGui.qApp, QtCore.SLOT('quit()') )
        self.connect(self.quit_button, QtCore.SIGNAL('clicked()'), self.quit )
        
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
        v1.setAlignment( self.fbrowser_button, QtCore.Qt.AlignTop)
        v1.addWidget( self.fclear_button )
        v1.setAlignment( self.fclear_button, QtCore.Qt.AlignTop)

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
        
        # Scan
        sgroup = QtGui.QGroupBox("Scan section")

        hs0 = QtGui.QHBoxLayout()
        hs0.addWidget( self.qscan_button )
        hs0.addWidget( self.qscan_edit )
        hs0.addWidget( self.qscan_label )
        hs0.setAlignment( self.qscan_edit, QtCore.Qt.AlignLeft )
        hs1 = QtGui.QHBoxLayout()
        hs1.addWidget( self.scan_button )
        hs1.addWidget( self.scan_enable_button )
        hs1.addWidget( self.scan_label )
        hs2 = QtGui.QHBoxLayout()
        hs2.addWidget( self.fileinfo )
        
        v3 = QtGui.QVBoxLayout()
        v3.addLayout(hs0)
        v3.setAlignment(hs0, QtCore.Qt.AlignLeft)
        v3.addLayout(hs1)
        v3.setAlignment(hs1, QtCore.Qt.AlignLeft)
        v3.addLayout(hs2)
        #v3.addWidget( self.mpl_button )
        #v3.setAlignment(self.mpl_button, QtCore.Qt.AlignRight )
        #v3.addWidget( self.mpl2_button )
        #v3.setAlignment(self.mpl2_button, QtCore.Qt.AlignRight )

        h4 = QtGui.QHBoxLayout()
        h4.addLayout(v3)
        sgroup.setLayout(h4)
        
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
        l.addWidget(sgroup)
        
        #l.addLayout(h4)
        #l.addLayout(h5)
        #l.addLayout(self.det_layout)
        l.addLayout(h6)

        self.setCentralWidget(self.main_widget)



    #-------------------
    #  Public methods --
    #-------------------

    def quit(self):
        if self.pyanactrl is not None : 
            self.pyanactrl.quit_pyana()
        QtGui.qApp.closeAllWindows()


    #--------------------
    #  Private methods --
    #--------------------
    
    def file_browser(self):
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
        self.update_currentfiles()
        

    def add_file_from_cmd(self, filename):
        """Add file
        """
        if self.filenames.count(filename)==0:
            if os.path.isfile(filename) :
                self.filenames.append(filename)
                # add the last file opened to the line dialog
                self.lineedit.setText( str(filename) )
                self.update_currentfiles()
            

    def add_file_from_lineedit(self):
        """Add a file to list of files
        
        Add a file to list of files. Input from lineedit
        """
        filestring = str(self.lineedit.text())
        if self.filenames.count(filestring)==0:
            if os.path.isfile(filestring):
                self.filenames.append(filestring)
                self.update_currentfiles()
            
    def clear_file_list(self):
        """Empty the file list
        
        """
        self.filenames = []
        self.update_currentfiles()

        self.checks = []
        self.checkboxes = []
        
        if self.pyanactrl is not None :
            del self.pyanactrl
            self.pyanactrl = None
            
    def update_currentfiles(self):
        """Update text describing the list of current files
        """
        # number of files
        nfiles = len(self.filenames)
        status = "Currently selected:  %d file(s)  " % nfiles
        
        # total file size
        self.filesize = 0.0
        for filename in self.filenames :
            self.filesize += os.path.getsize(filename)
            
        filesizetxt = ""
        if nfiles > 0 :
            filesize = self.filesize/1024
            if filesize < 1024 :
                filesizetxt = "%.1fk" % (filesize)
            elif filesize < 1024**2 :
                filesizetxt = "%.1fM" % (filesize/1024)
            elif filesize < 1024**3 :
                filesizetxt = "%.1fG" % (filesize/1024**2)
            elif filesize < 1024**4 :
                filesizetxt = "%.1fT" % (filesize/1024**3)
            else :
                filesizetxt = "Big! "

            # if files, enable the buttons
            if self.qscan_button :
                self.qscan_button.setEnabled(True)
            if self.scan_button and self.scan_label :
                if self.filesize < 2*1024**3 :
                    self.scan_enable()
                    self.scan_label.setText("Scan all events (%s)"%filesizetxt)
                else :
                    self.scan_label.setText("Scan all events (%s!)"%filesizetxt)

        status+="\t %s \n" % filesizetxt
        for filename in self.filenames :
            addline = filename+"\n"
            status+=addline
        self.currentfiles.setText(status)
                


    def change_nev_qscan(self):
        self.nev_qscan = int(self.qscan_edit.text())
        self.qscan_label.setText("(Scan the first %d events)   "%self.nev_qscan)
        
    def scan_enable(self) :
        if self.scan_button :
            if self.scan_button.isEnabled() :
                self.scan_button.setDisabled(True)
                self.scan_enable_button.setText("Enable")
            else :
                self.scan_button.setEnabled(True)
                self.scan_enable_button.setText("Disable")


    def scan_files(self, quick=False):
        """Scan xtc files

        Run XtcScanner to scan the files.
        When scan is done, open a new Gui Widget
        to configure pyana / plotting
        """
        if self.scanner is None:
            self.scanner = XtcScanner()                
        self.scanner.setFiles(self.filenames)        
        if quick :
            self.scanner.setOption({'ndatagrams':self.nev_qscan})
        else :
            self.scanner.setOption({'ndatagrams':-1}) # all
        self.scanner.scan()

        # (re)make the pyana control object
        if self.pyanactrl: del self.pyanactrl
        self.pyanactrl = XtcPyanaControl()
                    
        self.pyanactrl.update(devices=self.scanner.devices.keys(),
                              epicsPVs=self.scanner.epicsPVs,
                              controls=self.scanner.controls,
                              moreinfo=self.scanner.moreinfo.values(),
                              filenames=self.filenames )

        if self.scan_button :
            self.scan_enable()
            
        fileinfo_text = "The scan found: \n     %d calib cycles (scan steps) "\
                        "for a total of %d L1Accepts (shots)"\
                        % (self.scanner.ncalib, sum(self.scanner.nevents) )
        if len(self.scanner.nevents) > 1 :
            fileinfo_text += ":\n     nShots[scanstep] = %s " % str(self.scanner.nevents)

        self.fileinfo.setText(fileinfo_text)


    def scan_files_quick(self):
        """Quick scan of xtc files
        """
        self.scan_files(quick=True)


    def documentation(self):        
        print "Documentation on Confluence"
        
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



    # ------------------
    # -- Experimental --
    # ------------------
    def on_draw(self):
        """ Redraws the figure
        """
        str = unicode(self.textbox.text())
        self.data = map(int, str.split())
        
        x = range(len(self.data))
        
        # clear the axes and redraw the plot anew
        #
        self.axes.clear()
        self.axes.grid(self.grid_cb.isChecked())
        
        self.axes.bar(
            left=x,
            height=self.data,
            width=self.slider.value() / 100.0,
            align='center',
            alpha=0.44,
            picker=5)
        
        self.canvas.draw()
        
        
    def makeplot(self):

        self.fig = plt.figure(200)
        axes = self.fig.add_subplot(111)
        axes.set_title("Hello MatPlotLib")
        
        plt.show()
        
        dark_image = np.load("pyana_cspad_average_image.npy")
        axim = plt.imshow( dark_image )#, origin='lower' )
        colb = plt.colorbar(axim,pad=0.01)
        
        plt.draw()
        
        print "Done drawing"
        
        axim = plt.imshow( dark_image[500:1000,1000:1500] )#, origin='lower' )
        
        return
 
    def makeplot2(self):
        number = 200
        print "number " , number
        self.fig = draw_on(number)
        return


        print "This does not work yet. Ignore"
        self.mpl_widget = QtGui.QWidget()
        self.dpi = 100
        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.mpl_widget)
        
        # Since we have only one plot, we can use add_axes
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = self.fig.add_subplot(111)
        #self.canvas.mpl_connect('pick_event', self.on_pick)

        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.mpl_widget)
        
        # Other GUI controls
        #
        self.textbox = QLineEdit()
        self.textbox.setMinimumWidth(200)
        self.connect(self.textbox, SIGNAL('editingFinished ()'), self.on_draw)
        
        self.draw_button = QPushButton("&Draw")
        self.connect(self.draw_button, SIGNAL('clicked()'), self.on_draw)
        
        self.grid_cb = QCheckBox("Show &Grid")
        self.grid_cb.setChecked(False)
        self.connect(self.grid_cb, SIGNAL('stateChanged(int)'), self.on_draw)
        
        slider_label = QLabel('Bar width (%):')
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(20)
        self.slider.setTracking(True)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.connect(self.slider, SIGNAL('valueChanged(int)'), self.on_draw)
        
        #
        # Layout with box sizers
        #
        hbox = QHBoxLayout()
        
        for w in [  self.textbox, self.draw_button, self.grid_cb,
                    slider_label, self.slider]:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)
            
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)

        self.mpl_widget.setLayout(vbox)
        self.setCentralWidget(self.mpl_widget)

        #myfig = display.XtcEventDisplay()
        #myfig.show()
        #"Showing figure?"


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    qApp = QtGui.QApplication(sys.argv)
    mainw = XtcBrowserMain()
    mainw.show()
    sys.exit(qApp.exec_())

    

