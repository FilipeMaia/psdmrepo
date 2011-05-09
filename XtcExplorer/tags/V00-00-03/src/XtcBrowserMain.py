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
import  subprocess 

from    PyQt4 import QtGui, QtCore
from  XtcScanner import XtcScanner

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

        self.checks = []
        self.checkboxes = []

        # make only one instance of the XtcScanner 
        self.scanner = None
        
        self.create_main_frame()



    def create_main_frame(self):
    
        self.pic = QtGui.QLabel(self)
        self.pic.setPixmap( QtGui.QPixmap('XtcEventBrowser/src/lclsLogo.gif'))

        self.status = QtGui.QLabel(self);
        self.__update_status()
        
        # buttons
        self.pyana_config = None
        self.pyana_button = None

        self.fbrowser_button = QtGui.QPushButton("&File Browser")
        self.connect(self.fbrowser_button, QtCore.SIGNAL('clicked()'), self.__file_browser )
        
        self.fclear_button = QtGui.QPushButton("&Clear File List")
        self.connect(self.fclear_button, QtCore.SIGNAL('clicked()'), self.__clear_file_list )

        self.lineedit = QtGui.QLineEdit("")
        self.lineedit.setMinimumWidth(200)
        self.connect(self.lineedit, QtCore.SIGNAL('returnPressed()'), self.__add_file )
        
        self.addfile_button = QtGui.QPushButton("&Add")
        self.connect(self.addfile_button, QtCore.SIGNAL('clicked()'), self.__add_file )
             
        self.scan_button = QtGui.QPushButton("&Scan File(s)")
        self.connect(self.scan_button, QtCore.SIGNAL('clicked()'), self.__scan_files )
        
        self.qscan_button = QtGui.QPushButton("&Quick Scan")
        self.connect(self.qscan_button, QtCore.SIGNAL('clicked()'), self.__scan_files_quick )

        # Quit application
        self.quit_button = QtGui.QPushButton("&Quit")
        self.connect(self.quit_button, QtCore.SIGNAL('clicked()'), self.__file_quit )
        
        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setFocus()
        

        # holds checkboxes, pyana configuration and pyana run-button
        self.det_selector = QtGui.QVBoxLayout()
        
        ### layout ###
        
        # header
        h0 = QtGui.QHBoxLayout()
        h0.addWidget( self.pic )
        h0.setAlignment( self.pic, QtCore.Qt.AlignLeft )

        # files
        v1 = QtGui.QVBoxLayout()
        v1.addWidget( self.fbrowser_button )
        v1.addWidget( self.fclear_button )

        v2 = QtGui.QVBoxLayout()
        v2.addWidget( self.status )
        v2.setAlignment( self.status, QtCore.Qt.AlignTop )

        h1 = QtGui.QHBoxLayout()
        h1.addLayout(v1)
        h1.addLayout(v2)

        h2 = QtGui.QHBoxLayout()
        h2.addWidget( self.lineedit )
        h2.addWidget( self.addfile_button )
        
        # action
        v3 = QtGui.QVBoxLayout()
        v3.addWidget( self.qscan_button )
        v3.setAlignment(self.qscan_button, QtCore.Qt.AlignLeft )
        v3.addWidget( self.scan_button )
        v3.setAlignment(self.scan_button, QtCore.Qt.AlignLeft )

        h4 = QtGui.QHBoxLayout()
        h4.addLayout(v3)
        
        # Pyana
        h5 = QtGui.QHBoxLayout()
        h5.addLayout( self.det_selector )

        # Quit
        h6 = QtGui.QHBoxLayout()
        h6.addWidget( self.quit_button )
        h6.setAlignment( self.quit_button, QtCore.Qt.AlignRight )

        l = QtGui.QVBoxLayout(self.main_widget)
        l.addLayout(h0)
        l.addLayout(h1)
        l.addLayout(h2)
        l.addLayout(h4)
        l.addLayout(h5)
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
        self.__update_status()

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
        self.__update_status()


    def __add_selector(self):
        """Draw a group of checkboxes to the GUI

        Draw a group of checkboxes to the GUI. 
        One checkbox for each Detector/Device found
        in the scan of xtc file
        """
        if self.scanner is None :
            print "Can't use selector before running the scan"
            return

        for d in sorted( self.scanner.devices.keys() ):
            clabel = d 
            if clabel.find("ProcInfo") >= 0 : continue
            if clabel.find("NoDetector") >= 0 : continue
            if self.checks.count( clabel )==0 :
                c = QtGui.QCheckBox(clabel)
                self.det_selector.addWidget(c)
                self.checkboxes.append(c)
                self.checks.append(clabel)

        if self.pyana_button is None: 
            self.pyana_button = QtGui.QPushButton("&Run pyana")
            self.connect(self.pyana_button, QtCore.SIGNAL('clicked()'), self.__run_pyana )
            self.det_selector.addWidget( self.pyana_button )
            self.det_selector.setAlignment( self.pyana_button, QtCore.Qt.AlignRight )
        


    def __add_file(self):
        """Add a file to list of files
        
        Add a file to list of files. Input from lineedit
        """
        if self.filenames.count( str(self.lineedit.text()))==0:
            self.filenames.append(str(self.lineedit.text()))
            self.__update_status()
            
    def __clear_file_list(self):
        """Empty the file list
        
        """
        self.filenames = []
        self.lineedit.setText("")
        self.__update_status()

        self.checks = []
        self.checkboxes = []
        
            
    def __update_status(self):
        """Update status text (list of files)
        """
        status = "Currently selected file(s):       (%d)\n " % len(self.filenames )

        for filename in self.filenames :
            addline = filename+"\n"
            status+=addline
                
        self.status.setText(status)


    def __scan_files(self):
        """Scan xtc files

        Run XtcScanner to scan the files
        """
        if self.scanner is None:
            self.scanner = XtcScanner()
                
        print self.filenames
        self.scanner.setFiles(self.filenames)
        self.scanner.setOption({'ndatagrams':-1}) # all
        self.scanner.scan()
        self.__add_selector()

    def __scan_files_quick(self):
        """Quick scan of xtc files

        Run XtcScanner to scan the first 1000 datagrams of the file(s)
        """
        if self.scanner is None:
            self.scanner = XtcScanner()

        print self.filenames
        self.scanner.setFiles(self.filenames)
        self.scanner.setOption({'ndatagrams':1000})
        self.scanner.scan()
        self.__add_selector()

    def __file_quit(self):
        """Close
        """
        self.close()

    def __write_config_script(self):

        nmodules = 0
        modules_to_run = []
        options_for_mod = []

        for box in self.checkboxes :
            if box.isChecked() :
                print "%s requested" % box.text() 

                # --- --- --- BLD --- --- ---
                if str(box.text()).find("BldInfo")>=0 :
                    index = None
                    try :
                        index = modules_to_run.index("XtcEventBrowser.pyana_bld")
                    except ValueError :
                        index = len(modules_to_run)
                        modules_to_run.append("XtcEventBrowser.pyana_bld")
                        options_for_mod.append([])
                    print "XtcEventBrowser.pyana_bld at ", index

                    if str(box.text()).find("EBeam")>=0 :
                        options_for_mod[index].append("\ndo_ebeam = True")
                    if str(box.text()).find("FEEGasDetEnergy")>=0 :
                        options_for_mod[index].append("\ndo_gasdetector = True")
                    if str(box.text()).find("PhaseCavity")>=0 :
                        options_for_mod[index].append("\ndo_phasecavity = True")

                # --- --- --- Ipimb --- --- ---
                if str(box.text()).find("Ipimb")>=0 :
                    index = None
                    try :
                        index = modules_to_run.index("XtcEventBrowser.pyana_ipimb")
                    except ValueError :
                        index = len(modules_to_run)
                        modules_to_run.append("XtcEventBrowser.pyana_ipimb")
                        options_for_mod.append([])
                    print "XtcEventBrowser.pyana_ipimb at ", index

                    address = str(box.text()).split(":")[1]
                    options_for_mod[index].append("\nipimb_addresses = %s" % address)
                    
                # --- --- --- TM6740 --- --- ---
                if str(box.text()).find("TM6740")>=0 :
                    index = None
                    try :
                        index = modules_to_run.index("XtcEventBrowser.pyana_image")
                    except ValueError :
                        index = len(modules_to_run)
                        modules_to_run.append("XtcEventBrowser.pyana_image")
                        options_for_mod.append([])
                    print "XtcEventBrowser.pyana_image at ", index

                    address = str(box.text()).split(":")[1]
                    options_for_mod[index].append("\nimage_source = %s" % address)
                    options_for_mod[index].append("\ngood_range = %d--%d" % (50,250) )
                    options_for_mod[index].append("\ndark_range = %d--%d" % (250,1050) )

                # --- --- --- Acqiris --- --- ---
                if str(box.text()).find("Acqiris")>=0 :
                    modules_to_run.append("XtcEventBrowser.pyana_acq")

                # --- --- --- pnCCD --- --- ---
                if str(box.text()).find("pnCCD")>=0 :
                    modules_to_run.append("XtcEventBrowser.pyana_misc")
                    configuration+="\nmodules = XtcEventBrowser.pyana_misc\n"
                    configuration+="\n[XtcEventBrowser.pyana_misc]"
                    configuration+="\nimage_source = %s "% str(box.text()).split(":")[1]

        nmodules = len(modules_to_run)
        if nmodules == 0 :
            print "No modules requested! Please select from list"
            return

        # if several values for same option, merge into a list
        for m in range(0,nmodules):
            tmpoptions = {}
            for options in options_for_mod[m] :
                n,v = options.split("=")
                if n in tmpoptions :
                    oldvalue = tmpoptions[n]
                    tmpoptions[n] = oldvalue+v
                else :
                    tmpoptions[n] = v

            newoptions = []
            for n, v in tmpoptions.iteritems() :
                optstring = "%s = %s" % (n, v)
                newoptions.append(optstring)

            options_for_mod[m] = newoptions

        configuration = "[pyana]"
        configuration += "\nmodules ="
        for module in modules_to_run :
            configuration += " "
            configuration += module

        count_m = 0
        for module in modules_to_run :
            configuration += "\n\n["
            configuration += module
            configuration += "]"
            #if len( options_for_mod[ count_m ] )>0 :
            for options in options_for_mod[ count_m ] :
                configuration += options
            count_m +=1
            
        print "Configuration:\n"
        print configuration
        print
        configfile = "xb_pyana_%d.cfg" % random.randint(1000,9999)

        f = open(configfile,'w')
        f.write(configuration)
        f.close()
        
        return  configfile


    def __run_pyana(self):
        """Run pyana

        Open a dialog to allow chaging options to pyana. Wait for OK, then
        run pyana with the needed modules and configurations as requested
        based on the the checkboxes
        """

        configfile = self.__write_config_script()
        print "pyana config file has been generated : ", configfile

        # pop up emacs window to edit the config file as needed:
        subprocess.Popen("emacs %s" % configfile, shell=True) 
        

        print "Now we'll run pyana..... "

        runstring = "pyana -n 1000 -c %s " % configfile
        for file in self.filenames :
            runstring += file
            runstring +=" "

        dialog =  QtGui.QInputDialog()
        dialog.setMinimumWidth(1500)
        text, ok = dialog.getText(self,
                                  'Pyana options',
                                  'Run pyana with the following command (edit as needed and click OK):',
                                  QtGui.QLineEdit.Normal,
                                  text=runstring )
        
        if ok:
            print "Running pyana:"
            print text
            runstring = str(text)
        else :
            return

        print "Calling pyana.... "


        subprocess.Popen(runstring, shell=True) # this runs in separate thread.

        # alternative ways of running pyana:
        #
        #os.system(runstring)  # this will block 
        #
        #program = QtCore.QString("pyana")
        #arguments = QtCore.QStringList()
        #arguments.append("-n 1000")
        #arguments.append("-m XtcEventBrowser.pyana_bld")
        #arguments.append("/reg/d/psdm/CXI/cxi80410/xtc/e55-r0088-s00-c00.xtc")        
        #pyana = QtCore.QProcess()
        #pyana.start(program,arguments)
        #pyana.waitForFinished() 
        #result = pyana.readAll()
        #print result
        #pyana.close()    



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    qApp = QtGui.QApplication(sys.argv)
    mainw = XtcBrowserMain()
    mainw.show()
    sys.exit(qApp.exec_())

    

