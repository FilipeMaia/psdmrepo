#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module XtcPyanaControl...
#
#------------------------------------------------------------------------

"""Brief one-line description of the module.

@see XtcBrowserMain.py

@version $Id: template!python!py 4 2011-02-04 16:01:36Z ofte $

@author Ingrid Ofte
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys, random
import  subprocess 

#---------------------------------
#  Imports of base class module --
#---------------------------------
import matplotlib
matplotlib.use('Qt4Agg')

from PyQt4 import QtCore, QtGui
import matplotlib.pyplot as plt

#-----------------------------
# Imports for other modules --
#-----------------------------
import pyanascript

#----------------------------------
# Local non-exported definitions --
#----------------------------------



#------------------------
# Exported definitions --
#------------------------



#---------------------
#  Class definition --
#---------------------
class XtcPyanaControl ( QtGui.QWidget ) :
    """Gui interface to pyana configuration & control

    @see pyana
    @see XtcBrowserMain
    """

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None) :
        """Constructor.
        
        """
        QtGui.QWidget.__init__(self, parent)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setStyleSheet("QWidget {background-color: #FFFFFF }")

        self.setWindowTitle('Pyana Control Center')
        self.setWindowIcon(QtGui.QIcon('XtcEventBrowser/src/lclsLogo.gif'))

        self.checks = []
        self.checkboxes = []
        self.filenames = []
        
        # buttons
        self.pyana_config = QtGui.QLabel(self);
        self.config_button = None
        self.econfig_button = None
        self.pyana_button = None

        self.set_layout()
        self.show()


    def set_files(self, filenames = [] ):
        self.filenames = filenames
        
    def set_layout(self) :

        # Layout of window
        self.v0 = QtGui.QVBoxLayout(self)

        # header
        h0 = QtGui.QHBoxLayout()

        # Icon
        pic = QtGui.QLabel(self)
        pic.setPixmap( QtGui.QPixmap('XtcEventBrowser/src/lclsLogo.gif'))

        h0.addWidget( pic )
        h0.setAlignment( pic, QtCore.Qt.AlignLeft )

        # Layout of devices and configuration
        h1 = QtGui.QHBoxLayout()

        # Layout of device selection
        self.dgroup = QtGui.QGroupBox("Available Detectors/Devices:")
        self.lgroup = QtGui.QVBoxLayout()
        self.dgroup.setLayout(self.lgroup)

        self.box_pconf = QtGui.QGroupBox("Current pyana configuration:")
        self.ly_pconf = QtGui.QVBoxLayout()
        self.ly_pconf.addWidget(self.pyana_config )
        self.box_pconf.setLayout(self.ly_pconf)
        

        v1 = QtGui.QVBoxLayout()
        v1.addWidget( self.box_pconf )
        v1.setAlignment( self.pyana_config, QtCore.Qt.AlignTop )

        h1.addWidget(self.dgroup)
        h1.addLayout(v1)

        # header
        self.v0.addLayout(h0)
        self.v0.addLayout(h1)




    #-------------------
    #  Public methods --
    #-------------------

    def add_selector(self, devices={} ):
        """Draw a group of checkboxes to the GUI

        Draw a group of checkboxes to the GUI. 
        One checkbox for each Detector/Device found
        in the scan of xtc file
        """
        if len( devices ) == 0 :
            print "Can't use selector before running the scan"
            return
        

        for d in sorted( devices.keys() ):
            clabel = d 
            if clabel.find("ProcInfo") >= 0 : continue
            if clabel.find("NoDetector") >= 0 : continue
            if self.checks.count( clabel )==0 :
                self.c = QtGui.QCheckBox(clabel, self)
                self.connect(self.c, QtCore.SIGNAL('stateChanged(int)'), self.__write_configuration )
                self.lgroup.addWidget(self.c)
                self.checkboxes.append(self.c)
                self.checks.append(clabel)





    def __write_configuration(self):
        """Write the configuration text (to be written to file later)
        
        """
        nmodules = 0
        modules_to_run = []
        options_for_mod = []

        self.configuration = ""
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
                    options_for_mod[index].append("\ndraw_each_event = 1")
                    

                # --- --- --- CsPad --- --- ---
                if str(box.text()).find("Cspad")>=0 :
                    index = None
                    try :
                        index = modules_to_run.index("XtcEventBrowser.pyana_cspad")
                    except ValueError :
                        index = len(modules_to_run)
                        modules_to_run.append("XtcEventBrowser.pyana_cspad")
                        options_for_mod.append([])
                    print "XtcEventBrowser.pyana_cspad at ", index

                    address = str(box.text()).split(":")[1]
                    options_for_mod[index].append("\nimage_source = %s" % address)
                    options_for_mod[index].append("\ndraw_each_event = 1")
                    options_for_mod[index].append("\ncollect_darks = 0")
                    options_for_mod[index].append("\n#dark_img_file = pyana_cspad_average_image.npy")

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

        self.configuration = "[pyana]"
        self.configuration += "\nmodules ="
        for module in modules_to_run :
            self.configuration += " "
            self.configuration += module

        count_m = 0
        for module in modules_to_run :
            self.configuration += "\n\n["
            self.configuration += module
            self.configuration += "]"
            #if len( options_for_mod[ count_m ] )>0 :
            for options in options_for_mod[ count_m ] :
                self.configuration += options
            count_m +=1
            
        self.pyana_config.setText(self.configuration)


        if self.config_button is None: 
            self.config_button = QtGui.QPushButton("&Write configuration to file")
            self.connect(self.config_button, QtCore.SIGNAL('clicked()'), self.__write_configfile )
            self.ly_pconf.addWidget( self.config_button )
            self.ly_pconf.setAlignment( self.config_button, QtCore.Qt.AlignLeft )


    def __print_configuration(self):
        print "----------------------------------------"
        print "Configuration file (%s): " % self.configfile
        print "----------------------------------------"
        print self.configuration
        print "----------------------------------------"
        return

    def __write_configfile(self):
        """Write the configuration text to a file. Filename is generated randomly
        """

        self.configfile = "xb_pyana_%d.cfg" % random.randint(1000,9999)

        self.box_pconf.setTitle("Current pyana configuration: (%s)" % self.configfile)

        f = open(self.configfile,'w')
        f.write(self.configuration)
        f.close()

        print "----------------------------------------"
        print "Configuration file (%s): " % self.configfile
        print "----------------------------------------"
        print self.configuration
        print "----------------------------------------"

        self.__print_configuration
        
        

        if self.econfig_button is None: 
            self.econfig_button = QtGui.QPushButton("&Edit configuration file")
            self.connect(self.econfig_button, QtCore.SIGNAL('clicked()'), self.__edit_configfile )
            self.ly_pconf.addWidget( self.econfig_button )
            self.ly_pconf.setAlignment( self.econfig_button, QtCore.Qt.AlignRight )

        if self.pyana_button is None: 
            self.pyana_button = QtGui.QPushButton("&Run pyana")
            self.connect(self.pyana_button, QtCore.SIGNAL('clicked()'), self.__run_pyana )
            self.v0.addWidget( self.pyana_button )
            self.v0.setAlignment( self.pyana_button, QtCore.Qt.AlignRight )


    def __edit_configfile(self):

        # pop up emacs window to edit the config file as needed:
        proc_emacs = subprocess.Popen("emacs %s" % self.configfile, shell=True) 
        stdout_value = proc_emacs.communicate()[0]
        print stdout_value
        
        f = open(self.configfile,'r')
        self.configuration = f.read()
        f.close()

        self.box_pconf.setTitle("Current pyana configuration: (%s)" % self.configfile)
        self.pyana_config.setText(self.configuration)

        print "----------------------------------------"
        print "Configuration file (%s): " % self.configfile
        print "----------------------------------------"
        print self.configuration
        print "----------------------------------------"

        self.__print_configuration
        print "Done"


    def __run_pyana(self):
        """Run pyana

        Open a dialog to allow chaging options to pyana. Wait for OK, then
        run pyana with the needed modules and configurations as requested
        based on the the checkboxes
        """
        print "Now we'll run pyana..... "

        #poptions = []
        #poptions.append("pyana")
        #poptions.append("-n")
        #poptions.append("10")
        #poptions.append("-c")
        #poptions.append("%s" % configfile)
        #for file in self.filenames :
        #    poptions.append(file)
        #pyanascript.main(poptions)


        runstring = "pyana -n 1000 -c %s " % self.configfile
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

        plt.show()
        

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


    #--------------------------------
    #  Static/class public methods --
    #--------------------------------


    #--------------------
    #  Private methods --
    #--------------------

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
