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
import sys, random, os, signal, time
import  subprocess 
#from multiprocessing import Process

#---------------------------------
#  Imports of base class module --
#---------------------------------
import matplotlib
matplotlib.use('Qt4Agg')

from PyQt4 import QtCore, QtGui

#-----------------------------
# Imports for other modules --
#-----------------------------
#from pyana import pyanamod

#----------------------------------
# Local non-exported definitions --
#----------------------------------



#------------------------
# Exported definitions --
#------------------------


#---------------------
#  Class definition --
#---------------------
class myPopen(subprocess.Popen):
    def kill(self, signal = signal.SIGTERM):
        os.kill(self.pid, signal)
        

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

        # these must be initialized before use 
        self.filenames = []
        self.devices = []
        self.epicsPVs = []

        self.checklabels = []
        self.checkboxes = []

        self.proc_pyana = None
        self.configfile = None

        self.pvWindow = None
        self.scrollArea = None
        
        self.pvlabels = []
        self.pvboxes = []
        self.pvGroupLayout = None

        # buttons
        self.pyana_config = QtGui.QLabel(self);
        self.config_button = None
        self.econfig_button = None
        self.pyana_button = None
        self.quit_pyana_button = None

        self.show()

        # assume all events        
        self.nevents = None 
        
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
        
        self.hconf = QtGui.QHBoxLayout()
        self.config_button = QtGui.QPushButton("&Write configuration to file")
        self.connect(self.config_button, QtCore.SIGNAL('clicked()'), self.write_configfile )
        self.hconf.addWidget( self.config_button )
        self.econfig_button = QtGui.QPushButton("&Edit configuration file")
        self.connect(self.econfig_button, QtCore.SIGNAL('clicked()'), self.edit_configfile )
        self.hconf.addWidget( self.econfig_button )
        self.config_button.hide()
        self.econfig_button.hide()

        self.v1 = QtGui.QVBoxLayout()
        self.v1.addWidget( self.box_pconf )
        self.v1.setAlignment( self.pyana_config, QtCore.Qt.AlignTop )
        self.v1.addLayout( self.hconf )

        h1.addWidget(self.dgroup)
        h1.addLayout(self.v1)

        # header
        self.v0.addLayout(h0)
        self.v0.addLayout(h1)




    #-------------------
    #  Public methods --
    #-------------------

    def run_pyana(self):
        """Run pyana

        Open a dialog to allow chaging options to pyana. Wait for OK, then
        run pyana with the needed modules and configurations as requested
        based on the the checkboxes
        """

        # Make a command sequence 
        poptions = []
        poptions.append("pyana")
        if self.nevents is not None:
            poptions.append("-n")
            poptions.append(str(self.nevents))
        poptions.append("-c")
        poptions.append("%s" % self.configfile)
        for file in self.filenames :
            poptions.append(file)

        # turn sequence into a string, allow user to modify it
        runstring = ' '.join(poptions)
        dialog =  QtGui.QInputDialog()
        dialog.resize(400,400)
        #dialog.setMinimumWidth(1500)
        text, ok = dialog.getText(self,
                                  'Pyana options',
                                  'Run pyana with the following command (edit as needed and click OK):',
                                  QtGui.QLineEdit.Normal,
                                  text=runstring )
        if ok:
            runstring = str(text)
            poptions = runstring.split(' ')
        else :
            return

        print "Calling pyana.... "
        print "     ", ' '.join(poptions)
        
        
        if 1 :
            # calling a new process
            self.proc_pyana = myPopen(poptions) # this runs in separate thread.
            #stdout_value = proc_pyana.communicate()[0]
            #print stdout_value
            # the benefit of this option is that the GUI remains unlocked
            # the drawback is that pyana needs to supply its own plots, ie. no Qt plots?
            
        if 0 :
            # calling as module... plain
            pyanamod.pyana(argv=poptions)
            # the benefit of this option is that pyana will draw plots on the GUI. 
            # the drawback is that GUI hangs while waiting for pyana to finish...

        if 0 :
            # calling as module... using multiprocessing
            kwargs = {'argv':poptions}
            p = Process(target=pyanamod.pyana,kwargs=kwargs)
            p.start()
            p.join()
            # this option is nothing but trouble
            
            
        if self.quit_pyana_button is None :
            self.quit_pyana_button = QtGui.QPushButton("&Quit pyana")
            self.connect(self.quit_pyana_button, QtCore.SIGNAL('clicked()'), self.quit_pyana )
            self.v0.addWidget( self.quit_pyana_button )
            self.v0.setAlignment( self.quit_pyana_button, QtCore.Qt.AlignRight )


    def quit_pyana(self) :
        """Kill the pyana process
        """
        if self.proc_pyana is None:
            print "No pyana process to stop"
            return
        self.proc_pyana.kill()



    def change_epics_channel(self):
        for pv in self.pvboxes :
            if pv.checkState() :
                self.lgroup.addWidget(pv) # move widget to checkboxes
            else :
                pass
                #self.pvGroupLayout.addWidget(pv)
                
    def update_gui_epics(self):
        """Connect the epics variables to checkboxes
        """
        if self.epics_checkbox.isChecked():
            if self.pvWindow is None:
                self.make_epics_window()
            else :
                self.pvWindow.show()

            # add epics channels to list of checkboxes, place them in a different widget
            for self.pv in self.epicsPVs:
                pvtext = "EpicsPV:" + self.pv
                self.pvi = QtGui.QCheckBox(pvtext,self.pvWindow)
                self.connect(self.pvi, QtCore.SIGNAL('stateChanged(int)'), self.write_configuration )
                self.checkboxes.append(self.pvi)
                self.checklabels.append(self.pvi.text())
                self.pvGroupLayout.addWidget(self.pvi)

        else :
            # 
            for box in self.checkboxes :
                if str(box.text()).find("EpicsPV")>=0 :
                    box.setCheckState(0)
            if self.pvWindow:
                self.pvWindow.hide()


    def make_epics_window(self):
        # open Epics window
        self.pvWindow = QtGui.QWidget()
        self.pvWindow.setStyleSheet("QWidget {background-color: #FFFFFF }")
        self.pvWindow.setWindowTitle('Available Epics PVs')
        self.pvWindow.setWindowIcon(QtGui.QIcon('XtcEventBrowser/src/lclsLogo.gif'))
        self.pvWindow.setMinimumWidth(300)
        self.pvWindow.setMinimumHeight(700)

        # scroll area
        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
                
        # list of PVs, a child of self.scrollArea
        pvGroup = QtGui.QGroupBox("Epics channels:")
        self.scrollArea.setWidget(pvGroup)

        self.pvGroupLayout = QtGui.QVBoxLayout()
        pvGroup.setLayout(self.pvGroupLayout)

        # layout of pvWindow:
        pvLayout = QtGui.QHBoxLayout(self.pvWindow)
        self.pvWindow.setLayout(pvLayout)

        # show window
        #pvLayout.addWidget(pvGroup)
        pvLayout.addWidget(self.scrollArea)
        self.pvWindow.show()
        
            
    def update(self, filenames=[],devices=[],epicsPVs=[]):
        """Update lists of filenames, devices and epics channels
           Make sure GUI gets updated too
        """
        self.filenames=filenames
        self.devices=devices
        self.epicsPVs=epicsPVs

        self.update_gui_checkboxes()


    def update_gui_checkboxes(self) :
        """Draw a group of checkboxes to the GUI

        One checkbox for each Detector/Device found
        in the scan of xtc file
        """
        if len( self.devices ) == 0 :
            print "Can't use selector before running the scan"
            return
        
        for label in sorted( self.devices ):
            if label.find("ProcInfo") >= 0 : continue  # ignore
            if label.find("NoDetector") >= 0 : continue  # ignore
            
            if self.checklabels.count(label)!=0 : continue # avoid duplicates

            # make checkbox for this device
            checkbox = QtGui.QCheckBox(label, self)
            self.connect(checkbox, QtCore.SIGNAL('stateChanged(int)'), self.write_configuration )
            
            self.lgroup.addWidget(checkbox)
            self.checkboxes.append(checkbox)
            self.checklabels.append(label)
            
            # special case: EPICS
            if label.find("EpicsArch") >= 0 :
                self.connect(checkbox, QtCore.SIGNAL('stateChanged(int)'), self.update_gui_epics )
                self.epics_checkbox = checkbox
                

    def add_module(self,box,modules_to_run,options_for_mod) :

        index = None
        plot_every_n = 10
        # --- --- --- BLD --- --- ---
        if str(box.text()).find("BldInfo")>=0 :

            try :
                index = modules_to_run.index("XtcEventBrowser.pyana_bld")
            except ValueError :
                index = len(modules_to_run)
                modules_to_run.append("XtcEventBrowser.pyana_bld")
                options_for_mod.append([])

            #print "XtcEventBrowser.pyana_bld at ", index
            options_for_mod[index].append("\nplot_every_n = %d" % plot_every_n)
            options_for_mod[index].append("\nfignum = %d" % (index+1))
            if str(box.text()).find("EBeam")>=0 :
                options_for_mod[index].append("\ndo_ebeam = True")
            if str(box.text()).find("FEEGasDetEnergy")>=0 :
                options_for_mod[index].append("\ndo_gasdetector = True")
            if str(box.text()).find("PhaseCavity")>=0 :
                options_for_mod[index].append("\ndo_phasecavity = True")
            return

        # --- --- --- Ipimb --- --- ---
        if str(box.text()).find("Ipimb")>=0 :

            try :
                index = modules_to_run.index("XtcEventBrowser.pyana_ipimb")
            except ValueError :
                index = len(modules_to_run)
                modules_to_run.append("XtcEventBrowser.pyana_ipimb")
                options_for_mod.append([])

            #print "XtcEventBrowser.pyana_ipimb at ", index
            address = str(box.text()).split(":")[1]
            options_for_mod[index].append("\nipimb_addresses = %s" % address)
            options_for_mod[index].append("\nplot_every_n = %d" % plot_every_n)
            options_for_mod[index].append("\nfignum = %d" % (index+1))
            return
                    
        # --- --- --- TM6740 --- --- ---
        if ( str(box.text()).find("TM6740")>=0 or
             str(box.text()).find("Opal1000")>=0 or
             str(box.text()).find("Princeton")>=0 ) :

            try :
                index = modules_to_run.index("XtcEventBrowser.pyana_image")
            except ValueError :
                index = len(modules_to_run)
                modules_to_run.append("XtcEventBrowser.pyana_image")
                options_for_mod.append([])

            #print "XtcEventBrowser.pyana_image at ", index
            address = str(box.text()).split(":")[1]
            options_for_mod[index].append("\nimage_addresses = %s" % address)
            options_for_mod[index].append("\nimage_rotations = " )
            options_for_mod[index].append("\nimage_shifts = " )
            options_for_mod[index].append("\nimage_scales = " )
            options_for_mod[index].append("\nimage_manipulations = ")
            options_for_mod[index].append("\ngood_range = %d--%d" % (0,99999999.9) )
            options_for_mod[index].append("\ndark_range = %d--%d" % (0,0) )
            options_for_mod[index].append("\nplot_every_n = %d" % plot_every_n)
            options_for_mod[index].append("\nfignum = %d" % (index+1))
            options_for_mod[index].append("\noutput_file = ")
            options_for_mod[index].append("\nn_hdf5 = ")        
            return

        # --- --- --- CsPad --- --- ---
        if str(box.text()).find("Cspad")>=0 :

            try :
                index = modules_to_run.index("XtcEventBrowser.pyana_cspad")
            except ValueError :
                index = len(modules_to_run)
                modules_to_run.append("XtcEventBrowser.pyana_cspad")
                options_for_mod.append([])

            #print "XtcEventBrowser.pyana_cspad at ", index
            address = str(box.text()).split(":")[1]
            options_for_mod[index].append("\nimage_source = %s" % address)
            options_for_mod[index].append("\nplot_every_n = %d" % plot_every_n)
            options_for_mod[index].append("\nfignum = %d" % (index+1))
            options_for_mod[index].append("\ndark_img_file = ")
            options_for_mod[index].append("\noutput_file = ")                    
            options_for_mod[index].append("\nplot_vrange = ")
            options_for_mod[index].append("\nthreshold = ")
            options_for_mod[index].append("\nthr_area = ")
            return

        # --- --- --- Epics --- --- ---
        if str(box.text()).find("EpicsArch")>=0 :
            return

        if str(box.text()).find("EpicsPV:")>=0 :

            try :
                index = modules_to_run.index("XtcEventBrowser.pyana_epics")
            except ValueError :
                index = len(modules_to_run)
                modules_to_run.append("XtcEventBrowser.pyana_epics")
                options_for_mod.append([])

            #print "XtcEventBrowser.pyana_epics at ", index
            pvname = str(box.text()).split("PV:")[1]
            options_for_mod[index].append("\npv = %s" % pvname)
            options_for_mod[index].append("\nplot_every_n = %d" % plot_every_n )
            options_for_mod[index].append("\nfignum = %d" % (index+1))
            return
        
        print "FIXME! %s requested, not implemented" % box.text() 


    def write_configuration(self):
        """Write the configuration text (to be written to file later)
        
        """
        # clear title 
        self.configfile = None
        if self.econfig_button is not None : self.econfig_button.hide()
        self.box_pconf.setTitle("Current pyana configuration:")

        modules_to_run = []
        options_for_mod = []
        self.configuration = ""
        for box in self.checkboxes :
            if box.isChecked() :
                self.add_module(box, modules_to_run, options_for_mod)

        nmodules = len(modules_to_run)
        if nmodules > 0 :
            # at the end, append plotter module:
            modules_to_run.append("XtcEventBrowser.pyana_plotter")
            options_for_mod.append([])
            options_for_mod[nmodules].append("\ndisplay_mode = Interactive")

        # if several values for same option, merge into a list
        for m in range(0,nmodules):
            tmpoptions = {}
            for options in options_for_mod[m] :
                n,v = options.split("=")
                if n in tmpoptions :
                    oldvalue = tmpoptions[n]
                    if oldvalue!=v:   # avoid duplicates
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

        self.config_button.show()
        self.econfig_button.show()
        self.config_button.setEnabled(True)
        self.econfig_button.setDisabled(True)


    def print_configuration(self):
        print "----------------------------------------"
        print "Configuration file (%s): " % self.configfile
        print "----------------------------------------"
        print self.configuration
        print "----------------------------------------"
        return

    def write_configfile(self):
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

        self.print_configuration
        
        self.config_button.setDisabled(True)
        self.econfig_button.setEnabled(True)

        if self.pyana_button is None: 
            self.pyana_button = QtGui.QPushButton("&Run pyana")
            self.connect(self.pyana_button, QtCore.SIGNAL('clicked()'), self.run_pyana )
            self.v0.addWidget( self.pyana_button )
            self.v0.setAlignment( self.pyana_button, QtCore.Qt.AlignRight )


    def edit_configfile(self):

        # pop up emacs window to edit the config file as needed:
        proc_emacs = myPopen("emacs %s" % self.configfile, shell=True) 
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

        self.print_configuration
        print "Done"



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
