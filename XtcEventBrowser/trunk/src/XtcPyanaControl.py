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

#---------------------------------
#  Imports of base class module --
#---------------------------------
import matplotlib
matplotlib.use('Qt4Agg')

from PyQt4 import QtCore, QtGui

#-----------------------------
# Imports for other modules --
#-----------------------------

import threading
from multiprocessing import Process
from pyana import pyanamod

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
        print "pyana process %d has been killed "% self.pid

#class MyThread( threading.Thread ):
class MyThread( QtCore.QThread ):
    """Run pyana module in a separate thread.
    This allows the GUI windows to stay active.
    The only problem is that Matplotlib windows
    need to me made beforehand, by the Gui. Not
    in pyana. Not a problem as long as they are
    declared beforehand. Pyana can still call the
    plt.figure command, that way it can be run
    standalone or from the GUI. 
    In principle...
    Still some issues to look into:
    - matplotlib figure must be created before pyana runs
    - This begs embedded mpl in a tool GUI. 
    """
    def __init__(self,pyanastring = ""):
        self.lpoptions = pyanastring
        QtCore.QThread.__init__(self)
        #threading.Thread.__init__ ( self )        
        
    def run(self):
        pyanamod.pyana(argv=self.lpoptions)

    def kill(self):
        print " !   python threads cannot be interupted..."
        print " !   you'll have to wait..."
        print " !   or ^Z and kill the whole xtcbrowser process."

    

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

        # --------------- INPUT --------------
        # these must be initialized before use (by calling module)
        self.filenames = []
        self.devices = []
        self.epicsPVs = []
        self.controls = []
        self.moreinfo = []

        # pyana options
        self.poptions = None
        # -------------------------------------
        

        # ------- SELECTION / CONFIGURATION ------
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

        self.scan_widget = None
        self.pyana_widget = None
        self.info_widget = None

        # assume all events        
        self.nevents = None 

        self.define_layout()

        self.show()
        


    def define_layout(self):
        """ Main layout of Pyana Control Center
        """
        self.layout = QtGui.QVBoxLayout(self)

        # header: icon
        h0 = QtGui.QHBoxLayout()
        pic = QtGui.QLabel(self)
        pic.setPixmap( QtGui.QPixmap('XtcEventBrowser/src/lclsLogo.gif'))
        h0.addWidget( pic )
        h0.setAlignment( pic, QtCore.Qt.AlignLeft )

        # mid layer: almost everything
        h1 = QtGui.QHBoxLayout()
        # to the left: 
        detector_gbox = QtGui.QGroupBox("In the file(s):")
        # layout of the group must be global, checkboxes added later
        self.lgroup = QtGui.QVBoxLayout()
        detector_gbox.setLayout(self.lgroup)
        h1.addWidget(detector_gbox)

        # to the right:
        self.config_tabs = QtGui.QTabWidget()
        self.config_tabs.setMinimumWidth(500)
        self.intro_tab()
        h1.addWidget(self.config_tabs)

        # header
        self.layout.addLayout(h0)
        self.layout.addLayout(h1)

                
    def intro_tab(self):
        # First tab: help/info
        self.help_widget = QtGui.QWidget()
        self.help_layout = QtGui.QHBoxLayout(self.help_widget)
        self.help_subwg1 = QtGui.QLabel(self.help_widget)
        self.help_subwg1_text = """
    Configure your Event Display / Analysis:

    Select the information / detectors of interest to you from list
    to the left. Pyana modules will be configured for you to analyze
    the information.

    You can edit the configuration or pyana modules afterwards, 
    if you want to further customize your analysis.
    """
        self.help_subwg1.setText(self.help_subwg1_text)
        self.help_layout.addWidget(self.help_subwg1)
        self.config_tabs.addTab(self.help_widget,"Help")
        self.config_tabs.tabBar().hide()

    def scan_tab(self, who):
        """ Second tab: Scan
        """
        if self.scan_widget is None :
            self.scan_widget = QtGui.QWidget()
            self.scan_layout = QtGui.QVBoxLayout(self.scan_widget)

            message = QtGui.QLabel()
            message.setText("Scan vs. %s"%"Hallo")

            self.scan_layout.addWidget(message)

            self.config_tabs.addTab(self.scan_widget,"Scan Configuration")

        self.config_tabs.setCurrentWidget(self.scan_widget)
        self.config_tabs.tabBar().show()
        self.write_configuration()

        
    def pyana_tab(self):
        """Pyana configuration text
        """
        if self.pyana_widget is None :
            pyana_widget = QtGui.QWidget()
            pyana_layout = QtGui.QVBoxLayout(pyana_widget)

            pyana_txtbox = QtGui.QGroupBox("Current pyana configuration:")
            pyana_txtbox_layout = QtGui.QVBoxLayout()
            pyana_txtbox_layout.addWidget(self.pyana_config)
            pyana_txtbox.setLayout(pyana_txtbox_layout)
            pyana_layout.addWidget(pyana_txtbox)
            self.pyana_txtbox = pyana_txtbox
        
            pyana_button_layout = QtGui.QHBoxLayout()
            self.config_button = QtGui.QPushButton("&Write configuration to file")
            self.connect(self.config_button, QtCore.SIGNAL('clicked()'), self.write_configfile )
            pyana_button_layout.addWidget( self.config_button )
            self.econfig_button = QtGui.QPushButton("&Edit configuration file")
            self.connect(self.econfig_button, QtCore.SIGNAL('clicked()'), self.edit_configfile )
            pyana_button_layout.addWidget( self.econfig_button )
            self.config_button.hide()
            self.econfig_button.hide()
            pyana_layout.addLayout(pyana_button_layout)
            
            self.config_tabs.addTab(pyana_widget,"Pyana")
            self.config_tabs.setCurrentWidget(pyana_widget)
            self.config_tabs.tabBar().show()
            self.pyana_widget = pyana_widget


    #-------------------
    #  Public methods --
    #-------------------
                
    def update(self, filenames=[],devices=[],epicsPVs=[],controls=[],moreinfo=[]):
        """Update lists of filenames, devices and epics channels
           Make sure GUI gets updated too
        """
        self.filenames = filenames
        self.devices = devices
        self.moreinfo = moreinfo
        self.epicsPVs = epicsPVs
        self.controls = controls

        # show all of this in the Gui
        self.setup_gui_checkboxes()
        for ch in self.checkboxes:
            print ch.text()

    def setup_gui_checkboxes(self) :
        """Draw a group of checkboxes to the GUI

        Each checkbox gets connected to the function process_checkboxes,
        i.e., whenerver *one* checkbox is checked/unchecked, the state of 
        every checkbox is investigated. Not pretty, but works OK.
        """
        # lists of QCheckBoxes and their text lables. 
        self.checkboxes = []
        self.checklabels = []


        # from the controls list
        nctrl = 0
        for ctrl in self.controls:
            ckbox = QtGui.QCheckBox("ControlPV: %s"%ctrl, self)
            self.checkboxes.append(ckbox)
            self.checklabels.append(ckbox.text())
            self.connect(ckbox, QtCore.SIGNAL('stateChanged(int)'), self.process_checkboxes)
            nctrl += 1

        for label in sorted(self.devices):
            if label.find("ProcInfo") >= 0 : continue  # ignore
            if label.find("NoDetector") >= 0 : continue  # ignore
            
            if self.checklabels.count(label)!=0 : continue # avoid duplicates

            # make checkbox for this device
            checkbox = QtGui.QCheckBox(': '.join(label.split(":")), self)
            self.connect(checkbox, QtCore.SIGNAL('stateChanged(int)'), self.process_checkboxes )
            
            # special case: Epics PVs
            if label.find("Epics") >= 0 : 
                checkbox.setText("Epics Process Variables (%d)"%len(self.epicsPVs))
                self.connect(checkbox, QtCore.SIGNAL('stateChanged(int)'), self.setup_gui_epics )
                # add epics to front
                self.checkboxes.insert(nctrl,checkbox)
                self.checklabels.insert(nctrl,checkbox.text())
                # make global
                self.epics_checkbox = checkbox
            else :
                # add everything else to the end
                self.checkboxes.append(checkbox)
                self.checklabels.append(label)
                
        # finally, add each to the layout
        for checkbox in self.checkboxes :
            self.lgroup.addWidget(checkbox)
            

    def setup_gui_epics(self):
        """Open a new window if epics_checkbox is checked.
        If not, clear all fields and hide. 
        Add checkboxes for each known epics PV channel.
        connect each of these to process_checkboxes
        """
        if self.epics_checkbox.isChecked():
            if self.pvWindow is None:
                self.make_epics_window()

                # add epics channels to list of checkboxes, place them in a different widget
                for self.pv in self.epicsPVs:
                    pvtext = "EpicsPV:  " + self.pv
                    self.pvi = QtGui.QCheckBox(pvtext,self.pvWindow)

                    ## check those that are control pvs
                    #for ctrl in self.controls: 
                    #    if ctrl in pvtext :
                    #        self.pvi.setChecked(True)

                    self.connect(self.pvi, QtCore.SIGNAL('stateChanged(int)'), self.process_checkboxes )
                    self.checkboxes.append(self.pvi)
                    self.checklabels.append(self.pvi.text())
                    self.pvGroupLayout.addWidget(self.pvi)

            else :
                self.pvWindow.show()

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
        pvGroup = QtGui.QGroupBox("Epics channels (%d):"%len(self.epicsPVs))
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
        
            

    def process_checkboxes(self):
        """Process the list of checkboxes and
        call the appropriate function based on the
        checkbox name/label
        """
        self.pyana_tab()
        
        # clear title 
        self.configfile = None
        if self.econfig_button is not None : self.econfig_button.hide()
        if self.pyana_button is not None: self.pyana_button.hide()
        if self.quit_pyana_button is not None: self.quit_pyana_button.hide()

        self.pyana_txtbox.setTitle("Current pyana configuration:")

        modules_to_run = []
        options_for_mod = []
        self.configuration= ""

        print 
        do_scan = False
        for box in sorted(self.checkboxes):
            if box.isChecked() :
                if str(box.text()).find("ControlPV:")>=0 :
                    do_scan = True
                    self.add_module(box, modules_to_run, options_for_mod)
                elif do_scan :
                    self.add_to_scan(box, modules_to_run, options_for_mod)
                else :
                    self.add_module(box, modules_to_run, options_for_mod)
                
        nmodules = len(modules_to_run)
        if nmodules > 0 :
            # at the end, append plotter module:
            modules_to_run.append("XtcEventBrowser.pyana_plotter")
            options_for_mod.append([])
            options_for_mod[nmodules].append("\ndisplay_mode = SlideShow")

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
            

        # add linebreaks if needed
        self.configuration = self.add_linebreaks(self.configuration, width=70)
        #print self.configuration

        self.pyana_config.setText(self.configuration)

        self.config_button.show()
        self.econfig_button.show()
        self.config_button.setEnabled(True)
        self.econfig_button.setDisabled(True)


    def add_to_scan(self,box,modules_to_run,options_for_mod) :
  
        index = None
        try:
            index = modules_to_run.index("XtcEventBrowser.pyana_scan")
        except ValueError :
            print "ValueError"
            
        #print "XtcEventBrowser.pyana_scan at ", index
        source = str(box.text())
        if source.find("BldInfo")>=0 :
            options_for_mod[index].append("\ninput_scalars = %s" % source.split(": ")[1] )
            return
        if source.find("EpicsPV")>=0 :
            options_for_mod[index].append("\ninput_epics = %s" % source.split(": ")[1])
            return
        if source.find("DetInfo")>=0 :
            options_for_mod[index].append("\ninput_scalars = %s" % source.split(": ")[1])
            return



    def add_module(self,box,modules_to_run,options_for_mod) :

        index = None
        plot_every_n = 10

        # The following sets up one out of two analysis modes:
        #      1) scan
        #      2) all-in-one analysis 

        # --- --- --- Scan --- --- ---
        if str(box.text()).find("ControlPV:")>=0 :
            try :
                index = modules_to_run.index("XtcEventBrowser.pyana_scan")
            except ValueError :
                index = len(modules_to_run)
                modules_to_run.append("XtcEventBrowser.pyana_scan")
                options_for_mod.append([])

            #print "XtcEventBrowser.pyana_scan at ", index
            pvname = str(box.text()).split("PV: ")[1]
            options_for_mod[index].append("\ncontrolpv = %s" % pvname)
            options_for_mod[index].append("\ninput_epics = ")
            options_for_mod[index].append("\ninput_scalars = ")
            options_for_mod[index].append("\nplot_every_n = %d" % plot_every_n )
            options_for_mod[index].append("\nfignum = %d" % (index+1))
            return

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
        
        # --- --- --- Waveform --- --- ---
        if ( str(box.text()).find("Acq")>=0  or
             str(box.text()).find("ETof")>=0  or
             str(box.text()).find("ITof")>=0  or
             str(box.text()).find("Mbes")>=0  or
             str(box.text()).find("Camp")>=0  ) :
            try :
                index = modules_to_run.index("XtcEventBrowser.pyana_waveform")
            except ValueError :
                index = len(modules_to_run)
                modules_to_run.append("XtcEventBrowser.pyana_waveform")
                options_for_mod.append([])

            #print "XtcEventBrowser.pyana_ipimb at ", index
            address = str(box.text()).split(":")[1].strip()
            options_for_mod[index].append("\nsources = %s" % address)
            options_for_mod[index].append("\nplot_every_n = %d" % plot_every_n)
            options_for_mod[index].append("\nfignum = %d" % (index+1))
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
            address = str(box.text()).split(": ")[1].strip()
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
            address = str(box.text()).split(": ")[1].strip()
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
            address = str(box.text()).split(":")[1].strip()
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
        if str(box.text()).find("Epics Process Variables")>=0 :
            return

        if str(box.text()).find("EpicsPV:")>=0 :

            try :
                index = modules_to_run.index("XtcEventBrowser.pyana_epics")
            except ValueError :
                index = len(modules_to_run)
                modules_to_run.append("XtcEventBrowser.pyana_epics")
                options_for_mod.append([])

            #print "XtcEventBrowser.pyana_epics at ", index
            pvname = str(box.text()).split("PV:  ")[1]
            options_for_mod[index].append("\npv = %s" % pvname)
            options_for_mod[index].append("\nplot_every_n = %d" % plot_every_n )
            options_for_mod[index].append("\nfignum = %d" % (index+1))
            return
        
        print "FIXME! %s requested, not implemented" % box.text() 

    def add_linebreaks(self, configtext, width=70):
        lines = configtext.split('\n')
        l = 0
        for line in lines :
            if len(line) > width : # split line
                words = line.split(" ")
                i = 0
                newlines = []
                newline = ""
                while len(newline) <= width and i <len(words) :
                    newline += (words[i]+" ")
                    i += 1
                    if len(newline) > width or i==len(words):
                        newlines.append(newline)
                        newline = "     "
                        
                # now replace the original line with newlines
                l = lines.index(line)
                lines.remove(line)
                if len(newlines)>1 :
                    newlines.reverse()
                for linje in newlines :
                    if linje.strip() != '' :
                        lines.insert(l,linje)
                    
        configtext = "\n".join(lines)
        return configtext
        

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

        self.pyana_txtbox.setTitle("Current pyana configuration: (%s)" % self.configfile)

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
            self.layout.addWidget( self.pyana_button )
            self.layout.setAlignment( self.pyana_button, QtCore.Qt.AlignRight )
        else :
            self.pyana_button.show()

    def edit_configfile(self):

        # pop up emacs window to edit the config file as needed:
        proc_emacs = myPopen("emacs %s" % self.configfile, shell=True) 
        stdout_value = proc_emacs.communicate()[0]
        print stdout_value
        
        f = open(self.configfile,'r')
        self.configuration = f.read()
        f.close()

        self.pyana_txtbox.setTitle("Current pyana configuration: (%s)" % self.configfile)
        self.pyana_config.setText(self.configuration)

        print "----------------------------------------"
        print "Configuration file (%s): " % self.configfile
        print "----------------------------------------"
        print self.configuration
        print "----------------------------------------"

        self.print_configuration
        print "Done"


    def run_pyana(self):
        """Run pyana

        Open a dialog to allow chaging options to pyana. Wait for OK, then
        run pyana with the needed modules and configurations as requested
        based on the the checkboxes
        """

        # Make a command sequence 
        lpoptions = []
        if self.poptions is not None:
            # remember line from last time
            lpoptions = self.poptions
            # but replace with the latest config file
            lpoptions[ lpoptions.index("-c")+1 ] = self.configfile
        else :
            lpoptions.append("pyana")
            if self.nevents is not None:
                lpoptions.append("-n")
                lpoptions.append(str(self.nevents))
            lpoptions.append("-c")
            lpoptions.append("%s" % self.configfile)
            for file in self.filenames :
                lpoptions.append(file)

        # turn sequence into a string, allow user to modify it
        runstring = ' '.join(lpoptions)
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
            lpoptions = runstring.split(' ')
        else :
            return

        print "Calling pyana.... "
        print "     ", ' '.join(lpoptions)

        # use this as default next time
        self.poptions = lpoptions
        
        if 1 :
            # calling a new process
            self.proc_pyana = myPopen(lpoptions) # this runs in separate thread.
            #stdout_value = proc_pyana.communicate()[0]
            #print stdout_value
            # the benefit of this option is that the GUI remains unlocked
            # the drawback is that pyana needs to supply its own plots, ie. no Qt plots?
            
        if 0 :
            # calling as module... plain
            pyanamod.pyana(argv=lpoptions)
            # the benefit of this option is that pyana will draw plots on the GUI. 
            # the drawback is that GUI hangs while waiting for pyana to finish...

        if 0 :
            # calling as module... using multiprocessing
            kwargs = {'argv':lpoptions}
            p = Process(target=pyanamod.pyana,kwargs=kwargs)
            p.start()
            p.join()
            # this option is nothing but trouble

        if 0 :
            # calling as module... using threading
            self.proc_pyana = MyThread(lpoptions)
            self.proc_pyana.start()
            print "I'm back"
            
            
        if self.quit_pyana_button is None :
            self.quit_pyana_button = QtGui.QPushButton("&Quit pyana")
            self.connect(self.quit_pyana_button, QtCore.SIGNAL('clicked()'), self.quit_pyana )
            self.layout.addWidget( self.quit_pyana_button )
            self.layout.setAlignment( self.quit_pyana_button, QtCore.Qt.AlignRight )
        else :
            self.quit_pyana_button.show()

    def quit_pyana(self) :
        """Kill the pyana process
        """
        if self.proc_pyana :
            self.proc_pyana.kill()
            return

        print "No pyana process to stop"


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
