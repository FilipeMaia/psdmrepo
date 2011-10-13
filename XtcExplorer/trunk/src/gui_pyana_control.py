#--------------------------------------------------------------------------
# Description:
#  Module gui_pyana_control...
#  GUI for writing pyana configuration file
#------------------------------------------------------------------------

"""GUI for writing pyana configuration file
@see gui_explorer_main.py
@author Ingrid Ofte
"""
#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys, random, os, signal

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
import subprocess 
from pyana import pyanamod

import config_pyana as cfg
import gui_config_panels as panels

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
        code = self.poll()
        if code is None: 
            os.kill(self.pid, signal)
            print "pyana process %d has been killed "% self.pid
            return 1
        else :
            print "No pyana process to kill... "
            if code==0 : 
                print "Finished successfully"
            else :
                print "Exited with code ", code
                code = 1
            return code

#class MyThread( threading.Thread ):
class MyThread( QtCore.QThread ):

    """Run pyana module in a separate thread. This allows the GUI windows
    to stay active. The only problem is that Matplotlib windows need to me
    made beforehand, by the GUI -- not in pyana. Not a problem as long as
    they are declared beforehand. Pyana can still call the plt.figure command,
    that way it can be run standalone or from the GUI. 
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

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def kill(self):
        print " !   python threads cannot be interupted..."
        print " !   you'll have to wait..."
        print " !   or ^Z and kill the whole xtcbrowser process."
        # SOLUTION: Run the Plot gui in a subprocess, this subprocess then calls the thread.
        # That should keep the other GUIs active, while the Plot GUI waits (or not) for pyana.
        # Killing pyana thread then requires killing the Plot GUI subprocess.
        #self.terminate()  ... hangs indefinitely & freezes up the GUI
        #self.exit(0) ..... does nothing
        #self.quit() .... does nothing
        print "done killing"

        
class XtcPyanaControl ( QtGui.QWidget ) :
    """Gui interface to pyana configuration & control

    @see pyana
    @see XtcExplorerMain
    """

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self,
                  data, 
                  parent = None) :
        """Constructor.

        @param data    object that holds information about the data
        @param parent  parent widget, if any
        """
        QtGui.QWidget.__init__(self, parent)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setStyleSheet("QWidget {background-color: #FFFFFF }")

        self.setWindowTitle('Pyana Control Center')
        self.setWindowIcon(QtGui.QIcon('XtcExplorer/src/lclsLogo.gif'))

        # container for information about the data
        self.filenames = data.files
        self.devices = data.devices.keys()
        self.epicsPVs = data.epicsPVs
        self.controls = data.controls
        self.moreinfo = data.moreinfo
        self.nevents = data.nevents
        self.ncalib = len(data.nevents)

        # configuration object for pyana job
        self.settings = cfg.Configuration()

        self.config_tab = { 'pyana_image_beta'    : self.image_tab,
                            'pyana_scan_beta'     : self.scan_tab,
                            'pyana_ipimb_beta'    : self.ipimb_tab,
                            'pyana_bld_beta'      : self.bld_tab,
                            'pyana_waveform_beta' : self.waveform_tab
                            }
        # ------- SELECTION / CONFIGURATION ------
        self.checklabels = None
        self.checkboxes = None

        self.proc_pyana = None
        self.proc_status = None

        self.pvWindow = None
        self.pvGroupLayout = None

        self.pyana_config_text = QtGui.QLabel(self);

        # buttons
        self.config_button = None
        self.econfig_button = None
        self.pyana_button = None
        self.quit_pyana_button = None

        self.scan_widget = None
        self.pyana_widget = None 
        self.info_widget = None

        self.define_layout()
        self.show()
        
        # show all of this in the Gui
        self.setup_gui_checkboxes()

        # if scan, plot every calib cycle 
        if self.ncalib > 1 :
            print "Have %d scan steps a %d events each. Set up to plot after every %d events" %\
                  (self.ncalib, self.nevents[0], self.nevents[0] )
            self.plotn_enter.setText( str(self.nevents[0]) )
            self.plotn_change()
            self.plotn_enter.setText("")


    def define_layout(self):
        """ Main layout of Pyana Control Center
        """
        self.layout = QtGui.QVBoxLayout(self)

        # header: icon
        h0 = QtGui.QHBoxLayout()
        pic = QtGui.QLabel(self)
        pic.setPixmap( QtGui.QPixmap('XtcExplorer/src/lclsLogo.gif'))
        h0.addWidget( pic )
        h0.setAlignment( pic, QtCore.Qt.AlignLeft )
        label = QtGui.QLabel(self)
        label_text = """
Configure your analysis here... 
Start with selecting data of interest to you from list on the left and general run / display options from the tab(s) on the right. 
"""
        label.setText(label_text)
        h0.addWidget( label )
        h0.setAlignment( label, QtCore.Qt.AlignRight )

        # mid layer: almost everything
        h1 = QtGui.QHBoxLayout()

        # to the left:
        detector_gbox = QtGui.QGroupBox("In the file(s):")
        self.detector_box_layout = QtGui.QVBoxLayout()
        detector_gbox.setLayout(self.detector_box_layout)
        h1.addWidget(detector_gbox)

        # to the right:
        self.cfg_tabs_tally = {}
        self.cfg_tabs = QtGui.QTabWidget()
        self.cfg_tabs.tabsClosable()
        self.cfg_tabs.setMinimumWidth(600)

        self.general_tab()
        self.pyana_tab()
        h1.addWidget(self.cfg_tabs)

        # header
        self.layout.addLayout(h0)
        self.layout.addLayout(h1)

    def setup_gui_checkboxes(self) :
        """Draw a group of checkboxes to the GUI

        Each checkbox gets connected to the function process_checkbox,
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
            self.connect(ckbox, QtCore.SIGNAL('stateChanged(int)'), self.process_checkbox)
            nctrl += 1

        for label in sorted(self.devices):
            if label.find("ProcInfo") >= 0 : continue  # ignore
            if label.find("NoDetector") >= 0 : continue  # ignore
            
            if self.checklabels.count(label)!=0 : continue # avoid duplicates

            # make checkbox for this device
            #checkbox = QtGui.QCheckBox(': '.join(label.split(":")), self)
            checkbox = QtGui.QCheckBox( label.split(":")[1], self)
            self.connect(checkbox, QtCore.SIGNAL('stateChanged(int)'), self.process_checkbox )
            
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
            self.detector_box_layout.addWidget(checkbox)
            
    def general_tab(self):
        tabname = "General Settings"
        if tabname in self.cfg_tabs_tally:
            index = self.cfg_tabs_tally[ tabname ]
            self.cfg_tabs.setCurrentIndex(index)
            return

        general_widget = panels.JobConfigGui( self.settings )
        self.connect( general_widget.apply_button,
                      QtCore.SIGNAL('clicked()'), self.update_pyana_tab )
        
        num = self.cfg_tabs.addTab(general_widget,tabname)
        self.cfg_tabs_tally[tabname] = num
        #print "general tab tally: ",self.cfg_tabs_tally

    def bld_tab(self, mod, remove=False):
        tabname = "BldInfo"
        if tabname in self.cfg_tabs_tally:
            index = self.cfg_tabs_tally[ tabname ]
            self.cfg_tabs.setCurrentIndex(index)
            if remove:
                self.cfg_tabs.removeTab(index)
                del self.cfg_tabs_tally[tabname]
            return

        bld_widget = panels.BldConfigGui()
        self.connect( bld_widget.apply_button,
                      QtCore.SIGNAL('clicked()'), self.update_pyana_tab )

        num = self.cfg_tabs.addTab(bld_widget,tabname)
        self.cfg_tabs_tally[tabname] = num
        #print "general tab tally: ",self.cfg_tabs_tally
        self.cfg_tabs.setCurrentWidget(bld_widget)

                
    def waveform_tab(self, mod, remove=False):
        tabname = "%s"%mod.address

        if tabname in self.cfg_tabs_tally:
            index = self.cfg_tabs_tally[ tabname ]
            self.cfg_tabs.setCurrentIndex(index)
            if remove:
                self.cfg_tabs.removeTab(index)
                del self.cfg_tabs_tally[tabname]
            return

        wf_widget = panels.WaveformConfigGui(self,title=tabname)
        print wf_widget
        self.connect( wf_widget.apply_button,
                      QtCore.SIGNAL('clicked()'), self.update_pyana_tab )

        num = self.cfg_tabs.addTab(wf_widget,tabname)
        self.cfg_tabs_tally[tabname] = num
        #print "general tab tally: ",self.cfg_tabs_tally
        self.cfg_tabs.setCurrentWidget(wf_widget)

                
    
    def image_tab(self, mod, remove=False):
        tabname = "%s"%mod.address

        if tabname in self.cfg_tabs_tally:
            index = self.cfg_tabs_tally[ tabname ]
            self.cfg_tabs.setCurrentIndex(index)
            if remove:
                self.cfg_tabs.removeTab(index)
                del self.cfg_tabs_tally[tabname]
            return

        image_widget = panels.ImageConfigGui(mod,self)
        print image_widget
        self.connect(image_widget.apply_button,
                     QtCore.SIGNAL('clicked()'), self.update_pyana_tab )        

        num = self.cfg_tabs.addTab(image_widget,"%s"%mod.address)
        self.cfg_tabs_tally["%s"%mod.address] = num
        #print "tabs tally: ", self.cfg_tabs_tally
        self.cfg_tabs.setCurrentWidget(image_widget)
        return 
            
    def ipimb_tab(self, mod, remove=False):
        tabname = "%s"%mod.address
        if tabname in self.cfg_tabs_tally:
            index = self.cfg_tabs_tally[ tabname ]
            self.cfg_tabs.setCurrentIndex(index)
            if remove:
                self.cfg_tabs.removeTab(index)
                del self.cfg_tabs_tally[tabname]
            return

        ipimb_widget = QtGui.QWidget()
        page_layout = QtGui.QVBoxLayout(ipimb_widget)

        # has one widget: checkboxes for plotting
        selection_box = QtGui.QGroupBox("Select what to plot:")
        selection_layout = QtGui.QHBoxLayout()
        selection_box.setLayout( selection_layout )
        
        button_update_layout = QtGui.QHBoxLayout()
        button_update = QtGui.QPushButton("Apply")
        button_update.setMaximumWidth(90)
        #button_update.setDisabled(True)
        self.connect(button_update, QtCore.SIGNAL('clicked()'), self.update_pyana_tab )
        button_update_layout.addStretch()
        button_update_layout.addWidget(button_update)
        
        page_layout.addWidget( selection_box )
        page_layout.addLayout( button_update_layout )

        # -------------------------------------------
        # checkboxes"
        fex_ch_label = QtGui.QLabel("Fex data", self)
        checkbox_sum = QtGui.QCheckBox("Sum", self) 
        checkbox_posx = QtGui.QCheckBox("Position X", self) 
        checkbox_posy = QtGui.QCheckBox("Position Y", self) 
        checkbox_chfex  = QtGui.QCheckBox("All Channels", self) 
        checkbox_ch0fex = QtGui.QCheckBox("Ch0", self) 
        checkbox_ch1fex = QtGui.QCheckBox("Ch1", self) 
        checkbox_ch2fex = QtGui.QCheckBox("Ch2", self) 
        checkbox_ch3fex = QtGui.QCheckBox("Ch3", self) 

        self.connect(checkbox_sum, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_sum )
        self.connect(checkbox_posx, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_posX )
        self.connect(checkbox_posy, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_posY )
        self.connect(checkbox_chfex, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_chFex )
        self.connect(checkbox_ch0fex, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch0Fex )
        self.connect(checkbox_ch1fex, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch1Fex )
        self.connect(checkbox_ch2fex, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch2Fex )
        self.connect(checkbox_ch3fex, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch3Fex )
        fex_ch_layout = QtGui.QVBoxLayout()
        fex_ch_layout.addWidget(fex_ch_label)
        fex_ch_layout.addWidget(checkbox_sum)
        fex_ch_layout.addWidget(checkbox_posx)
        fex_ch_layout.addWidget(checkbox_posy)
        fex_ch_layout.addWidget(checkbox_chfex)
        fex_ch_layout.addWidget(checkbox_ch0fex)
        fex_ch_layout.addWidget(checkbox_ch1fex)
        fex_ch_layout.addWidget(checkbox_ch2fex)
        fex_ch_layout.addWidget(checkbox_ch3fex)
        selection_layout.addLayout(fex_ch_layout)
        #
        raw_ch_label = QtGui.QLabel("Raw counts", self)
        checkbox_ch = QtGui.QCheckBox("All Channels", self) 
        checkbox_ch0 = QtGui.QCheckBox("Ch0", self) 
        checkbox_ch1 = QtGui.QCheckBox("Ch1", self) 
        checkbox_ch2 = QtGui.QCheckBox("Ch2", self) 
        checkbox_ch3 = QtGui.QCheckBox("Ch3", self) 
        self.connect(checkbox_ch, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_chRaw )
        self.connect(checkbox_ch0, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch0 )
        self.connect(checkbox_ch1, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch1 )
        self.connect(checkbox_ch2, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch2 )
        self.connect(checkbox_ch3, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch3 )
        raw_ch_layout = QtGui.QVBoxLayout()
        raw_ch_layout.addWidget(raw_ch_label)
        raw_ch_layout.addWidget(checkbox_ch)
        raw_ch_layout.addWidget(checkbox_ch0)
        raw_ch_layout.addWidget(checkbox_ch1)
        raw_ch_layout.addWidget(checkbox_ch2)
        raw_ch_layout.addWidget(checkbox_ch3)
        selection_layout.addLayout(raw_ch_layout)
        #
        volt_ch_label = QtGui.QLabel("Raw voltages", self)
        checkbox_chV  = QtGui.QCheckBox("All Channels", self) 
        checkbox_ch0V = QtGui.QCheckBox("Ch0", self) 
        checkbox_ch1V = QtGui.QCheckBox("Ch1", self) 
        checkbox_ch2V = QtGui.QCheckBox("Ch2", self) 
        checkbox_ch3V = QtGui.QCheckBox("Ch3", self) 
        self.connect(checkbox_chV, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_chVolt )
        self.connect(checkbox_ch0V, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch0Volt )
        self.connect(checkbox_ch1V, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch1Volt )
        self.connect(checkbox_ch2V, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch2Volt )
        self.connect(checkbox_ch3V, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_ch3Volt )        
        volt_ch_layout = QtGui.QVBoxLayout()
        volt_ch_layout.addWidget(volt_ch_label)
        volt_ch_layout.addWidget(checkbox_chV)
        volt_ch_layout.addWidget(checkbox_ch0V)
        volt_ch_layout.addWidget(checkbox_ch1V)
        volt_ch_layout.addWidget(checkbox_ch2V)
        volt_ch_layout.addWidget(checkbox_ch3V)
        selection_layout.addLayout(volt_ch_layout)
        # -------------------------------------------


        num = self.cfg_tabs.addTab(ipimb_widget,tabname)
        self.cfg_tabs_tally[tabname] = num
        #print "tabs tally: ", self.cfg_tabs_tally
        self.cfg_tabs.setCurrentWidget(ipimb_widget)
        return 
            
    def draw_image(self):
        print "Note to self: remember to draw the image!"
        


    def scan_tab(self, who, remove=False):
        """ Second tab: Scan
        """
        tabname = "Scan"
        if tabname in self.cfg_tabs_tally:
            index = self.cfg_tabs_tally[ tabname ]
            self.cfg_tabs.setCurrentIndex(index)
            if remove:
                self.cfg_tabs.removeTab(index)
                del self.cfg_tabs_tally[tabname]
            return

        if self.scan_widget is None :
            self.scan_widget = QtGui.QWidget()
            self.scan_layout = QtGui.QVBoxLayout(self.scan_widget)

            message = QtGui.QLabel()
            message.setText("Scan vs. %s"%"Hallo")

            self.scan_layout.addWidget(message)

            
            num = self.cfg_tabs.addTab(self.scan_widget,tabname)
            self.cfg_tabs_tally[tabname] = num
            #print "tabs tally: ", self.cfg_tabs_tally


        self.cfg_tabs.setCurrentWidget(self.scan_widget)
        self.cfg_tabs.tabBar().show()
        self.write_configuration()
        
    def pyana_tab(self):
        """Pyana configuration text
        """
        tabname = "Pyana Configuration"
        pyana_widget = QtGui.QWidget()
        pyana_layout = QtGui.QVBoxLayout(pyana_widget)
        pyana_widget.setLayout(pyana_layout)
        self.pyana_config_label = QtGui.QLabel("Current pyana configuration:")
        
        # scroll area for the configuration file text
        scrollArea = QtGui.QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget( self.pyana_config_text )
        
        pyana_layout.addWidget(self.pyana_config_label)
        pyana_layout.addWidget(scrollArea)
        
        # add some buttons to this tab, for writing/editing config file, and run and quit pyana
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

        num = self.cfg_tabs.addTab(pyana_widget,tabname)
        self.cfg_tabs_tally[tabname] = num
        #print "tabs tally: ", self.cfg_tabs_tally

        self.cfg_tabs.tabBar().show()
        self.pyana_widget = pyana_widget
        
    def update_pyana_tab(self):
        
        self.settings.update_text()
        self.pyana_config_text.setText(self.settings.config_text)

        #self.cfg_tabs.setCurrentWidget(self.pyana_widget)
        # clear title 
        #if self.econfig_button is not None : self.econfig_button.hide()
        #if self.pyana_button is not None: self.pyana_button.hide()
        #if self.quit_pyana_button is not None: self.quit_pyana_button.hide()
        self.config_button.show()
        self.econfig_button.show()
        self.config_button.setEnabled(True)
        self.econfig_button.setDisabled(True)

        self.cfg_tabs.setCurrentWidget(self.pyana_widget)

    #-------------------
    #  Public methods --
    #-------------------

    def setup_gui_epics(self):
        """Open a new window if epics_checkbox is checked.
        If not, clear all fields and hide. 
        Add checkboxes for each known epics PV channel.
        connect each of these to process_checkbox
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

                    self.connect(self.pvi, QtCore.SIGNAL('stateChanged(int)'), self.process_checkbox )
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
        self.pvWindow.setWindowIcon(QtGui.QIcon('XtcExplorer/src/lclsLogo.gif'))
        self.pvWindow.setMinimumWidth(300)
        self.pvWindow.setMinimumHeight(700)

        # scroll area
        scrollArea = QtGui.QScrollArea()
        scrollArea.setWidgetResizable(True)
                
        # list of PVs, a child of scrollArea
        pvGroup = QtGui.QGroupBox("Epics channels (%d):"%len(self.epicsPVs))
        scrollArea.setWidget(pvGroup)

        self.pvGroupLayout = QtGui.QVBoxLayout()
        pvGroup.setLayout(self.pvGroupLayout)

        # layout of pvWindow:
        pvLayout = QtGui.QHBoxLayout(self.pvWindow)
        self.pvWindow.setLayout(pvLayout)

        # show window
        #pvLayout.addWidget(pvGroup)
        pvLayout.addWidget(scrollArea)
        self.pvWindow.show()
        

    def process_checkbox(self):
        """Process checkbox
        """
        checkbox = self.sender()
        checkbox_label = str(checkbox.text())

        if checkbox.isChecked():
            # 1) Add module to module list 
            module = self.settings.add_module( checkbox_label )
            self.update_pyana_tab()
        
            # 2) Open tab to configure plots
            self.config_tab[module.name](module)            
        else :
            # 1) Remove module from module list
            module = self.settings.remove_module( checkbox_label )
            self.update_pyana_tab()

            # 2) Close tab for configuring plots
            self.config_tab[module.name](module,remove=True)
            

    def add_to_scan(self,box,modules_to_run,options_for_mod) :
  
        index = None
        try:
            index = modules_to_run.index("XtcExplorer.pyana_scan")
        except ValueError :
            print "ValueError"
            
        #print "XtcExplorer.pyana_scan at ", index
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

        

    def print_configuration(self):
        print "----------------------------------------"
        print "Configuration file (%s): " % self.settings.file
        print "----------------------------------------"
        print self.settings.config_text
        print "----------------------------------------"
        return

    def write_configfile(self):
        """Write the configuration text to a file. Filename is generated randomly
        """

        self.settings.file = "xb_pyana_%d.cfg" % random.randint(1000,9999)

        self.pyana_config_label.setText("Current pyana configuration: (%s)" % self.settings.file)

        f = open(self.settings.file,'w')
        f.write(self.settings.config_text)
        f.close()

        self.print_configuration()
        
        self.config_button.setDisabled(True)
        self.econfig_button.setEnabled(True)

        if self.pyana_button is None: 
            self.pyana_button = QtGui.QPushButton("&Run pyana")
            self.pyana_button.setMaximumWidth(120)
            self.proc_status = QtGui.QLabel("")
            
            self.connect(self.pyana_button, QtCore.SIGNAL('clicked()'), self.run_pyana)

            pyana_button_line = QtGui.QHBoxLayout()
            pyana_button_line.addWidget( self.proc_status )
            pyana_button_line.addWidget( self.pyana_button )
            self.layout.addLayout( pyana_button_line  )
            self.layout.setAlignment( pyana_button_line, QtCore.Qt.AlignRight )
        else :
            self.pyana_button.show()

    def edit_configfile(self):
        proc_emacs = None
        try: 
            myeditor = os.environ['EDITOR']
            print "Launching your favorite editor %s to edit config file" % myeditor
            proc_emacs = myPopen("$EDITOR %s" % self.settings.file, shell=True) 
        except :
            print "Launching emacs to edit the config file."
            print "To launch another editor of your choice, make sure to",
            print "set the EDITOR variable in your shell environment."
            proc_emacs = myPopen("emacs %s" % self.settings.file, shell=True)

        stdout_value = proc_emacs.communicate()[0]
        print stdout_value
        #proc_emacs = MyThread("emacs %s" % self.settings.file) 
        #proc_emacs.start()
        
        f = open(self.settings.file,'r')
        configtext = f.read()
        f.close()

        self.pyana_config_label.setText("Current pyana configuration: (%s)" % self.settings.file)
        self.pyana_config_text.setText(configtext)

        # should add:
        # PARSE FILE & UPDATE ANY GUI FIELDS OR BUTTONS


    def run_pyana(self):
        """Run pyana

        Open a dialog to allow chaging options to pyana. Wait for OK, then
        run pyana with the needed modules and configurations as requested
        based on the the checkboxes
        """

        # Make a command sequence 
        lpoptions = []
        lpoptions.append("pyana")
        if self.settings.run_n is not None:
            lpoptions.append("-n")
            lpoptions.append("%s"%str(self.settings.run_n))
        if self.settings.skip_n is not None:
            lpoptions.append("-s")
            lpoptions.append("%s"%str(self.settings.skip_n))
        if self.settings.num_cpu is not None:
            lpoptions.append("-p")
            lpoptions.append("%s"%str(self.settings.num_cpu))
        lpoptions.append("-c")
        lpoptions.append("%s" % self.settings.file)
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

# DISABLE run-dialogue feedback for now
#            # and update run_n and skip_n in the Gui:
#            if "-n" in lpoptions:
#                self.settings.run_n = int(lpoptions[ lpoptions.index("-n")+1 ])
#                general_widget.run_n_status.setText("Process %s events"%\
#                                                    self.settings.run_n)
#            if "-s" in lpoptions:
#                self.settings.skip_n = int(lpoptions[ lpoptions.index("-s")+1 ])
#                general_widget.skip_n_status.setText("Skip the fist %s events of xtc file"%\
#                                                     self.settings.skip_n)
#            if "-p" in lpoptions:
#                self.settings.num_cpu = int(lpoptions[ lpoptions.index("-p")+1 ])
#                general_widget.mproc_status.setText("Multiprocessing with %s CPUs"%\
#                                                    self.settings.num_cpu)
        else :
            return

        print "Calling pyana.... "
        print "     ", ' '.join(lpoptions)

        if 1 :
            # calling a new process
            self.proc_pyana = myPopen(lpoptions) # this runs in separate thread.
            self.proc_status.setText("pyana process %d is running "%self.proc_pyana.pid)
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
            #kwargs = {'argv':lpoptions}
            #p = mp.Process(target=pyanamod.pyana,kwargs=kwargs)
            #p.start()
            #p.join()
            # this option is nothing but trouble
            pass
        if 0 :
            # calling as module... using threading.
            self.proc_pyana = MyThread(lpoptions)
            self.proc_pyana.start()
            print "I'm back"
            
            
        self.pyana_button.setDisabled(True)
            
        if self.quit_pyana_button is None :
            self.quit_pyana_button = QtGui.QPushButton("&Quit pyana")
            self.quit_pyana_button.setMaximumWidth(120)
            self.connect(self.quit_pyana_button, QtCore.SIGNAL('clicked()'), self.quit_pyana )
            self.layout.addWidget( self.quit_pyana_button )
            self.layout.setAlignment( self.quit_pyana_button, QtCore.Qt.AlignRight )
        else :
            self.quit_pyana_button.show()

    def quit_pyana(self) :
        """Kill the pyana process
        """
        if self.proc_pyana :

            statustext = {1 : "process %d killed"%self.proc_pyana.pid,
                          0 : "process %d finished successfully"%self.proc_pyana.pid }
            status = self.proc_pyana.kill()

            self.pyana_button.setDisabled(False)        
            self.proc_status.setText(statustext[status])
            self.quit_pyana_button.hide()
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
