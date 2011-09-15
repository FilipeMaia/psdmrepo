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
class RegionInput(QtGui.QWidget):
    def __init__(self, name, module, layout):
        """Region input (x1, x2, y1, y2)
        @param name     is the name of the quantity we're modifying
        @param module   is the relevant module-configuration container
        @param layout   is a QtGui.QBoxLayout widget that this widget belongs to
        """
        QtGui.QWidget.__init__(self)
        self.name=name

        self.label = QtGui.QLabel("Pixels [x1-x2:y1-y2]: ")
        self.xmin = QtGui.QLineEdit("")
        self.xmin.setMaximumWidth(30)
        self.xmax = QtGui.QLineEdit("")
        self.xmax.setMaximumWidth(30)
        self.ymin = QtGui.QLineEdit("")
        self.ymin.setMaximumWidth(30)
        self.ymax = QtGui.QLineEdit("")
        self.ymax.setMaximumWidth(30)
        self.button = QtGui.QPushButton("OK") 

        self.layout = layout
        self.module = module

    def add_to_layout(self):
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.xmin)
        self.layout.addWidget(self.xmax)
        self.layout.addWidget(self.ymin)
        self.layout.addWidget(self.ymax)
        self.layout.addWidget(self.button)
        self.label.show()
        self.xmin.show()
        self.xmax.show()
        self.ymin.show()
        self.ymax.show()
        self.button.show()

    def hide(self):
        self.label.hide()
        self.xmin.hide()
        self.xmax.hide()
        self.ymin.hide()
        self.ymax.hide()
        self.button.hide()

    def update_label(self):
        x1 = str(self.xmin.text())
        x2 = str(self.xmax.text())
        y1 = str(self.ymin.text())
        y2 = str(self.ymax.text())
        pixels = "[%s,%s,%s,%s]"%(x1,x2,y1,y2)
        self.label.setText("Pixels %s: " % pixels)
        self.add_to_layout()
        self.module.add_modifier(quantity=self.name,modifier=pixels)

    def connect_button(self):
        #print "Connected"
        self.connect(self.button, QtCore.SIGNAL('clicked()'), self.update_label )


class AxisInput(QtGui.QWidget):
    def __init__(self, name, module, layout ):
        """Widget taking 3 inputs: low, high, nbins
        @param name     is the name of the quantity we're modifying        
        @param module   is the relevant module-configuration container
        @param layout   is a QtGui.QBoxLayout widget that this widget belongs to
        """
        QtGui.QWidget.__init__(self)
        self.name = name

        self.range_label = QtGui.QLabel("Range: ")
        self.min = QtGui.QLineEdit("")
        self.min.setMaximumWidth(50)
        self.max = QtGui.QLineEdit("")
        self.max.setMaximumWidth(50)

        self.nbins_label = QtGui.QLabel("NBins: ")
        self.nbins = QtGui.QLineEdit("")
        self.nbins.setMaximumWidth(40)

        self.button = QtGui.QPushButton("OK") 
        self.button.setMaximumWidth(40)

        self.layout = layout
        self.module = module

    def add_to_layout(self):
        self.layout.addWidget(self.range_label)
        self.layout.addWidget(self.min)
        self.layout.addWidget(self.max)
        self.layout.addWidget(self.nbins_label)
        self.layout.addWidget(self.nbins)
        self.layout.addWidget(self.button)
        self.range_label.show()
        self.min.show()
        self.max.show()
        self.nbins_label.show()
        self.nbins.show()
        self.button.show()

    def hide(self):
        self.range_label.hide()
        self.min.hide()
        self.max.hide()
        self.nbins_label.hide()
        self.nbins.hide()
        self.button.hide()

    def update_label(self):
        fro = str(self.min.text())
        to = str(self.max.text())
        n = str(self.nbins.text())
        axis = "[%s,%s,%s]" % (fro,to,n)
        self.range_label.setText("Range = (%s, %s)"%(fro, to))
        self.nbins_label.setText("NBins =%s"%(n))
        self.module.add_modifier( quantity=self.name, modifier=axis )
        
    def connect_button(self):
        #print "Connected"
        self.connect(self.button, QtCore.SIGNAL('clicked()'), self.update_label )



class myPopen(subprocess.Popen):
    def kill(self, signal = signal.SIGTERM):
        os.kill(self.pid, signal)
        print "pyana process %d has been killed "% self.pid



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
        self.moreinfo = data.moreinfo.values()
        self.nevents = data.nevents
        self.ncalib = len(data.nevents)

        # configuration object for pyana job
        self.pyana_cfg = cfg.Configuration()

        self.config_tab = { 'pyana_image_beta' : self.image_tab,
                            'pyana_scan'       : self.scan_tab,
                            'pyana_ipimb_beta' : self.ipimb_tab,
                            'pyana_bld'        : self.bld_tab
                            }
        # ------- SELECTION / CONFIGURATION ------
        self.checklabels = None
        self.checkboxes = None

        self.proc_pyana = None

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
        self.cfg_tabs.setMinimumWidth(600)

        self.general_tab()
        self.pyana_tab()
        h1.addWidget(self.cfg_tabs)

        # header
        self.layout.addLayout(h0)
        self.layout.addLayout(h1)

    def general_tab(self):

        general_widget = panels.JobConfigGui( self.pyana_cfg )
        self.connect( general_widget.apply_button,
                      QtCore.SIGNAL('clicked()'), self.update_pyana_tab )
        
        num = self.cfg_tabs.addTab(general_widget,"General Settings")
        self.cfg_tabs_tally["General Settings"] = num
        #print "general tab tally: ",self.cfg_tabs_tally

    def bld_tab(self, mod):

        if "BldInfo" in self.cfg_tabs_tally:
            index = self.cfg_tabs_tally["BldInfo"]
            self.cfg_tabs.setCurrentIndex(index)
            return

        bld_widget = panels.BldConfigGui()
        self.connect( bld_widget.apply_button,
                      QtCore.SIGNAL('clicked()'), self.update_pyana_tab )

        num = self.cfg_tabs.addTab(bld_widget,"BldInfo")
        self.cfg_tabs_tally["BldInfo"] = num
        #print "general tab tally: ",self.cfg_tabs_tally
        self.cfg_tabs.setCurrentWidget(bld_widget)

                
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
            
    
    def image_tab(self, mod):
        image_widget = QtGui.QWidget()
        page_layout = QtGui.QVBoxLayout(image_widget)

        # has two widgets: checkboxes for plotting, alterations
        selection_box = QtGui.QGroupBox("Select what to plot:")
        selection_layout = QtGui.QVBoxLayout()
        selection_box.setLayout( selection_layout )
        
        alterations_box = QtGui.QGroupBox("Background subtraction etc:")
        alterations_layout = QtGui.QVBoxLayout()
        alterations_box.setLayout( alterations_layout )

        button_update_layout = QtGui.QHBoxLayout()
        button_update = QtGui.QPushButton("Apply")
        button_update.setMaximumWidth(90)
        #button_update.setDisabled(True)
        self.connect(button_update, QtCore.SIGNAL('clicked()'), self.update_pyana_tab )
        button_update_layout.addStretch()
        button_update_layout.addWidget(button_update)
        
        page_layout.addWidget( selection_box )
        page_layout.addWidget( alterations_box )
        page_layout.addLayout( button_update_layout )
        
        # checkbox 'image'
        layout_image_conf = QtGui.QHBoxLayout()
        checkbox_image = QtGui.QCheckBox("Main image (x vs y)", self)
        layout_image_conf.addWidget(checkbox_image)
        selection_layout.addLayout(layout_image_conf)

        self.connect(checkbox_image, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_imXY )
        checkbox_image.setChecked(True)

        # checkbox 'roi'
        roi_layout = QtGui.QHBoxLayout()
        checkbox_roi = QtGui.QCheckBox("Region of interest", self) 
        roi_layout.addWidget(checkbox_roi)
        roi_layout.addStretch()
        selection_layout.addLayout(roi_layout)

        roi_input = RegionInput("roi",mod,roi_layout)
        roi_input.connect_button()
        def ask_about_roi(value):
            mod.set_opt_roi(value)
            if value == 2 :
                roi_input.add_to_layout()
            else:
                roi_input.hide()
        self.connect(checkbox_roi, QtCore.SIGNAL('stateChanged(int)'), ask_about_roi )


        spectrum_layout = QtGui.QHBoxLayout()
        checkbox_spectrum = QtGui.QCheckBox("Intensity spectrum", self)
        spectrum_layout.addWidget(checkbox_spectrum)
        spectrum_layout.addStretch()
        selection_layout.addLayout(spectrum_layout)

        spectrum_input = AxisInput("spectrum",mod,spectrum_layout)
        spectrum_input.connect_button()
        def ask_about_spectrum(value):
            mod.set_opt_spectr(value)
            if value == 2:
                spectrum_input.add_to_layout()
            else:
                spectrum_input.hide()                
        self.connect(checkbox_spectrum, QtCore.SIGNAL('stateChanged(int)'), ask_about_spectrum )

        checkbox_projX = QtGui.QCheckBox("ProjX", self)
        self.connect(checkbox_projX, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_projX )
        selection_layout.addWidget(checkbox_projX)

        checkbox_projY = QtGui.QCheckBox("ProjY", self)
        self.connect(checkbox_projY, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_projY )
        selection_layout.addWidget(checkbox_projY)
        
        checkbox_projR = QtGui.QCheckBox("ProjR", self)
        self.connect(checkbox_projR, QtCore.SIGNAL('stateChanged(int)'), mod.set_opt_projR )
        selection_layout.addWidget(checkbox_projR)

        num = self.cfg_tabs.addTab(image_widget,"%s"%mod.address)
        self.cfg_tabs_tally["%s"%mod.address] = num
        #print "tabs tally: ", self.cfg_tabs_tally
        self.cfg_tabs.setCurrentWidget(image_widget)
        return 
            
    def ipimb_tab(self, mod):
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


        
        num = self.cfg_tabs.addTab(ipimb_widget,"%s"%mod.address)
        self.cfg_tabs_tally["%s"%mod.address] = num
        #print "tabs tally: ", self.cfg_tabs_tally
        self.cfg_tabs.setCurrentWidget(ipimb_widget)
        return 
            
    def draw_image(self):
        print "Note to self: remember to draw the image!"
        


    def scan_tab(self, who):
        """ Second tab: Scan
        """
        if self.scan_widget is None :
            self.scan_widget = QtGui.QWidget()
            self.scan_layout = QtGui.QVBoxLayout(self.scan_widget)

            message = QtGui.QLabel()
            message.setText("Scan vs. %s"%"Hallo")

            self.scan_layout.addWidget(message)

            num = self.cfg_tabs.addTab(self.scan_widget,"Scan Configuration")
            self.cfg_tabs_tally["Scan Configuration"] = num
            #print "tabs tally: ", self.cfg_tabs_tally


        self.cfg_tabs.setCurrentWidget(self.scan_widget)
        self.cfg_tabs.tabBar().show()
        self.write_configuration()
        
    def pyana_tab(self):
        """Pyana configuration text
        """
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

        num = self.cfg_tabs.addTab(pyana_widget,"Pyana Configuration")
        self.cfg_tabs_tally["Pyana Configuration"] = num
        #print "tabs tally: ", self.cfg_tabs_tally

        self.cfg_tabs.tabBar().show()
        self.pyana_widget = pyana_widget
        
    def update_pyana_tab(self):
        self.pyana_cfg.update_text()
        self.pyana_config_text.setText(self.pyana_cfg.config_text)

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
                
    def update(self, filenames=[],devices=[],epicsPVs=[],controls=[],moreinfo=[],nevents=[]):
        """Update lists of filenames, devices and epics channels
           Make sure GUI gets updated too
        """
        self.filenames = filenames
        self.devices = devices
        self.moreinfo = moreinfo
        self.epicsPVs = epicsPVs
        self.controls = controls
        self.nevents = nevents
        self.ncalib = len(nevents)

        # show all of this in the Gui
        self.setup_gui_checkboxes()

        # if scan, plot every calib cycle 
        if self.ncalib > 1 :
            print "Have %d scan steps a %d events each. Set up to plot after every %d events" %\
                  (self.ncalib, self.nevents[0], self.nevents[0] )
            self.plotn_enter.setText( str(self.nevents[0]) )
            self.plotn_change()
            self.plotn_enter.setText("")

        print "Configure pyana by selecting from the detector list"


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
            module = self.pyana_cfg.add_module( checkbox_label )
            self.update_pyana_tab()
        
            # 2) Open tab to configure plots
            self.config_tab[module.name](module)
            
        else :
            # 1) Remove module from module list
            self.pyana_cfg.remove_module( checkbox_label )
            # 2) Close tab for configuring plots
            #self.cfg_tabs.removeTab( self.cfg_tabs_tally[checkbox_label] )
            self.cfg_tabs.removeTab( self.cfg_tabs_tally[checkbox_label] )
            self.update_pyana_tab()


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
        print "Configuration file (%s): " % self.pyana_cfg.file
        print "----------------------------------------"
        print self.pyana_cfg.config_text
        print "----------------------------------------"
        return

    def write_configfile(self):
        """Write the configuration text to a file. Filename is generated randomly
        """

        self.pyana_cfg.file = "xb_pyana_%d.cfg" % random.randint(1000,9999)

        self.pyana_config_label.setText("Current pyana configuration: (%s)" % self.pyana_cfg.file)

        f = open(self.pyana_cfg.file,'w')
        f.write(self.pyana_cfg.config_text)
        f.close()

        self.print_configuration()
        
        self.config_button.setDisabled(True)
        self.econfig_button.setEnabled(True)

        if self.pyana_button is None: 
            self.pyana_button = QtGui.QPushButton("&Run pyana")
            self.connect(self.pyana_button, QtCore.SIGNAL('clicked()'), self.run_pyana)
            self.layout.addWidget( self.pyana_button )
            self.layout.setAlignment( self.pyana_button, QtCore.Qt.AlignRight )
        else :
            self.pyana_button.show()

    def edit_configfile(self):

        # pop up emacs window to edit the config file as needed:
        #proc_emacs = myPopen("emacs %s" % self.pyana_cfg.file, shell=True)
        #proc_emacs = myPopen("nano %s" % self.pyana_cfg.file, shell=True)
        proc_emacs = myPopen("$EDITOR %s" % self.pyana_cfg.file, shell=True) 
        stdout_value = proc_emacs.communicate()[0]
        print stdout_value
        #proc_emacs = MyThread("emacs %s" % self.pyana_cfg.file) 
        #proc_emacs.start()
        
        f = open(self.pyana_cfg.file,'r')
        configtext = f.read()
        f.close()

        self.pyana_config_label.setText("Current pyana configuration: (%s)" % self.pyana_cfg.file)
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
        if self.pyana_cfg.jobconfig.run_n is not None:
            lpoptions.append("-n")
            lpoptions.append(self.pyana_cfg.jobconfig.run_n)
        if self.pyana_cfg.jobconfig.skip_n is not None:
            lpoptions.append("-s")
            lpoptions.append(self.pyana_cfg.jobconfig.skip_n)
        if self.pyana_cfg.jobconfig.num_cpu is not None:
            lpoptions.append("-p")
            lpoptions.append(self.pyana_cfg.jobconfig.num_cpu)
        lpoptions.append("-c")
        lpoptions.append("%s" % self.pyana_cfg.file)
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

            # and update run_n and skip_n in the Gui:
            if "-n" in lpoptions:
                self.pyana_cfg.jobconfig.run_n = int(lpoptions[ lpoptions.index("-n")+1 ])
                self.run_n_status.setText("Process %s events"% self.pyana_cfg.jobconfig.run_n)
            if "-s" in lpoptions:
                self.pyana_cfg.jobconfig.skip_n = int(lpoptions[ lpoptions.index("-s")+1 ])
                self.skip_n_status.setText("Skip the fist %s events of xtc file"% self.pyana_cfg.jobconfig.skip_n)
            if "-p" in lpoptions:
                self.pyana_cfg.jobconfig.num_cpu = int(lpoptions[ lpoptions.index("-p")+1 ])
                self.mproc_status.setText("Multiprocessing with %s CPUs"% self.pyana_cfg.jobconfig.num_cpu)
        else :
            return

        print "Calling pyana.... "
        print "     ", ' '.join(lpoptions)

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
