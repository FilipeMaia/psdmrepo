import multiprocessing as mp
from PyQt4 import QtCore, QtGui

class BldConfigGui( QtGui.QWidget ):
    """
    """
    def __init__(self, parent = None ):
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(QtCore.QRect(20,20,800,800))

        widget_layout = QtGui.QVBoxLayout(self)

        self.apply_button = QtGui.QPushButton()
        self.apply_button.setGeometry(QtCore.QRect(470,420,96,30))
        self.apply_button.setText("Apply")
        self.apply_button.setMaximumWidth(60)

        # -------------------------------------------------------
        ebeam_gbox = QtGui.QGroupBox("EBeam")
        ebeam_gbox.setGeometry(QtCore.QRect(30,30,200,200))
        ebeam_gbox.setMinimumHeight(140)

        checkbox_energy = QtGui.QCheckBox("L3Energy", ebeam_gbox)
        checkbox_energy.setGeometry(QtCore.QRect(10,30,140,21))
        
        checkbox_current = QtGui.QCheckBox("PkCurrBC2", ebeam_gbox)
        checkbox_current.setGeometry(QtCore.QRect(170,30,140,21))

        checkbox_posx = QtGui.QCheckBox("PositionX", ebeam_gbox)
        checkbox_posx.setGeometry(QtCore.QRect(10,60,140,21))

        checkbox_posy = QtGui.QCheckBox("PositionY", ebeam_gbox)
        checkbox_posy.setGeometry(QtCore.QRect(10,80,140,21))

        checkbox_posxy = QtGui.QCheckBox("Position X vs. Y", ebeam_gbox)
        checkbox_posxy.setGeometry(QtCore.QRect(10,110,140,21))

        checkbox_angx = QtGui.QCheckBox("AngleX", ebeam_gbox)
        checkbox_angx.setGeometry(QtCore.QRect(170,60,140,21))

        checkbox_angy = QtGui.QCheckBox("AngleY", ebeam_gbox)
        checkbox_angy.setGeometry(QtCore.QRect(170,80,140,21))

        checkbox_angxy = QtGui.QCheckBox("Angle X vs. Y", ebeam_gbox)
        checkbox_angxy.setGeometry(QtCore.QRect(170,110,140,21))

        checkbox_xposang = QtGui.QCheckBox("X Position vs. Angle", ebeam_gbox)
        checkbox_xposang.setGeometry(QtCore.QRect(330,60,140,21))

        checkbox_yposang = QtGui.QCheckBox("Y Position vs. Angle", ebeam_gbox)
        checkbox_yposang.setGeometry(QtCore.QRect(330,80,140,21))

        # -------------------------------------------------------
        gasdet_gbox = QtGui.QGroupBox("FEEGasDetector")
        gasdet_gbox.setGeometry(QtCore.QRect(30,30,200,70))
        gasdet_gbox.setMinimumHeight(60)
        
        checkbox_earray = QtGui.QCheckBox("Energy array", gasdet_gbox)
        checkbox_earray.setGeometry(QtCore.QRect(10,30,100,21))
        
        
        # -------------------------------------------------------
        phasecavity_gbox = QtGui.QGroupBox("PhaseCavity")
        phasecavity_gbox.setGeometry(QtCore.QRect(30,30,200,200))
        phasecavity_gbox.setMinimumHeight(120)
        
        checkbox_time1 = QtGui.QCheckBox("FitTime1", phasecavity_gbox)
        checkbox_time1.setGeometry(QtCore.QRect(10,30,140,21))

        checkbox_time2 = QtGui.QCheckBox("FitTime2", phasecavity_gbox)
        checkbox_time2.setGeometry(QtCore.QRect(10,50,140,21))

        checkbox_time12 = QtGui.QCheckBox("Time1 vs Time2", phasecavity_gbox)
        checkbox_time12.setGeometry(QtCore.QRect(10,80,140,21))

        checkbox_charge1 = QtGui.QCheckBox("Charge1", phasecavity_gbox)
        checkbox_charge1.setGeometry(QtCore.QRect(170,30,140,21))

        checkbox_charge2 = QtGui.QCheckBox("Charge2", phasecavity_gbox)
        checkbox_charge2.setGeometry(QtCore.QRect(170,50,100,21))

        checkbox_charge12 = QtGui.QCheckBox("Charge1 vs Charge2", phasecavity_gbox)
        checkbox_charge12.setGeometry(QtCore.QRect(170,80,140,21))

        checkbox_1tch = QtGui.QCheckBox("Time1 vs Charge1", phasecavity_gbox)
        checkbox_1tch.setGeometry(QtCore.QRect(330,30,140,21))

        checkbox_2tch = QtGui.QCheckBox("Time1 vs Charge1", phasecavity_gbox)
        checkbox_2tch.setGeometry(QtCore.QRect(330,50,140,21))

        widget_layout.addWidget(ebeam_gbox)
        widget_layout.addWidget(gasdet_gbox)
        widget_layout.addWidget(phasecavity_gbox)
        widget_layout.addWidget(self.apply_button)
        widget_layout.setAlignment( self.apply_button, QtCore.Qt.AlignRight )

        
class JobConfigGui( QtGui.QWidget ):
    """JobConfigGui represents the the panel to configure the pyana job
    """
    def __init__(self, pyana_config, parent = None ):
        """Initialize with:
        @param pyana_config a pointer to the container object for pyana configurations
        @param parent       parent widget, if any
        """ 
        QtGui.QWidget.__init__(self, parent)
        self.pyana_cfg = pyana_config

        self.make_layout()
        

    def make_layout(self):
        general_layout = QtGui.QVBoxLayout(self)

        # has two sub-widgets: pyana options, general plotting options        
        run_options_box = QtGui.QGroupBox("Pyana run options:")
        run_options_layout = QtGui.QVBoxLayout()
        run_options_box.setLayout( run_options_layout )

        plot_options_box = QtGui.QGroupBox("Plotting options:")
        plot_options_layout = QtGui.QVBoxLayout()
        plot_options_box.setLayout( plot_options_layout )

        apply_button = QtGui.QPushButton("Apply")
        apply_button.setMaximumWidth(90)
        
        general_layout.addWidget( run_options_box )
        general_layout.addWidget( plot_options_box )
        general_layout.addWidget( apply_button )
        general_layout.setAlignment( apply_button, QtCore.Qt.AlignRight )

        self.apply_button = apply_button

        # Global Display mode
        self.pyana_cfg.jobconfig.displaymode = "SlideShow"
        dmode_status = QtGui.QLabel("Display mode is %s" % self.pyana_cfg.jobconfig.displaymode)
        dmode_menu = QtGui.QComboBox()
        dmode_menu.setMaximumWidth(90)
        dmode_menu.addItem("NoDisplay")
        dmode_menu.addItem("SlideShow")
        dmode_menu.addItem("Interactive")
        dmode_menu.setCurrentIndex(1) # SlideShow

        dmode_layout = QtGui.QHBoxLayout()
        dmode_layout.addWidget(dmode_status)
        dmode_layout.addWidget(dmode_menu)

        plot_options_layout.addLayout(dmode_layout, QtCore.Qt.AlignRight)

        def dmode_changed():
            mode = dmode_menu.currentText()
            dmode_status.setText("Display mode is %s"%mode)
            self.pyana_cfg.jobconfig.displaymode = mode

            if mode == "NoDisplay" :
                plotn_status.setText("Plot only after all events")
                self.pyana_cfg.jobconfig.plot_n = 0

        self.connect(dmode_menu,  QtCore.SIGNAL('currentIndexChanged(int)'), dmode_changed )


        # run options
        run_n_status = QtGui.QLabel("Process all events (or enter how many to process)")
        run_n_enter = QtGui.QLineEdit("")
        run_n_enter.setMaximumWidth(90)
        run_n_button = QtGui.QPushButton("Update") 

        run_n_layout = QtGui.QHBoxLayout()
        run_n_layout.addWidget(run_n_status)
        run_n_layout.addStretch()
        run_n_layout.addWidget(run_n_enter)
        run_n_layout.addWidget(run_n_button)

        run_options_layout.addLayout(run_n_layout, QtCore.Qt.AlignRight )

        def run_n_changed():
            text = run_n_enter.text()            
            if text == "" or text == "all" or text == "All" or text == "None" :
                run_n_status.setText("Process all events (or enter how many to process)")
                self.pyana_cfg.jobconfig.run_n = None
            else :
                run_n_status.setText("Process %s events"% text )
                run_n_enter.setText("")
                self.pyana_cfg.jobconfig.run_n = text

        self.connect(run_n_button, QtCore.SIGNAL('clicked()'), run_n_changed )
        self.connect(run_n_enter, QtCore.SIGNAL('returnPressed()'), run_n_changed )


        # skip options
        skip_n_status = QtGui.QLabel("Skip no events (or enter how many to skip)")
        skip_n_enter = QtGui.QLineEdit("")
        skip_n_enter.setMaximumWidth(90)
        skip_n_button = QtGui.QPushButton("Update") 

        skip_n_layout = QtGui.QHBoxLayout()
        skip_n_layout.addWidget(skip_n_status)
        skip_n_layout.addStretch()
        skip_n_layout.addWidget(skip_n_enter)
        skip_n_layout.addWidget(skip_n_button)

        run_options_layout.addLayout(skip_n_layout, QtCore.Qt.AlignRight )

        def skip_n_changed():
            text = skip_n_enter.text()
            if text == "" or text == "0" or text == "no" or text == "None" :
                skip_n_status.setText("Skip no events (or enter how many to skip)")
                self.pyana_cfg.jobconfig.skip_n = None
            else :
                skip_n_status.setText("Skip the first %s events of xtc file" % text )
                skip_n_enter.setText("")
                self.pyana_cfg.jobconfig.skip_n = text
                
        self.connect(skip_n_enter, QtCore.SIGNAL('returnPressed()'), skip_n_changed )
        self.connect(skip_n_button, QtCore.SIGNAL('clicked()'), skip_n_changed )


        # Multiprocessing?
        mproc_status = QtGui.QLabel("Multiprocessing? No, single CPU")
        mproc_menu = QtGui.QComboBox()
        mproc_menu.setMaximumWidth(90)
        for i in range (0,mp.cpu_count()):
            mproc_menu.addItem(str(i+1))
        mproc_menu.setCurrentIndex(0) # SlideShow
        

        mproc_layout = QtGui.QHBoxLayout()
        mproc_layout.addWidget(mproc_status)
        mproc_layout.addStretch()
        mproc_layout.addWidget(mproc_menu)

        run_options_layout.addLayout(mproc_layout, QtCore.Qt.AlignRight)

        def mproc_changed():
            text = str(mproc_menu.currentText())
            if ( text is None ) or ( text == "1" ):
                mproc_status.setText("Multiprocessing? No, single CPU")
            else:
                mproc_status.setText("Multiprocessing with %s CPUs"%text)
                self.pyana_cfg.jobconfig.num_cpu = text

        self.connect(mproc_menu, QtCore.SIGNAL('currentIndexChanged(int)'), mproc_changed )


        # plot every N events
        self.pyana_cfg.jobconfig.plot_n = 10
        plotn_status = QtGui.QLabel("Plot every %s events" % 10)
        plotn_enter = QtGui.QLineEdit()
        plotn_enter.setMaximumWidth(90)
        plotn_button = QtGui.QPushButton("Update") 
        
        plot_n_layout = QtGui.QHBoxLayout()
        plot_n_layout.addWidget(plotn_status)
        plot_n_layout.addStretch()
        plot_n_layout.addWidget(plotn_enter)
        plot_n_layout.addWidget(plotn_button)

        plot_options_layout.addLayout(plot_n_layout, QtCore.Qt.AlignRight )

        def plotn_changed():
            plotN = str( plotn_enter.text() )
            plotn_enter.setText("")
            
            if (plotN == "" or plotN == "0" or plotN == "all" or plotN == "All" ):
                plotN = None
                plotn_status.setText("Plot only after all events")
            self.pyana_cfg.jobconfig.plot_n = plotN
            if plotN is not None: 
                plotn_status.setText("Plot every %s events" % plotN )
                if self.pyana_cfg.jobconfig.displaymode == "NoDisplay" :
                    self.pyana_cfg.jobconfig.displaymode = "SlideShow"
                    dmode_status.setText("Display mode is %s"%self.pyana_cfg.jobconfig.displaymode)
                    dmode_menu.setCurrentIndex(1) 
        self.connect(plotn_enter, QtCore.SIGNAL('returnPressed()'), plotn_changed )
        self.connect(plotn_button, QtCore.SIGNAL('clicked()'), plotn_changed )


        # Accumulate N events (reset after N events)
        accumn_dtext = "Accumulate all events (or enter how many to accumulate)"
        accumn_status = QtGui.QLabel(accumn_dtext)
        accumn_enter = QtGui.QLineEdit()
        accumn_enter.setMaximumWidth(90)
        accumn_button = QtGui.QPushButton("Update") 

        accum_n_layout = QtGui.QHBoxLayout()
        accum_n_layout.addWidget(accumn_status)
        accum_n_layout.addStretch()
        accum_n_layout.addWidget(accumn_enter)
        accum_n_layout.addWidget(accumn_button)

        plot_options_layout.addLayout(accum_n_layout, QtCore.Qt.AlignRight )

        def accumn_changed():
            accuN = str( accumn_enter.text() )
            accumn_enter.setText("")
            if ( accuN == "" or accuN == "0" or accuN == "all" or accuN == "All" ):
                accuN = None
                accumn_status.setText(accumn_dtext)
            self.pyana_cfg.jobconfig.accum_n = accuN
            if accuN is not None :
                accumn_status.setText("Accumulate %s events (reset after)" % accuN )
                if self.pyana_cfg.jobconfig.displaymode == "NoDisplay" :
                    self.pyana_cfg.jobconfig.displaymode = "SlideShow"
                    dmode_status.setText("Display mode is %s"%self.pyana_cfg.jobconfig.displaymode)
                    dmode_menu.setCurrentIndex(1)                    
        self.connect(accumn_enter, QtCore.SIGNAL('returnPressed()'), accumn_changed )
        self.connect(accumn_button, QtCore.SIGNAL('clicked()'), accumn_changed )


        # Drop into iPython session at the end of the job?
        bool_string = { False: "No" , True: "Yes" }

        self.pyana_cfg.jobconfig.ipython = False
        ipython_status = QtGui.QLabel("Drop into iPython at the end of the job?  %s" \
                                      % bool_string[ self.pyana_cfg.jobconfig.ipython ] )
        ipython_menu = QtGui.QComboBox()
        ipython_menu.setMaximumWidth(90)
        ipython_menu.addItem("No")
        ipython_menu.addItem("Yes")

        ipython_layout = QtGui.QHBoxLayout()
        ipython_layout.addWidget(ipython_status)
        ipython_layout.addWidget(ipython_menu)

        plot_options_layout.addLayout(ipython_layout)

        def ipython_changed():
            status = bool( ipython_menu.currentIndex() )
            status_text = bool_string[ status ]
            ipython_status.setText("Drop into iPython at the end of the job?  %s"%status_text)
            self.pyana_cfg.jobconfig.ipython = status
            
        self.connect(ipython_menu,  QtCore.SIGNAL('currentIndexChanged(int)'), ipython_changed )


