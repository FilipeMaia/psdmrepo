#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigParametersCorAna...
#
#------------------------------------------------------------------------

"""Is intended as a storage for configuration parameters for CorAna project.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

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
from copy import deepcopy
 
#-----------------------------
# Imports for other modules --
#-----------------------------
#import ConfigParameters as cpbase
from ConfigParameters import * # ConfigParameters
from Logger import logger

from PyQt4 import QtGui        # for icons only...
import AppDataPath as apputils # for icons

#---------------------
#  Class definition --
#---------------------

class ConfigParametersCorAna ( ConfigParameters ) :
    """Is intended as a storage for configuration parameters for CorAna project.
    #@see BaseClass ConfigParameters
    #@see OtherClass Parameters
    """

    list_of_dets   = ['CSPAD', 'CSPAD2x2', 'Princeton', 'pnCCD', 'Tm6740', 'Opal1000', 'Opal2000', 'Opal4000', 'Opal8000']
    list_pars = []

    def __init__ ( self, fname=None ) :
        """Constructor.
        @param fname  the file name with configuration parameters, if not specified then it will be set to the default value at declaration.
        """
        ConfigParameters.__init__(self)
        self.declareCorAnaParameters()
        self.readParametersFromFile ( fname )
        self.initRunTimeParameters()
        self.defineStyles()
  
    def initRunTimeParameters( self ) :
        self.char_expand = u' \u25BE' # down-head triangle

        self.iconsAreLoaded  = False
        self.plotarray_is_on = False
        self.plotg2_is_on    = False

        self.procDarkStatus  = 0 # 0=inctive, 1=scan, 2=averaging, 3=both
        self.procDataStatus  = 0 # 0=inctive, 1=scan, 2=averaging, 3=both
        self.autoRunStatus   = 0 # 0=inctive, 1=split, 2=process, 3=merge

        #self.plotimgspe      = None
        self.plotimgspe_g    = None

        self.list_of_dets_lower = [det.lower() for det in self.list_of_dets]
        

#-----------------------------

    def setIcons(self) :

        if self.iconsAreLoaded : return

        self.iconsAreLoaded = True

        path_icon_contents      = apputils.AppDataPath('CorAna/icons/contents.png').path()
        path_icon_mail_forward  = apputils.AppDataPath('CorAna/icons/mail-forward.png').path()
        path_icon_button_ok     = apputils.AppDataPath('CorAna/icons/button_ok.png').path()
        path_icon_button_cancel = apputils.AppDataPath('CorAna/icons/button_cancel.png').path()
        path_icon_exit          = apputils.AppDataPath('CorAna/icons/exit.png').path()
        path_icon_home          = apputils.AppDataPath('CorAna/icons/home.png').path()
        path_icon_redo          = apputils.AppDataPath('CorAna/icons/redo.png').path()
        path_icon_undo          = apputils.AppDataPath('CorAna/icons/undo.png').path()
        path_icon_reload        = apputils.AppDataPath('CorAna/icons/reload.png').path()
        path_icon_save          = apputils.AppDataPath('CorAna/icons/save.png').path()
        path_icon_save_cfg      = apputils.AppDataPath('CorAna/icons/fileexport.png').path()
        path_icon_edit          = apputils.AppDataPath('CorAna/icons/edit.png').path()
        path_icon_browser       = apputils.AppDataPath('CorAna/icons/fileopen.png').path()
        path_icon_monitor       = apputils.AppDataPath('CorAna/icons/icon-monitor.png').path()
        path_icon_unknown       = apputils.AppDataPath('CorAna/icons/icon-unknown.png').path()
        path_icon_logviewer     = apputils.AppDataPath('CorAna/icons/logviewer.png').path()
        path_icon_locked        = apputils.AppDataPath('CorAna/icons/locked-icon.png').path()
        path_icon_unlocked      = apputils.AppDataPath('CorAna/icons/unlocked-icon.png').path()


        self.icon_contents      = QtGui.QIcon(path_icon_contents )
        self.icon_mail_forward  = QtGui.QIcon(path_icon_mail_forward)
        self.icon_button_ok     = QtGui.QIcon(path_icon_button_ok)
        self.icon_button_cancel = QtGui.QIcon(path_icon_button_cancel)
        self.icon_exit          = QtGui.QIcon(path_icon_exit     )
        self.icon_home          = QtGui.QIcon(path_icon_home     )
        self.icon_redo          = QtGui.QIcon(path_icon_redo     )
        self.icon_undo          = QtGui.QIcon(path_icon_undo     )
        self.icon_reload        = QtGui.QIcon(path_icon_reload   )
        self.icon_save          = QtGui.QIcon(path_icon_save     )
        self.icon_save_cfg      = QtGui.QIcon(path_icon_save_cfg )
        self.icon_edit          = QtGui.QIcon(path_icon_edit     )
        self.icon_browser       = QtGui.QIcon(path_icon_browser  )
        self.icon_monitor       = QtGui.QIcon(path_icon_monitor  )
        self.icon_unknown       = QtGui.QIcon(path_icon_unknown  )
        self.icon_logviewer     = QtGui.QIcon(path_icon_logviewer)
        self.icon_lock          = QtGui.QIcon(path_icon_locked   )
        self.icon_unlock        = QtGui.QIcon(path_icon_unlocked )


        #base_dir = '/usr/share/icons/Bluecurve/24x24/'
        #self.icon_contents      = QtGui.QIcon(base_dir + 'actions/contents.png')
        #self.icon_mail_forward  = QtGui.QIcon(base_dir + '../../gnome/24x24/actions/mail-forward.png')
        #self.icon_button_ok     = QtGui.QIcon(base_dir + 'actions/button_ok.png')
        #self.icon_button_cancel = QtGui.QIcon(base_dir + 'actions/button_cancel.png')
        #self.icon_exit          = QtGui.QIcon(base_dir + 'actions/exit.png')
        #self.icon_home          = QtGui.QIcon(base_dir + 'actions/gohome.png')
        #self.icon_redo          = QtGui.QIcon(base_dir + 'actions/redo.png')
        #self.icon_undo          = QtGui.QIcon(base_dir + 'actions/undo.png')
        #self.icon_reload        = QtGui.QIcon(base_dir + 'actions/reload.png')
        #self.icon_stop          = QtGui.QIcon(base_dir + 'actions/stop.png')
        #self.icon_save_cfg      = QtGui.QIcon(base_dir + 'actions/fileexport.png')
        #self.icon_save          = QtGui.QIcon(base_dir + 'stock/stock-save.png')
        #self.icon_edit          = QtGui.QIcon(base_dir + 'actions/edit.png')
        #self.icon_browser       = QtGui.QIcon(base_dir + 'actions/fileopen.png')
        #self.icon_monitor       = QtGui.QIcon(base_dir + 'apps/icon-monitor.png')
        #self.icon_unknown       = QtGui.QIcon(base_dir + 'apps/icon-unknown.png')
        #self.icon_logviewer     = QtGui.QIcon(base_dir + '../32x32/apps/logviewer.png')


        self.icon_logger        = self.icon_edit
        self.icon_help          = self.icon_unknown
        self.icon_reset         = self.icon_reload


#-----------------------------

        
    def declareCorAnaParameters( self ) :
        # Possible typs for declaration : 'str', 'int', 'long', 'float', 'bool' 

        # GUIInstrExpRun.py.py
#       self.fname_cp           = self.declareParameter( name='FNAME_CONFIG_PARS', val_def='confpars.txt', type='str' ) 
#        self.fname_ped          = self.declareParameter( name='FNAME_PEDESTALS',   val_def='my_ped.txt',   type='str' ) 
#        self.fname_dat          = self.declareParameter( name='FNAME_DATA',        val_def='my_dat.txt',   type='str' ) 
#        self.instr_dir          = self.declareParameter( name='INSTRUMENT_DIR',    val_def='/reg/d/psdm',  type='str' ) 
#        self.instr_name         = self.declareParameter( name='INSTRUMENT_NAME',   val_def='XCS',          type='str' ) 
#        self.exp_name           = self.declareParameter( name='EXPERIMENT_NAME',   val_def='xcsi0112',     type='str' ) 
#        self.str_run_number     = self.declareParameter( name='RUN_NUMBER',        val_def='0015',         type='str' ) 
#        self.str_run_number_dark= self.declareParameter( name='RUN_NUMBER_DARK',   val_def='0014',         type='str' ) 

        # GUIMainTB.py
        # GUIMainSplit.py
        self.current_tab       = self.declareParameter( name='CURRENT_TAB'      , val_def='Files',        type='str' )

        # GUILogger.py
        self.log_level         = self.declareParameter( name='LOG_LEVEL_OF_MSGS', val_def='info',         type='str' )

        # GUIFiles.py
        self.current_file_tab  = self.declareParameter( name='CURRENT_FILE_TAB' , val_def='Work/Results', type='str' )

        # GUIRun.py
        self.current_run_tab   = self.declareParameter( name='CURRENT_RUN_TAB' , val_def='Input', type='str' )

        # GUIWorkResDirs.py
        self.dir_work          = self.declareParameter( name='DIRECTORY_WORK',        val_def='./work',       type='str' )
        self.dir_results       = self.declareParameter( name='DIRECTORY_RESULTS',     val_def='./results',    type='str' )
        self.fname_prefix      = self.declareParameter( name='FILE_NAME_PREFIX',      val_def='cora-',        type='str' )
        #self.fname_prefix_cora = self.declareParameter( name='FILE_NAME_PREFIX_CORA', val_def='cora-proc',    type='str' )

        # GUIDark.py
        self.use_dark_xtc_all  = self.declareParameter( name='USE_DARK_XTC_ALL_CHUNKS', val_def=True,  type='bool' )
        self.in_dir_dark       = self.declareParameter( name='IN_DIRECTORY_DARK', val_def='/reg/d/ana12/xcs/xcsi0112/xtc',type='str' )
        self.in_file_dark      = self.declareParameter( name='IN_FILE_NAME_DARK', val_def='e167-r0020-s00-c00.xtc',type='str' )

        # GUIFlatField.py
        self.ccdcorr_flatfield  = self.declareParameter( name='CCD_CORRECTION_FLATFIELD',     val_def=False,  type='bool' )
        self.dname_flat         = self.declareParameter( name='DIRECTORY_FLAT', val_def='.',type='str' )
        self.fname_flat         = self.declareParameter( name='FILE_NAME_FLAT', val_def='flat_field.txt',type='str' )
        #self.in_dir_flat       = self.declareParameter( name='IN_DIRECTORY_FLAT', val_def='/reg/d/psdm/XCS/xcsi0112/xtc',type='str' )
        #self.in_file_flat      = self.declareParameter( name='IN_FILE_NAME_FLAT', val_def='e167-r0020-s00-c00.xtc',type='str' )

        # GUIBlemish.py
        self.ccdcorr_blemish    = self.declareParameter( name='CCD_CORRECTION_BLEMISH',       val_def=False,  type='bool' )
        self.dname_blem         = self.declareParameter( name='DIRECTORY_BLEM', val_def='.',type='str' )
        self.fname_blem         = self.declareParameter( name='FILE_NAME_BLEM', val_def='blemish.txt',type='str' )
        #self.in_dir_blem       = self.declareParameter( name='IN_DIRECTORY_BLEM', val_def='/reg/d/psdm/XCS/xcsi0112/xtc',type='str' )
        #self.in_file_blem      = self.declareParameter( name='IN_FILE_NAME_BLEM', val_def='e167-r0020-s00-c00.xtc',type='str' )

        # GUIData.py
        self.use_data_xtc_all   = self.declareParameter( name='USE_DATA_XTC_ALL_CHUNKS', val_def=True,  type='bool' )
        self.is_active_data_gui = self.declareParameter( name='IS_ACTIVE_DATA_GUI', val_def=True,  type='bool' )
        self.in_dir_data        = self.declareParameter( name='IN_DIRECTORY_DATA',  val_def='/reg/d/ana12/xcs/xcsi0112/xtc',type='str' )
        self.in_file_data       = self.declareParameter( name='IN_FILE_NAME_DATA',  val_def='e167-r0015-s00-c00.xtc',type='str' )

        # GUISetupBeamZero.py
        self.x_coord_beam0      = self.declareParameter( name='X_COORDINATE_BEAM_ZERO',   val_def=722.0,      type='float' ) 
        self.y_coord_beam0      = self.declareParameter( name='Y_COORDINATE_BEAM_ZERO',   val_def=632.0,      type='float' ) 
        self.x0_pos_in_beam0    = self.declareParameter( name='X_CCD_POS_IN_BEAM_ZERO',   val_def=0.0,        type='float' ) 
        self.y0_pos_in_beam0    = self.declareParameter( name='Y_CCD_POS_IN_BEAM_ZERO',   val_def=0.0,        type='float' ) 

        # GUISetupSpecular.py
        self.x_coord_specular   = self.declareParameter( name='X_COORDINATE_SPECULAR',    val_def=0.0,        type='float' ) 
        self.y_coord_specular   = self.declareParameter( name='Y_COORDINATE_SPECULAR',    val_def=0.0,        type='float' ) 
        self.x0_pos_in_specular = self.declareParameter( name='X_CCD_POS_IN_SPECULAR',    val_def=0.0,        type='float' ) 
        self.y0_pos_in_specular = self.declareParameter( name='Y_CCD_POS_IN_SPECULAR',    val_def=0.0,        type='float' ) 

        # GUISetupData.py
        self.x0_pos_in_data     = self.declareParameter( name='X_CCD_POS_IN_DATA',        val_def=0.0,        type='float' )
        self.y0_pos_in_data     = self.declareParameter( name='Y_CCD_POS_IN_DATA',        val_def=0.0,        type='float' )

        # GUISetupInfoLeft.py
        self.sample_det_dist    = self.declareParameter( name='SAMPLE_TO_DETECTOR_DISTANCE', val_def=7500.0,      type='float' )
        self.exp_setup_geom     = self.declareParameter( name='EXP_SETUP_GEOMETRY',          val_def='Beam Zero', type='str' )
        self.photon_energy      = self.declareParameter( name='PHOTON_ENERGY',               val_def=7.6543,      type='float' )
        self.nominal_angle      = self.declareParameter( name='NOMINAL_ANGLE',               val_def=-1,          type='float' )
        self.real_angle         = self.declareParameter( name='REAL_ANGLE',                  val_def=-1,          type='float' )
        self.tilt_angle         = self.declareParameter( name='TILT_ANGLE',                  val_def=-1,          type='float' )

        # GUIImgSizePosition.py
        self.col_begin          = self.declareParameter( name='IMG_COL_BEGIN',        val_def=0,             type='int' )
        self.col_end            = self.declareParameter( name='IMG_COL_END',          val_def=1340,          type='int' )
        self.row_begin          = self.declareParameter( name='IMG_ROW_BEGIN',        val_def=0,             type='int' )
        self.row_end            = self.declareParameter( name='IMG_ROW_END',          val_def=1300,          type='int' )

        # GUIKineticMode.py
        self.kin_mode           = self.declareParameter( name='KINETICS_MODE',        val_def='Non-Kinetics',type='str' )
        self.kin_win_size       = self.declareParameter( name='KINETICS_WIN_SIZE',    val_def=1,             type='int' )
        self.kin_top_row        = self.declareParameter( name='KINETICS_TOP_ROW',     val_def=2,             type='int' )
        self.kin_slice_first    = self.declareParameter( name='KINETICS_SLICE_FIRST', val_def=3,             type='int' )
        self.kin_slice_last     = self.declareParameter( name='KINETICS_SLICE_LAST',  val_def=4,             type='int' )

        # GUISetupPars.py
        self.bat_num           = self.declareParameter( name='BATCH_NUM',             val_def= 1,       type='int' )
        self.bat_num_max       = self.declareParameter( name='BATCH_NUM_MAX',         val_def= 9,       type='int' )
        #self.bat_data_is_used  = self.declareParameter( name='BATCH_DATA_IS_USED',    val_def=True,     type='bool' )
        self.bat_data_start    = self.declareParameter( name='BATCH_DATA_START',      val_def= 1,       type='int' )
        self.bat_data_end      = self.declareParameter( name='BATCH_DATA_END'  ,      val_def=-1,       type='int' )
        self.bat_data_total    = self.declareParameter( name='BATCH_DATA_TOTAL',      val_def=-1,       type='int' )
        self.bat_data_time     = self.declareParameter( name='BATCH_DATA_TIME' ,      val_def=-1.0,     type='float' )
        self.bat_data_dt_ave   = self.declareParameter( name='BATCH_DATA_DT_AVE',     val_def=-1.0,     type='float' )
        self.bat_data_dt_rms   = self.declareParameter( name='BATCH_DATA_DT_RMS',     val_def=0.0,      type='float' )

        self.bat_dark_is_used  = self.declareParameter( name='BATCH_DARK_IS_USED',    val_def=True,     type='bool' )
        self.bat_dark_start    = self.declareParameter( name='BATCH_DARK_START',      val_def= 1,       type='int' )
        self.bat_dark_end      = self.declareParameter( name='BATCH_DARK_END'  ,      val_def=-1,       type='int' )
        self.bat_dark_total    = self.declareParameter( name='BATCH_DARK_TOTAL',      val_def=-1,       type='int' )
        self.bat_dark_time     = self.declareParameter( name='BATCH_DARK_TIME' ,      val_def=-1.0,     type='float' )
        self.bat_dark_dt_ave   = self.declareParameter( name='BATCH_DARK_DT_AVE',     val_def=-1.0,     type='float' )
        self.bat_dark_dt_rms   = self.declareParameter( name='BATCH_DARK_DT_RMS',     val_def=0.0,      type='float' )
        #self.bat_flat_is_used  = self.declareParameter( name='BATCH_FLAT_IS_USED',    val_def=True,     type='bool' )
        self.bat_flat_start    = self.declareParameter( name='BATCH_FLAT_START',      val_def= 1,       type='int' )
        self.bat_flat_end      = self.declareParameter( name='BATCH_FLAT_END'  ,      val_def=-1,       type='int' )
        self.bat_flat_total    = self.declareParameter( name='BATCH_FLAT_TOTAL',      val_def=-1,       type='int' )
        self.bat_flat_time     = self.declareParameter( name='BATCH_FLAT_TIME' ,      val_def=-1.0,     type='float' )
        self.bat_queue         = self.declareParameter( name='BATCH_QUEUE',           val_def='psfehq', type='str' )
        self.bat_det_info      = self.declareParameter( name='BATCH_DET_INFO',        val_def='DetInfo(:Princeton)',  type='str' )
        #self.bat_det_info      = self.declareParameter( name='BATCH_DET_INFO',        val_def='DetInfo(XcsBeamline.0:Princeton.0)', type='str' )
        self.bat_img_rec_mod   = self.declareParameter( name='BATCH_IMG_REC_MODULE',  val_def='ImgAlgos.PrincetonImageProducer',  type='str' )
        self.detector          = self.declareParameter( name='DETECTOR',              val_def=self.list_of_dets[2], type='str' )

        # BatchLogParser.py
        self.bat_img_rows      = self.declareParameter( name='BATCH_IMG_ROWS',      val_def= 1300,       type='int' )
        self.bat_img_cols      = self.declareParameter( name='BATCH_IMG_COLS',      val_def= 1340,       type='int' )
        self.bat_img_size      = self.declareParameter( name='BATCH_IMG_SIZE',      val_def= 1300*1340,  type='int' )
        self.bat_img_nparts    = self.declareParameter( name='BATCH_IMG_NPARTS',    val_def=  8,         type='int' )

        # GUIAnaSettingsLeft.py
        self.ana_type          = self.declareParameter( name='ANA_TYPE',              val_def='Static',type='str' )

        self.ana_stat_meth_q   = self.declareParameter( name='ANA_STATIC_METHOD_Q',       val_def='evenly-spaced',type='str' )
        self.ana_stat_meth_phi = self.declareParameter( name='ANA_STATIC_METHOD_PHI',     val_def='evenly-spaced',type='str' )
        self.ana_dyna_meth_q   = self.declareParameter( name='ANA_DYNAMIC_METHOD_Q',      val_def='evenly-spaced',type='str' )
        self.ana_dyna_meth_phi = self.declareParameter( name='ANA_DYNAMIC_METHOD_PHI',    val_def='evenly-spaced',type='str' )

        self.ana_stat_part_q   = self.declareParameter( name='ANA_STATIC_PARTITION_Q',    val_def='50',type='str' )
        self.ana_stat_part_phi = self.declareParameter( name='ANA_STATIC_PARTITION_PHI',  val_def='6',type='str' )
        self.ana_dyna_part_q   = self.declareParameter( name='ANA_DYNAMIC_PARTITION_Q',   val_def='4',type='str' )
        self.ana_dyna_part_phi = self.declareParameter( name='ANA_DYNAMIC_PARTITION_PHI', val_def='3',type='str' )

        self.ana_mask_type     = self.declareParameter( name='ANA_MASK_TYPE',             val_def='no-mask',type='str' )
        self.ana_mask_fname    = self.declareParameter( name='ANA_MASK_FILE',             val_def='mask-roi.txt',type='str' )
        self.ana_mask_dname    = self.declareParameter( name='ANA_MASK_DIRECTORY',        val_def='.',type='str' )

        # GUIListOfTau.py
        self.ana_tau_list_type  = self.declareParameter( name='ANA_TAU_LIST_TYPE',             val_def='auto',type='str' )
        self.ana_tau_list_fname = self.declareParameter( name='ANA_TAU_LIST_FILE',             val_def='tau-list.txt',type='str' )
        self.ana_tau_list_dname = self.declareParameter( name='ANA_TAU_LIST_DIRECTORY',        val_def='.',type='str' )

        # GUIAnaSettingsRight.py
        self.ana_ndelays       = self.declareParameter( name='ANA_NDELAYS_PER_MTAU_LEVEL',       val_def=4,     type='int' )
        self.ana_nslice_delays = self.declareParameter( name='ANA_NSLICE_DELAYS_PER_MTAU_LEVEL', val_def=4,     type='int' )
        self.ana_npix_to_smooth= self.declareParameter( name='ANA_NPIXELS_TO_SMOOTH',            val_def=100,   type='int' )
        self.ana_smooth_norm   = self.declareParameter( name='ANA_SMOOTH_SYM_NORM',              val_def=False, type='bool' )
        self.ana_two_corfuns   = self.declareParameter( name='ANA_TWO_TIME_CORFUNS_CONTROL',     val_def=False, type='bool' )
        self.ana_spec_stab     = self.declareParameter( name='ANA_CHECK_SPECKLE_STABILITY',      val_def=False, type='bool' )

        self.lld_type          = self.declareParameter( name='LOW_LEVEL_DISC_TYPE',              val_def='NONE',type='str' )
        self.lld_adu           = self.declareParameter( name='LOW_LEVEL_DISC_ADU',               val_def=20,    type='float' )
        self.lld_rms           = self.declareParameter( name='LOW_LEVEL_DISC_RMS',               val_def=2,     type='float' )

        self.res_ascii_out     = self.declareParameter( name='RES_ASCII_OUTPUT',                 val_def=True,  type='bool' )
        self.res_fit1          = self.declareParameter( name='RES_PERFORM_FIT1',                 val_def=False, type='bool' )
        self.res_fit2          = self.declareParameter( name='RES_PERFORM_FIT1',                 val_def=False, type='bool' )  
        self.res_fit_cust      = self.declareParameter( name='RES_PERFORM_FIT_CUSTOM',           val_def=False, type='bool' ) 
        self.res_png_out       = self.declareParameter( name='RES_PNG_FILES',                    val_def=False, type='bool' )  
        self.res_save_log      = self.declareParameter( name='RES_SAVE_LOG_FILE',                val_def=False, type='bool' )  

        # GUILoadResults.py
        self.res_load_mode     = self.declareParameter( name='RES_LOAD_MODE',                    val_def='NONE',type='str' )
        self.res_fname         = self.declareParameter( name='RES_LOAD_FNAME',                   val_def='NONE',type='str' )

        # GUISystemSettingsRight.py
        self.thickness_type          = self.declareParameter( name='THICKNESS_TYPE',               val_def='NONORM',type='str' )
        self.thickness_sample        = self.declareParameter( name='THICKNESS_OF_SAMPLE',          val_def=-1,      type='float' )
        self.thickness_attlen        = self.declareParameter( name='THICKNESS_ATTENUATION_LENGTH', val_def=-2,      type='float' )
        self.ccd_orient              = self.declareParameter( name='CCD_ORIENTATION',              val_def='0',     type='str' )
        self.y_is_flip               = self.declareParameter( name='Y_IS_FLIPPED',                 val_def='True',  type='bool' )

        # GUICCDSettings.py
        self.ccdset_pixsize          = self.declareParameter( name='CCD_SETTINGS_PIXEL_SIZE',      val_def=0.02,  type='float' )
        self.ccdset_adcsatu          = self.declareParameter( name='CCD_SETTINGS_ADC_SATTURATION', val_def=65535, type='int' )
        self.ccdset_aduphot          = self.declareParameter( name='CCD_SETTINGS_ADU_PER_PHOTON',  val_def=10,    type='float' )
        self.ccdset_ccdeff           = self.declareParameter( name='CCD_SETTINGS_EFFICIENCY',      val_def=0.5,   type='float' )
        self.ccdset_ccdgain          = self.declareParameter( name='CCD_SETTINGS_GAIN',            val_def=1.0,   type='float' )

        self.mask_hot_thr            = self.declareParameter( name='MASK_HOT_PIX_ADU_THR_ON_RMS',  val_def=10.0,  type='float' )
        self.mask_hot_is_used        = self.declareParameter( name='MASK_HOT_PIX_IS_USED',         val_def=True,  type='bool' )


        # GUIELogPostingDialog.py 
        # GUIELogPostingFields.py 
        #self.elog_post_cbx_state = self.declareParameter( name='ELOG_POST_CBX_STATE',     val_def=True,        type='bool' )
        self.elog_post_rad       = self.declareParameter( name='ELOG_POST_RAD_STATE',     val_def='Default',   type='str' )
        self.elog_post_ins       = self.declareParameter( name='ELOG_POST_INSTRUMENT',    val_def='AMO',       type='str' )
        self.elog_post_exp       = self.declareParameter( name='ELOG_POST_EXPERIMENT',    val_def='amodaq09',  type='str' )
        self.elog_post_run       = self.declareParameter( name='ELOG_POST_RUN',           val_def='825',       type='str' )
        self.elog_post_tag       = self.declareParameter( name='ELOG_POST_TAG',           val_def='TAG1',      type='str' )
        self.elog_post_res       = self.declareParameter( name='ELOG_POST_RESPONCE',      val_def='None',      type='str' )
        self.elog_post_msg       = self.declareParameter( name='ELOG_POST_MESSAGE',       val_def='EMPTY MSG', type='str' )
        self.elog_post_att       = self.declareParameter( name='ELOG_POST_ATTACHED_FILE', val_def='None',      type='str' )

        #GUIViewControl.py 
        self.vc_cbx_show_more    = self.declareParameter( name='SHOW_MORE_BUTTONS', val_def=True,  type='bool' )
        
#-----------------------------

        imon_names = [ ('BldInfo(FEEGasDetEnergy)',       None ,'str'), \
                       ('BldInfo(XCS-IPM-02)',            None ,'str'), \
                       ('BldInfo(XCS-IPM-mono)',          None ,'str'), \
                       ('DetInfo(XcsBeamline.1:Ipimb.4)', None ,'str'), \
                       ('DetInfo(XcsBeamline.1:Ipimb.5)', None ,'str') ]

        self.imon_name_list = self.declareListOfPars( 'IMON_NAMES', imon_names )

#-----------------------------

        imon_short_names = [ ('FEEGasDetEnergy',       None ,'str'), \
                             ('XCS-IPM-02',            None ,'str'), \
                             ('XCS-IPM-mono',          None ,'str'), \
                             ('Ipimb.4',               None ,'str'), \
                             ('Ipimb.5',               None ,'str') ]

        self.imon_short_name_list = self.declareListOfPars( 'IMON_SHORT_NAMES', imon_short_names )

#-----------------------------

        imon_cbxs = [ (True, True ,'bool'), \
                      (True, True ,'bool'), \
                      (True, True ,'bool'), \
                      (True, True ,'bool'), \
                      (True, True ,'bool') ]

        self.imon_ch1_list = self.declareListOfPars( 'IMON_CH1', deepcopy(imon_cbxs) )
        self.imon_ch2_list = self.declareListOfPars( 'IMON_CH2', deepcopy(imon_cbxs) )
        self.imon_ch3_list = self.declareListOfPars( 'IMON_CH3', deepcopy(imon_cbxs) )
        self.imon_ch4_list = self.declareListOfPars( 'IMON_CH4', deepcopy(imon_cbxs) )

#-----------------------------

        imon_norm_cbx = [ (False, False ,'bool'), \
                          (False, False ,'bool'), \
                          (False, False ,'bool'), \
                          (False, False ,'bool'), \
                          (False, False ,'bool') ]

        self.imon_norm_cbx_list = self.declareListOfPars( 'IMON_NORM_CBX', imon_norm_cbx )

#-----------------------------

        imon_sele_cbx = [ (False, False ,'bool'), \
                          (False, False ,'bool'), \
                          (False, False ,'bool'), \
                          (False, False ,'bool'), \
                          (False, False ,'bool') ]

        self.imon_sele_cbx_list = self.declareListOfPars( 'IMON_SELE_CBX', imon_sele_cbx )

#-----------------------------

        imon_sele_min = [ (-1., -1. ,'float'), \
                          (-1., -1. ,'float'), \
                          (-1., -1. ,'float'), \
                          (-1., -1. ,'float'), \
                          (-1., -1. ,'float') ]

        self.imon_sele_min_list = self.declareListOfPars( 'IMON_SELE_MIN', imon_sele_min )

#-----------------------------

        imon_sele_max = [ (-1., -1. ,'float'), \
                          (-1., -1. ,'float'), \
                          (-1., -1. ,'float'), \
                          (-1., -1. ,'float'), \
                          (-1., -1. ,'float') ]

        self.imon_sele_max_list = self.declareListOfPars( 'IMON_SELE_MAX', imon_sele_max )

#-----------------------------

        imon_norm_ave = [ ( 1.,  1. ,'float'), \
                          ( 1.,  1. ,'float'), \
                          ( 1.,  1. ,'float'), \
                          ( 1.,  1. ,'float'), \
                          ( 1.,  1. ,'float') ]

        self.imon_norm_ave_list = self.declareListOfPars( 'IMON_NORM_AVE', imon_norm_ave )

#-----------------------------

        self.imon_pars_list = zip( self.imon_name_list,
                                   self.imon_ch1_list,
                                   self.imon_ch2_list,
                                   self.imon_ch3_list,
                                   self.imon_ch4_list,
                                   self.imon_norm_cbx_list,
                                   self.imon_sele_cbx_list,
                                   self.imon_sele_min_list,
                                   self.imon_sele_max_list,
                                   self.imon_norm_ave_list,
                                   self.imon_short_name_list )
        #print self.imon_pars_list

#-----------------------------

    def defineStyles( self ) :
        self.styleYellowish = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        self.stylePink      = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        self.styleYellowBkg = "background-color: rgb(255, 255, 120); color: rgb(0, 0, 0);" # Pinkish
        self.styleGray      = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        self.styleGreenish  = "background-color: rgb(100, 255, 200); color: rgb(0, 0, 0);" # Greenish
        self.styleGreenPure = "background-color: rgb(150, 255, 150); color: rgb(0, 0, 0);" # Green
        self.styleBluish    = "background-color: rgb(200, 200, 255); color: rgb(0, 0, 0);" # Bluish
        self.styleWhite     = "background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);"
        self.styleRedBkgd   = "background-color: rgb(255,   0,   0); color: rgb(0, 0, 0);" # Red background
        #self.styleTitle  = "background-color: rgb(239, 235, 231, 255); color: rgb(100, 160, 100);" # Gray bkgd
        #self.styleTitle  = "color: rgb(150, 160, 100);"
        self.styleBlue   = "color: rgb(000, 000, 255);"
        self.styleBuriy  = "color: rgb(150, 100, 50);"
        self.styleRed    = "color: rgb(255, 0, 0);"
        self.styleGreen  = "color: rgb(0, 150, 0);"
        self.styleYellow = "color: rgb(0, 150, 150);"

        self.styleBkgd         = self.styleYellowish
        self.styleTitle        = self.styleBuriy
        self.styleLabel        = self.styleBlue
        self.styleEdit         = self.styleWhite
        self.styleEditInfo     = self.styleGreenish
        self.styleEditBad      = self.styleRedBkgd
        self.styleButton       = self.styleGray
        self.styleButtonOn     = self.styleBluish
        self.styleButtonClose  = self.stylePink
        self.styleButtonWarning= self.styleYellowBkg
        self.styleButtonGood   = self.styleGreenPure
        self.styleButtonBad    = self.stylePink
        self.styleBox          = self.styleGray
        self.styleCBox         = self.styleYellowish
        self.styleStatusGood   = self.styleGreen
        self.styleStatusWarning= self.styleYellow
        self.styleStatusAlarm  = self.styleRed
        self.styleTitleBold    = self.styleTitle + 'font-size: 18pt; font-family: Courier; font-weight: bold;'
        self.styleWhiteFixed   = self.styleWhite + 'font-family: Fixed;'

        self.colorEditInfo     = QtGui.QColor(100, 255, 200)
        self.colorEditBad      = QtGui.QColor(255,   0,   0)
        self.colorEdit         = QtGui.QColor('white')

    def printParsDirectly( self ) :
        logger.info('Direct use of parameter:' + self.fname_cp .name() + ' ' + self.fname_cp .value(), __name__ )
        logger.info('Direct use of parameter:' + self.fname_ped.name() + ' ' + self.fname_ped.value(), __name__ )     
        logger.info('Direct use of parameter:' + self.fname_dat.name() + ' ' + self.fname_dat.value(), __name__ )    

#-----------------------------

confpars = ConfigParametersCorAna (fname=getConfigFileFromInput())

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    confpars.printParameters()
    #confpars.printParsDirectly()
    confpars.saveParametersInFile()

    confpars.printListOfPars('IMON_NAMES')


    sys.exit ( 'End of test for ConfigParametersCorAna' )

#-----------------------------
