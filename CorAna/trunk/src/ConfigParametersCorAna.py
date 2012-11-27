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

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

#-----------------------------
# Imports for other modules --
#-----------------------------
#import ConfigParameters as cpbase
from ConfigParameters import * # ConfigParameters
from Logger import logger

#---------------------
#  Class definition --
#---------------------

class ConfigParametersCorAna ( ConfigParameters ) :
    """Is intended as a storage for configuration parameters for CorAna project.
    #@see BaseClass ConfigParameters
    #@see OtherClass Parameters
    """

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
        pass


    def declareCorAnaParameters( self ) :
        # Possible typs for declaration : 'str', 'int', 'long', 'float', 'bool' 

        # GUIInstrExpRun.py.py
#       self.fname_cp           = self.declareParameter( name='FNAME_CONFIG_PARS', val_def='confpars.txt', typ='str' ) 
#        self.fname_ped          = self.declareParameter( name='FNAME_PEDESTALS',   val_def='my_ped.txt',   typ='str' ) 
#        self.fname_dat          = self.declareParameter( name='FNAME_DATA',        val_def='my_dat.txt',   typ='str' ) 
#        self.instr_dir          = self.declareParameter( name='INSTRUMENT_DIR',    val_def='/reg/d/psdm',  typ='str' ) 
#        self.instr_name         = self.declareParameter( name='INSTRUMENT_NAME',   val_def='XCS',          typ='str' ) 
#        self.exp_name           = self.declareParameter( name='EXPERIMENT_NAME',   val_def='xcsi0112',     typ='str' ) 
#        self.str_run_number     = self.declareParameter( name='RUN_NUMBER',        val_def='0015',         typ='str' ) 
#        self.str_run_number_dark= self.declareParameter( name='RUN_NUMBER_DARK',   val_def='0014',         typ='str' ) 

        # GUIFiles.py
        self.in_dir_dark       = self.declareParameter( name='IN_DIRECTORY_DARK', val_def='/reg/d/psdm/XCS/xcsi0112/xtc',typ='str' )
        self.in_dir_flat       = self.declareParameter( name='IN_DIRECTORY_FLAT', val_def='/reg/d/psdm/XCS/xcsi0112/xtc',typ='str' )
        self.in_dir_blam       = self.declareParameter( name='IN_DIRECTORY_BLAM', val_def='/reg/d/psdm/XCS/xcsi0112/xtc',typ='str' )
        self.in_dir_data       = self.declareParameter( name='IN_DIRECTORY_DATA', val_def='/reg/d/psdm/XCS/xcsi0112/xtc',typ='str' )
        self.in_file_dark      = self.declareParameter( name='IN_FILE_NAME_DARK', val_def='e167-r0020-s00-c00.xtc',typ='str' )
        self.in_file_flat      = self.declareParameter( name='IN_FILE_NAME_FLAT', val_def='e167-r0020-s00-c00.xtc',typ='str' )
        self.in_file_blam      = self.declareParameter( name='IN_FILE_NAME_BLAM', val_def='e167-r0020-s00-c00.xtc',typ='str' )
        self.in_file_data      = self.declareParameter( name='IN_FILE_NAME_DATA', val_def='e167-r0020-s00-c00.xtc',typ='str' )
        self.dir_work          = self.declareParameter( name='DIRECTORY_WORK'   , val_def='./work',       typ='str' )
        self.dir_results       = self.declareParameter( name='DIRECTORY_RESULTS', val_def='./results',    typ='str' )
        self.fname_prefix      = self.declareParameter( name='FILE_NAME_PREFIX' , val_def='cora-',        typ='str' )
        self.log_level         = self.declareParameter( name='LOG_LEVEL_OF_MSGS', val_def='info',         typ='str' )
        self.current_file_tab  = self.declareParameter( name='CURRENT_FILE_TAB' , val_def='Work/Results', typ='str' )
        self.current_tab       = self.declareParameter( name='CURRENT_TAB'      , val_def='Files',        typ='str' )

        self.dname_blam         = self.declareParameter( name='DIRECTORY_BLAM', val_def='.',typ='str' )
        self.fname_blam         = self.declareParameter( name='FILE_NAME_BLAM', val_def='blamish.txt',typ='str' )
        self.dname_flat         = self.declareParameter( name='DIRECTORY_FLAT', val_def='.',typ='str' )
        self.fname_flat         = self.declareParameter( name='FILE_NAME_FLAT', val_def='flat_field.txt',typ='str' )

        # GUIBeamZeroPars.py
        self.x_coord_beam0      = self.declareParameter( name='X_COORDINATE_BEAM_ZERO',   val_def=1234.5,     typ='float' ) 
        self.y_coord_beam0      = self.declareParameter( name='Y_COORDINATE_BEAM_ZERO',   val_def=1216.5,     typ='float' ) 
        self.x0_pos_in_beam0    = self.declareParameter( name='X0_POS_IN_BEAM_ZERO',      val_def=-59,        typ='int' ) 
        self.y0_pos_in_beam0    = self.declareParameter( name='Y0_POS_IN_BEAM_ZERO',      val_def=175,        typ='int' ) 

        # GUISpecularPars.py
        self.x_coord_specular   = self.declareParameter( name='X_COORDINATE_SPECULAR', val_def=-1,     typ='float' ) 
        self.y_coord_specular   = self.declareParameter( name='Y_COORDINATE_SPECULAR', val_def=-2,     typ='float' ) 
        self.x0_pos_in_specular = self.declareParameter( name='X0_SPEC_IN_SPECULAR',   val_def=-3,     typ='int' ) 
        self.y0_pos_in_specular = self.declareParameter( name='Y0_SPEC_IN_SPECULAR',   val_def=-4,     typ='int' ) 

        # GUISetupInfoLeft.py
        self.sample_det_dist    = self.declareParameter( name='SAMPLE_TO_DETECTOR_DISTANCE', val_def=4000.1,          typ='float' )
        self.exp_setup_geom     = self.declareParameter( name='EXP_SETUP_GEOMETRY',          val_def='Transmission',  typ='str' )
        self.photon_energy      = self.declareParameter( name='PHOTON_ENERGY',               val_def=7.6543,          typ='float' )
        self.nominal_angle      = self.declareParameter( name='NOMINAL_ANGLE',               val_def=-1,              typ='float' )
        self.real_angle         = self.declareParameter( name='REAL_ANGLE',                  val_def=-1,              typ='float' )

        # GUIImgSizePosition.py
        self.col_begin          = self.declareParameter( name='IMG_COL_BEGIN',        val_def=0,             typ='int' )
        self.col_end            = self.declareParameter( name='IMG_COL_END',          val_def=1339,          typ='int' )
        self.row_begin          = self.declareParameter( name='IMG_ROW_BEGIN',        val_def=1,             typ='int' )
        self.row_end            = self.declareParameter( name='IMG_ROW_END',          val_def=1299,          typ='int' )
        self.x_frame_pos        = self.declareParameter( name='X_FRAME_POS',          val_def=-51,           typ='int' )
        self.y_frame_pos        = self.declareParameter( name='Y_FRAME_POS',          val_def=183,           typ='int' )

        # GUIKineticMode.py
        self.kin_mode           = self.declareParameter( name='KINETICS_MODE',        val_def='Non-Kinetics',typ='str' )
        self.kin_win_size       = self.declareParameter( name='KINETICS_WIN_SIZE',    val_def=1,             typ='int' )
        self.kin_top_row        = self.declareParameter( name='KINETICS_TOP_ROW',     val_def=2,             typ='int' )
        self.kin_slice_first    = self.declareParameter( name='KINETICS_SLICE_FIRST', val_def=3,             typ='int' )
        self.kin_slice_last     = self.declareParameter( name='KINETICS_SLICE_LAST',  val_def=4,             typ='int' )

        # GUISetupPars.py
        self.bat_num           = self.declareParameter( name='BATCH_NUM',             val_def= 1,       typ='int' )
        self.bat_num_max       = self.declareParameter( name='BATCH_NUM_MAX',         val_def= 9,       typ='int' )
        self.bat_data_start    = self.declareParameter( name='BATCH_DATA_START',      val_def= 1,       typ='int' )
        self.bat_data_end      = self.declareParameter( name='BATCH_DATA_END'  ,      val_def=100,      typ='int' )
        self.bat_data_total    = self.declareParameter( name='BATCH_DATA_TOTAL',      val_def=-1,       typ='int' )
        self.bat_data_time     = self.declareParameter( name='BATCH_DATA_TIME' ,      val_def=-1.0,     typ='float' )
        self.bat_dark_start    = self.declareParameter( name='BATCH_DARK_START',      val_def= 1,       typ='int' )
        self.bat_dark_end      = self.declareParameter( name='BATCH_DARK_END'  ,      val_def=100,      typ='int' )
        self.bat_dark_total    = self.declareParameter( name='BATCH_DARK_TOTAL',      val_def=-1,       typ='int' )
        self.bat_dark_time     = self.declareParameter( name='BATCH_DARK_TIME' ,      val_def=-1.0,     typ='float' )
        self.bat_dark_dt_ave   = self.declareParameter( name='BATCH_DARK_DT_AVE',     val_def=-1.0,     typ='float' )
        self.bat_dark_dt_rms   = self.declareParameter( name='BATCH_DARK_DT_RMS',     val_def=0.0,      typ='float' )
        self.bat_flat_start    = self.declareParameter( name='BATCH_FLAT_START',      val_def= 1,       typ='int' )
        self.bat_flat_end      = self.declareParameter( name='BATCH_FLAT_END'  ,      val_def=100,      typ='int' )
        self.bat_flat_total    = self.declareParameter( name='BATCH_FLAT_TOTAL',      val_def=-1,       typ='int' )
        self.bat_flat_time     = self.declareParameter( name='BATCH_FLAT_TIME' ,      val_def=-1.0,     typ='float' )
        self.bat_queue         = self.declareParameter( name='BATCH_QUEUE',           val_def='psfehq', typ='str' )
        self.bat_det_info      = self.declareParameter( name='BATCH_DET_INFO',        val_def='DetInfo(:Princeton)',  typ='str' )
        self.bat_img_rec_mod   = self.declareParameter( name='BATCH_IMG_REC_MODULE',  val_def='ImgAlgos.PrincetonImageProducer',  typ='str' )

        # GUIAnaSettingsLeft.py
        self.ana_type          = self.declareParameter( name='ANA_TYPE',              val_def='Static',typ='str' )

        self.ana_stat_meth_q   = self.declareParameter( name='ANA_STATIC_METHOD_Q',       val_def='evenly-spaced',typ='str' )
        self.ana_stat_meth_phi = self.declareParameter( name='ANA_STATIC_METHOD_PHI',     val_def='evenly-spaced',typ='str' )
        self.ana_dyna_meth_q   = self.declareParameter( name='ANA_DYNAMIC_METHOD_Q',      val_def='non-evenly-spaced',typ='str' )
        self.ana_dyna_meth_phi = self.declareParameter( name='ANA_DYNAMIC_METHOD_PHI',    val_def='evenly-spaced',typ='str' )

        self.ana_stat_part_q   = self.declareParameter( name='ANA_STATIC_PARTITION_Q',    val_def='1',typ='str' )
        self.ana_stat_part_phi = self.declareParameter( name='ANA_STATIC_PARTITION_PHI',  val_def='2',typ='str' )
        self.ana_dyna_part_q   = self.declareParameter( name='ANA_DYNAMIC_PARTITION_Q',   val_def='3',typ='str' )
        self.ana_dyna_part_phi = self.declareParameter( name='ANA_DYNAMIC_PARTITION_PHI', val_def='4',typ='str' )

        self.ana_mask_type     = self.declareParameter( name='ANA_MASK_TYPE',             val_def='no-mask',typ='str' )
        self.ana_mask_file     = self.declareParameter( name='ANA_MASK_FILE',             val_def='mask.txt',typ='str' )

        # GUIAnaSettingsRight.py
        self.ana_ndelays       = self.declareParameter( name='ANA_NDELAYS_PER_MTAU_LEVEL',       val_def=4,     typ='int' )
        self.ana_nslice_delays = self.declareParameter( name='ANA_NSLICE_DELAYS_PER_MTAU_LEVEL', val_def=4,     typ='int' )
        self.ana_npix_to_smooth= self.declareParameter( name='ANA_NPIXELS_TO_SMOOTH',            val_def=100,   typ='int' )
        self.ana_smooth_norm   = self.declareParameter( name='ANA_SMOOTH_SYM_NORM',              val_def=False, typ='bool' )
        self.ana_two_corfuns   = self.declareParameter( name='ANA_TWO_TIME_CORFUNS_CONTROL',     val_def=False, typ='bool' )
        self.ana_spec_stab     = self.declareParameter( name='ANA_CHECK_SPECKLE_STABILITY',      val_def=False, typ='bool' )

        self.lld_type          = self.declareParameter( name='LOW_LEVEL_DISC_TYPE',              val_def='NONE',typ='str' )
        self.lld_adu           = self.declareParameter( name='LOW_LEVEL_DISC_ADU',               val_def=15,    typ='float' )
        self.lld_rms           = self.declareParameter( name='LOW_LEVEL_DISC_RMS',               val_def=4,     typ='float' )

        self.res_ascii_out     = self.declareParameter( name='RES_ASCII_OUTPUT',                 val_def=True,  typ='bool' )
        self.res_fit1          = self.declareParameter( name='RES_PERFORM_FIT1',                 val_def=False, typ='bool' )
        self.res_fit2          = self.declareParameter( name='RES_PERFORM_FIT1',                 val_def=False, typ='bool' )  
        self.res_fit_cust      = self.declareParameter( name='RES_PERFORM_FIT_CUSTOM',           val_def=False, typ='bool' ) 
        self.res_png_out       = self.declareParameter( name='RES_PNG_FILES',                    val_def=False, typ='bool' )  
        self.res_save_log      = self.declareParameter( name='RES_SAVE_LOG_FILE',                val_def=False, typ='bool' )  

        # GUILoadResults.py
        self.res_load_mode     = self.declareParameter( name='RES_LOAD_MODE',                    val_def='NONE',typ='str' )
        self.res_fname         = self.declareParameter( name='RES_LOAD_FNAME',                   val_def='NONE',typ='str' )

        # GUISystemSettingsRight.py
        self.thickness_type          = self.declareParameter( name='THICKNESS_TYPE',               val_def='NONORM',typ='str' )
        self.thickness_sample        = self.declareParameter( name='THICKNESS_OF_SAMPLE',          val_def=-1,      typ='float' )
        self.thickness_attlen        = self.declareParameter( name='THICKNESS_ATTENUATION_LENGTH', val_def=-2,      typ='float' )

        # GUICCDCorrectionSettings.py
        self.ccdcorr_blemish         = self.declareParameter( name='CCD_CORRECTION_BLEMISH',       val_def=False,  typ='bool' )
        self.ccdcorr_flatfield       = self.declareParameter( name='CCD_CORRECTION_FLATFIELD',     val_def=False,  typ='bool' )

        # GUICCDSettings.py
        self.ccdset_pixsize          = self.declareParameter( name='CCD_SETTINGS_PIXEL_SIZE',      val_def=0.1,   typ='float' )
        self.ccdset_adcsatu          = self.declareParameter( name='CCD_SETTINGS_ADC_SATTURATION', val_def=12345, typ='int' )
        self.ccdset_aduphot          = self.declareParameter( name='CCD_SETTINGS_ADU_PER_PHOTON',  val_def=123,   typ='float' )
        self.ccdset_ccdeff           = self.declareParameter( name='CCD_SETTINGS_EFFICIENCY',      val_def=0.55,  typ='float' )
        self.ccdset_ccddain          = self.declareParameter( name='CCD_SETTINGS_GAIN',            val_def=0.8,   typ='float' )

#-----------------------------

    def defineStyles( self ) :
        self.styleYellowish = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        self.stylePink      = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        self.styleGray      = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        self.styleGreenish  = "background-color: rgb(100, 255, 200); color: rgb(0, 0, 0);" # Greenish
        self.styleGreenPure = "background-color: rgb(0,   255, 150); color: rgb(0, 0, 0);" # Green
        self.styleBluish    = "background-color: rgb(200, 200, 255); color: rgb(0, 0, 0);" # Bluish
        self.styleWhite     = "background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);"
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
        self.styleButton       = self.styleGray
        self.styleButtonOn     = self.styleBluish
        self.styleButtonClose  = self.stylePink
        self.styleButtonGood   = self.styleGreenPure
        self.styleButtonBad    = self.stylePink
        self.styleBox          = self.styleGray
        self.styleStatusGood   = self.styleGreen
        self.styleStatusWarning= self.styleYellow
        self.styleStatusAlarm  = self.styleRed
        self.styleTitleBold    = self.styleTitle + 'font-size: 18pt; font-family: Courier; font-weight: bold;'
        self.styleWhiteFixed   = self.styleWhite + 'font-family: Fixed;'

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
    confpars.printParsDirectly()
    #confpars.saveParametersInFile()

    sys.exit ( 'End of test for ConfigParametersCorAna' )

#-----------------------------
