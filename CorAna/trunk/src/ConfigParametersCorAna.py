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
#       self.fname_cp           = self.declareParameter( name='FNAME_CONFIG_PARS', val_def='confpars.txt', typ='str' ) 
        self.fname_ped          = self.declareParameter( name='FNAME_PEDESTALS',   val_def='my_ped.txt',   typ='str' ) 
        self.fname_dat          = self.declareParameter( name='FNAME_DATA',        val_def='my_dat.txt',   typ='str' ) 
        self.instr_dir          = self.declareParameter( name='INSTRUMENT_DIR',    val_def='/reg/d/psdm',  typ='str' ) 
        self.instr_name         = self.declareParameter( name='INSTRUMENT_NAME',   val_def='XCS',          typ='str' ) 
        self.exp_name           = self.declareParameter( name='EXPERIMENT_NAME',   val_def='xcsi0112',     typ='str' ) 
        self.str_run_number     = self.declareParameter( name='RUN_NUMBER',        val_def='0015',         typ='str' ) 
        self.str_run_number_dark= self.declareParameter( name='RUN_NUMBER_DARK',   val_def='0014',         typ='str' ) 

        # GUIBeamZeroPars.py
        self.x_coord_beam0      = self.declareParameter( name='X_COORDINATE_BEAM_ZERO',   val_def=1234.5,     typ='float' ) 
        self.y_coord_beam0      = self.declareParameter( name='Y_COORDINATE_BEAM_ZERO',   val_def=1216.5,     typ='float' ) 
        self.x0_pos_in_beam0    = self.declareParameter( name='X0_POS_IN_BEAM_ZERO',      val_def=-59,        typ='int' ) 
        self.z0_pos_in_beam0    = self.declareParameter( name='Z0_POS_IN_BEAM_ZERO',      val_def=175,        typ='int' ) 

        # GUISpecularPars.py
        self.x_coord_specular   = self.declareParameter( name='X_COORDINATE_SPECULAR', val_def=-1,     typ='float' ) 
        self.y_coord_specular   = self.declareParameter( name='Y_COORDINATE_SPECULAR', val_def=-2,     typ='float' ) 
        self.x0_pos_in_specular = self.declareParameter( name='X0_SPEC_IN_SPECULAR',   val_def=-3,     typ='int' ) 
        self.z0_pos_in_specular = self.declareParameter( name='Z0_SPEC_IN_SPECULAR',   val_def=-4,     typ='int' ) 

        # GUIBatchInfoLeft.py
        self.sample_det_dist    = self.declareParameter( name='SAMPLE_TO_DETECTOR_DISTANCE', val_def=4000.1,          typ='float' )
        self.exp_setup_geom     = self.declareParameter( name='EXP_SETUP_GEOMETRY',          val_def='Transmission',  typ='str' )
        self.photon_energy      = self.declareParameter( name='PHOTON_ENERGY',               val_def=7.6543,          typ='float' )
        self.nominal_angle      = self.declareParameter( name='NOMINAL_ANGLE',               val_def=-1,              typ='float' )

        # GUIImgSizePosition.py
        self.col_begin          = self.declareParameter( name='IMG_COL_BEGIN',         val_def=0,              typ='int' )
        self.col_end            = self.declareParameter( name='IMG_COL_END',           val_def=1339,           typ='int' )
        self.row_begin          = self.declareParameter( name='IMG_ROW_BEGIN',         val_def=1,              typ='int' )
        self.row_end            = self.declareParameter( name='IMG_ROW_END',           val_def=1299,           typ='int' )
        self.x_frame_pos        = self.declareParameter( name='X_FRAME_POS',           val_def=-51,            typ='int' )
        self.z_frame_pos        = self.declareParameter( name='Z_FRAME_POS',           val_def=183,            typ='int' )

        # GUIKineticMode.py
        self.kin_mode               = self.declareParameter( name='KINETICS_MODE',        val_def='Non-Kinetics',typ='str' )
        self.kin_win_size           = self.declareParameter( name='KINETICS_WIN_SIZE',    val_def=1,             typ='int' )
        self.kin_top_row            = self.declareParameter( name='KINETICS_TOP_ROW',     val_def=2,             typ='int' )
        self.kin_slice_first        = self.declareParameter( name='KINETICS_SLICE_FIRST', val_def=3,             typ='int' )
        self.kin_slice_last         = self.declareParameter( name='KINETICS_SLICE_LAST',  val_def=4,             typ='int' )

        # GUIBatchPars.py
        self.bat_num           = self.declareParameter( name='BATCH_NUM',         val_def=1,  typ='int' )
        self.bat_num_max       = self.declareParameter( name='BATCH_NUM_MAX',     val_def=9,  typ='int' )
        self.bat_data_start    = self.declareParameter( name='BATCH_DATA_START',  val_def=1,  typ='int' )
        self.bat_data_end      = self.declareParameter( name='BATCH_DATA_END'  ,  val_def=2,  typ='int' )
        self.bat_data_time     = self.declareParameter( name='BATCH_DATA_TIME' ,  val_def=3,  typ='int' )
        self.bat_dark_start    = self.declareParameter( name='BATCH_DARK_START',  val_def=4,  typ='int' )
        self.bat_dark_end      = self.declareParameter( name='BATCH_DARK_END'  ,  val_def=5,  typ='int' )
        self.bat_dark_time     = self.declareParameter( name='BATCH_DARK_TIME' ,  val_def=6,  typ='int' )
        self.bat_flat_start    = self.declareParameter( name='BATCH_FLAT_START',  val_def=7,  typ='int' )
        self.bat_flat_end      = self.declareParameter( name='BATCH_FLAT_END'  ,  val_def=8,  typ='int' )
        self.bat_flat_time     = self.declareParameter( name='BATCH_FLAT_TIME' ,  val_def=9,  typ='int' )
        self.bat_flux          = self.declareParameter( name='BATCH_FLUX',        val_def=6.789e8,  typ='float' )
        self.bat_current       = self.declareParameter( name='BATCH_CURRENT',     val_def=102.205,  typ='float' )

        # GUILoadFiles.py
        self.in_dir_dark       = self.declareParameter( name='IN_DIRECTORY_DARK', val_def='/reg/d/psdm/dark/',typ='str' )
        self.in_dir_flat       = self.declareParameter( name='IN_DIRECTORY_FLAT', val_def='/reg/d/psdm/flat/',typ='str' )
        self.in_dir_blam       = self.declareParameter( name='IN_DIRECTORY_BLAM', val_def='/reg/d/psdm/blam/',typ='str' )
        self.in_dir_data       = self.declareParameter( name='IN_DIRECTORY_DATA', val_def='/reg/d/psdm/data/',typ='str' )
        self.in_file_dark      = self.declareParameter( name='IN_FILE_NAME_DARK', val_def='dark.xtc',typ='str' )
        self.in_file_flat      = self.declareParameter( name='IN_FILE_NAME_FLAT', val_def='flat.xtc',typ='str' )
        self.in_file_blam      = self.declareParameter( name='IN_FILE_NAME_BLAM', val_def='blam.xtc',typ='str' )
        self.in_file_data      = self.declareParameter( name='IN_FILE_NAME_DATA', val_def='data.xtc',typ='str' )
        self.dir_work          = self.declareParameter( name='DIRECTORY_WORK',    val_def='./work',typ='str' )
        self.fname_prefix      = self.declareParameter( name='FILE_NAME_PREFIX',  val_def='my-favor-exp-',typ='str' )

        # GUIAnaSettingsLeft.py
        self.ana_type          = self.declareParameter( name='ANA_TYPE',                  val_def='static',typ='str' )

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

#-----------------------------

    def defineStyles( self ) :
        self.styleYellow = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        self.stylePink   = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        self.styleGray   = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        self.styleGreen  = "background-color: rgb(100, 255, 200); color: rgb(0, 0, 0);" # Greenish
        self.styleBluish = "background-color: rgb(200, 200, 255); color: rgb(0, 0, 0);" # Bluish
        self.styleWhite  = "background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);"
        #self.styleTitle  = "background-color: rgb(239, 235, 231, 255); color: rgb(100, 160, 100);" # Gray bkgd
        #self.styleTitle  = "color: rgb(150, 160, 100);"
        self.styleBlue   = "color: rgb(000, 000, 255);"
        self.styleBuriy  = "color: rgb(150, 100, 50);"

        self.styleBkgd        = self.styleYellow
        self.styleTitle       = self.styleBuriy
        self.styleLabel       = self.styleBlue
        self.styleEdit        = self.styleWhite
        self.styleEditInfo    = self.styleGreen
        self.styleButton      = self.styleGray
        self.styleButtonOn    = self.styleBluish
        self.styleButtonClose = self.stylePink
        self.styleBox         = self.styleGray


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
