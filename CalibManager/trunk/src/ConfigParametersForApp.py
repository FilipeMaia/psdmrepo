#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigParametersForApp
#
#------------------------------------------------------------------------

"""ConfigParametersForApp - class supporting configuration parameters for specific application.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule
 
@version $Id:$

@author Mikhail S. Dubrovin
"""

#----------------------
#  Import of modules --
#----------------------

import sys
from Logger import logger
from ConfigParameters import *
from PyQt4 import QtGui, QtCore
import AppDataPath as apputils # for icons

#-----------------------------

class ConfigParametersForApp ( ConfigParameters ) :
    """Is intended as a storage for configuration parameters for CorAna project.
    #@see BaseClass ConfigParameters
    #@see OtherClass Parameters
    """
    name = 'ConfigParametersForApp'

    list_pars = []

    list_of_queues = ['psnehq', 'psfehq', 'psanacsq']
    list_of_instr  = ['AMO', 'SXR', 'XPP', 'XCS', 'CXI', 'MEC']
    list_of_show_runs = ['all', 'dark']
    list_of_show_dets = ['all', 'selected']

    def __init__ ( self, fname=None ) :
        """Constructor.
        @param fname  the file name with configuration parameters, if not specified then it will be set to the default value at declaration.
        """
        ConfigParameters.__init__(self)
        self.fname_cp = 'confpars-calibman.txt' # Re-define default config file name
        
        self.declareAllParameters()
        self.readParametersFromFile (fname)
        self.initRunTimeParameters()
        self.defineStyles()
  
    def initRunTimeParameters( self ) :
        self.iconsAreLoaded  = False
        #self.char_expand = u' \u25BE' # down-head triangle
        self.guilogger       = None
        self.guimain         = None
        self.guidark         = None
        self.guidarklist     = None
        self.guitabs         = None
        self.guistatus       = None
        self.guiinsexpdirdet = None
        self.guifilebrowser  = None 
        self.blsp            = None 
        self.guidarkcontrolbar = None 
  
#-----------------------------

    def setIcons(self) :

        if self.iconsAreLoaded : return

        self.iconsAreLoaded = True

        #path = './icons/'
        #path = "%s/icons/" % os.path.dirname(sys.argv[0])
        #print 'path to icons:', pat
        #logger.info('Load icons from directory: '+path, self.name)    

        path_icon_contents       = apputils.AppDataPath('CalibManager/icons/contents.png'     ).path()
        path_icon_mail_forward   = apputils.AppDataPath('CalibManager/icons/mail-forward.png' ).path()
        path_icon_button_ok      = apputils.AppDataPath('CalibManager/icons/button_ok.png'    ).path()
        path_icon_button_cancel  = apputils.AppDataPath('CalibManager/icons/button_cancel.png').path()
        path_icon_exit           = apputils.AppDataPath('CalibManager/icons/exit.png'         ).path()
        path_icon_home           = apputils.AppDataPath('CalibManager/icons/home.png'         ).path()
        path_icon_redo           = apputils.AppDataPath('CalibManager/icons/redo.png'         ).path()
        path_icon_undo           = apputils.AppDataPath('CalibManager/icons/undo.png'         ).path()
        path_icon_reload         = apputils.AppDataPath('CalibManager/icons/reload.png'       ).path()
        path_icon_save           = apputils.AppDataPath('CalibManager/icons/save.png'         ).path()
        path_icon_save_cfg       = apputils.AppDataPath('CalibManager/icons/fileexport.png'   ).path()
        path_icon_edit           = apputils.AppDataPath('CalibManager/icons/edit.png'         ).path()
        path_icon_browser        = apputils.AppDataPath('CalibManager/icons/fileopen.png'     ).path()
        path_icon_monitor        = apputils.AppDataPath('CalibManager/icons/icon-monitor.png' ).path()
        path_icon_unknown        = apputils.AppDataPath('CalibManager/icons/icon-unknown.png' ).path()
        path_icon_logviewer      = apputils.AppDataPath('CalibManager/icons/logviewer.png'    ).path()
        path_icon_lock           = apputils.AppDataPath('CalibManager/icons/locked-icon.png'  ).path()
        path_icon_unlock         = apputils.AppDataPath('CalibManager/icons/unlocked-icon.png').path()

        path_icon_table          = apputils.AppDataPath('CalibManager/icons/table.gif'        ).path()
        path_icon_folder_open    = apputils.AppDataPath('CalibManager/icons/folder_open.gif'  ).path()
        path_icon_folder_closed  = apputils.AppDataPath('CalibManager/icons/folder_closed.gif').path()


        self.icon_contents      = QtGui.QIcon(path_icon_contents     )
        self.icon_mail_forward  = QtGui.QIcon(path_icon_mail_forward )
        self.icon_button_ok     = QtGui.QIcon(path_icon_button_ok    )
        self.icon_button_cancel = QtGui.QIcon(path_icon_button_cancel)
        self.icon_exit          = QtGui.QIcon(path_icon_exit         )
        self.icon_home          = QtGui.QIcon(path_icon_home         )
        self.icon_redo          = QtGui.QIcon(path_icon_redo         )
        self.icon_undo          = QtGui.QIcon(path_icon_undo         )
        self.icon_reload        = QtGui.QIcon(path_icon_reload       )
        self.icon_save          = QtGui.QIcon(path_icon_save         )
        self.icon_save_cfg      = QtGui.QIcon(path_icon_save_cfg     )
        self.icon_edit          = QtGui.QIcon(path_icon_edit         )
        self.icon_browser       = QtGui.QIcon(path_icon_browser      )
        self.icon_monitor       = QtGui.QIcon(path_icon_monitor      )
        self.icon_unknown       = QtGui.QIcon(path_icon_unknown      )
        self.icon_logviewer     = QtGui.QIcon(path_icon_logviewer    )
        self.icon_lock          = QtGui.QIcon(path_icon_lock         )
        self.icon_unlock        = QtGui.QIcon(path_icon_unlock       )

        self.icon_table         = QtGui.QIcon(path_icon_table        )
        self.icon_folder_open   = QtGui.QIcon(path_icon_folder_open  )
        self.icon_folder_closed = QtGui.QIcon(path_icon_folder_closed)

        #self.icon_contents      = QtGui.QIcon(path + 'contents.png'      )
        #self.icon_mail_forward  = QtGui.QIcon(path + 'mail-forward.png'  )
        #self.icon_button_ok     = QtGui.QIcon(path + 'button_ok.png'     )
        #self.icon_button_cancel = QtGui.QIcon(path + 'button_cancel.png' )
        #self.icon_exit          = QtGui.QIcon(path + 'exit.png'          )
        #self.icon_home          = QtGui.QIcon(path + 'home.png'          )
        #self.icon_redo          = QtGui.QIcon(path + 'redo.png'          )
        #self.icon_undo          = QtGui.QIcon(path + 'undo.png'          )
        #self.icon_reload        = QtGui.QIcon(path + 'reload.png'        )
        #self.icon_save          = QtGui.QIcon(path + 'save.png'          )
        #self.icon_save_cfg      = QtGui.QIcon(path + 'fileexport.png'    )
        #self.icon_edit          = QtGui.QIcon(path + 'edit.png'          )
        #self.icon_browser       = QtGui.QIcon(path + 'fileopen.png'      )
        #self.icon_monitor       = QtGui.QIcon(path + 'icon-monitor.png'  )
        #self.icon_unknown       = QtGui.QIcon(path + 'icon-unknown.png'  )
        #self.icon_logviewer     = QtGui.QIcon(path + 'logviewer.png'     )
        #self.icon_lock          = QtGui.QIcon(path + 'locked-icon.png'   )
        #self.icon_unlock        = QtGui.QIcon(path + 'unlocked-icon.png' )


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
        
    def declareAllParameters( self ) :
        # Possible typs for declaration : 'str', 'int', 'long', 'float', 'bool' 

        # GUILogger.py
        self.log_level        = self.declareParameter( name='LOG_LEVEL_OF_MSGS',  val_def='info',         type='str' )
        self.log_file         = self.declareParameter( name='LOG_FILE_FOR_LEVEL', val_def='./log_for_level.txt',       type='str' )
        #self.log_file_total  = self.declareParameter( name='LOG_FILE_TOTAL',     val_def='./log_total.txt',           type='str' )
        self.save_log_at_exit = self.declareParameter( name='SAVE_LOG_AT_EXIT',   val_def=True,           type='bool')

        # GUIInsExpDirDet.py
        self.instr_dir          = self.declareParameter( name='INSTRUMENT_DIR',    val_def='/reg/d/psdm',  type='str' ) 
        self.instr_name         = self.declareParameter( name='INSTRUMENT_NAME',   val_def='Select',       type='str' ) # 'CXI'
        self.exp_name           = self.declareParameter( name='EXPERIMENT_NAME',   val_def='Select',       type='str' ) # 'cxitut13'
        self.det_name           = self.declareParameter( name='DETECTOR_NAME',     val_def='Select',       type='str' ) # 'CSPAD'
        self.calib_dir          = self.declareParameter( name='CALIB_DIRECTORY',   val_def='Select',       type='str' ) # '/reg/d/psdm/CXI/cxitut13/calib'

        # GUITabs.py
        self.current_tab    = self.declareParameter( name='CURRENT_TAB'      , val_def='Status',        type='str' )

        # GUIConfig.py
        self.current_config_tab = self.declareParameter( name='CURRENT_CONFIG_TAB', val_def='Config File', type='str' )
 

        # GUIMainSplit.py
        ####self.fname_cp       = self.declareParameter( name='FNAME_CONFIG_PARS', val=fname, val_def='confpars.txt', type='str' )

        # GUIConfigPars.py
        self.dir_work          = self.declareParameter( name='DIRECTORY_WORK',        val_def='./work',       type='str' )
        self.dir_results       = self.declareParameter( name='DIRECTORY_RESULTS',     val_def='./results',    type='str' )
        self.fname_prefix      = self.declareParameter( name='FILE_NAME_PREFIX',      val_def='clb-',         type='str' )
        self.save_cp_at_exit   = self.declareParameter( name='SAVE_CONFIG_AT_EXIT',   val_def=True,           type='bool')

        # GUIDark.py
        self.dark_more_opts    = self.declareParameter( name='DARK_MORE_OPTIONS',     val_def=True,          type='bool')

        # GUIDarkRunGo.py
        self.str_run_number    = self.declareParameter( name='STRING_RUN_NUMBER',     val_def='None',         type='str' )
        self.str_run_from      = self.declareParameter( name='STRING_RUN_FROM',       val_def='0000',         type='str' )
        self.str_run_to        = self.declareParameter( name='STRING_RUN_TO',         val_def='end',          type='str' )

        # GUIDarkControlBar.py
        self.dark_list_show_runs  = self.declareParameter( name='DARK_LIST_SHOW_RUNS', val_def=self.list_of_show_runs[0], type='str' )
        self.dark_list_show_dets  = self.declareParameter( name='DARK_LIST_SHOW_DETS', val_def=self.list_of_show_dets[0], type='str' )

        # GUIGrabSubmitELog.py
        #self.cbx_more_options    = self.declareParameter( name='CBX_SHOW_MORE_OPTIONS',   val_def=False,             type='bool' )
        #self.img_infname         = self.declareParameter( name='IMG_INPUT_FNAME',  val_def='./img-1.ppm',            type='str' )
        #self.img_oufname         = self.declareParameter( name='IMG_OUTPUT_FNAME', val_def='./img-1.ppm',            type='str' )

        #self.elog_post_des       = self.declareParameter( name='ELOG_POST_DESCRIPTION',   val_def='Image',           type='str' )
        #self.elog_post_tag       = self.declareParameter( name='ELOG_POST_TAG',           val_def='SCREENSHOT',      type='str' )
        #self.elog_post_ins       = self.declareParameter( name='ELOG_POST_INSTRUMENT',    val_def='AMO',             type='str' )
        #self.elog_post_exp       = self.declareParameter( name='ELOG_POST_EXPERIMENT',    val_def='amodaq09',        type='str' )
        #self.elog_post_in2       = self.declareParameter( name='ELOG_POST_INSTRUMENT_2',  val_def='NEH',             type='str' )
        #self.elog_post_ex2       = self.declareParameter( name='ELOG_POST_EXPERIMENT_2',  val_def='CXI Instrument',  type='str' )
        #self.elog_post_run       = self.declareParameter( name='ELOG_POST_RUN',           val_def='',                type='str' )
        #self.elog_post_res       = self.declareParameter( name='ELOG_POST_RESPONCE',      val_def='',                type='str' )
        #self.elog_post_msg       = self.declareParameter( name='ELOG_POST_MESSAGE',       val_def='',                type='str' )
        #self.elog_post_usr       = self.declareParameter( name='ELOG_POST_USER_NAME',     val_def='Unknown',         type='str' )
        #self.elog_post_sta       = self.declareParameter( name='ELOG_POST_STATION',       val_def='',                type='str' )
        #self.elog_post_url       = self.declareParameter( name='ELOG_POST_URL',           val_def='',                type='str' )
        #self.elog_post_cmd       = self.declareParameter( name='ELOG_POST_CHILD_COMMAND', val_def='',                type='str' )

        # GUIDark.py
        #self.use_dark_xtc_all  = self.declareParameter( name='USE_DARK_XTC_ALL_CHUNKS', val_def=True,  type='bool' )

        #self.in_dir_dark       = self.declareParameter( name='IN_DIRECTORY_DARK', val_def='/reg/d/ana12/xcs/xcsi0112/xtc',type='str' )
        #self.in_file_dark      = self.declareParameter( name='IN_FILE_NAME_DARK', val_def='e167-r0020-s00-c00.xtc',type='str' )

        #self.bat_dark_total    = self.declareParameter( name='BATCH_DARK_TOTAL',      val_def=-1,       type='int' )
        self.bat_dark_start    = self.declareParameter( name='BATCH_DARK_START',      val_def= 1,       type='int' )
        self.bat_dark_end      = self.declareParameter( name='BATCH_DARK_END'  ,      val_def=1000,     type='int' )
        self.bat_det_info      = self.declareParameter( name='BATCH_DET_INFO',        val_def='DetInfo(:Princeton)',  type='str' )
        self.bat_img_rec_mod   = self.declareParameter( name='BATCH_IMG_REC_MODULE',  val_def='ImgAlgos.PrincetonImageProducer',  type='str' )
        self.mask_hot_thr            = self.declareParameter( name='MASK_HOT_PIX_ADU_THR_ON_RMS',  val_def=10.0,  type='float' )
        self.mask_hot_is_used        = self.declareParameter( name='MASK_HOT_PIX_IS_USED',         val_def=True,  type='bool' )


        # For batch jobs
        self.bat_queue               = self.declareParameter( name='BATCH_QUEUE',                val_def='psnehq', type='str' )
        self.bat_submit_interval_sec = self.declareParameter( name='BATCH_SUBMIT_INTERVAL_SEC',  val_def=100,      type='int' )

        # GUIMaskEditor.py
        self.path_mask_img      = self.declareParameter( name='PATH_TO_MASK_IMAGE',        val_def='./work/*.txt',       type='str' )

#-----------------------------
    
        self.list_of_dets   = ['CSPAD', 'CSPAD2x2', 'Princeton', 'pnCCD', 'Tm6740', 'Opal2000', 'Opal4000', 'Acqiris']
        self.list_of_types  = ['Psana::CsPad::Data',
                               'Psana::CsPad2x2::Element',
                               'Psana::Princeton::Frame',
                               'Psana::PNCCD::FullFrame',
                               'Psana::Camera::Frame',
                               'Psana::Camera::Frame',
                               'Psana::Camera::Frame',
                               'Psana::Acqiris::DataDesc']
        self.dict_of_det_types = dict( zip(self.list_of_dets, self.list_of_types) )
        
#-----------------------------

        det_cbx_states = [ (False, False ,'bool'), \
                           (False, False ,'bool'), \
                           (False, False ,'bool'), \
                           (False, False ,'bool'), \
                           (False, False ,'bool'), \
                           (False, False ,'bool'), \
                           (False, False ,'bool'), \
                           (False, False ,'bool') ]
        self.det_cbx_states_list = self.declareListOfPars( 'DETECTOR_CBX_STATE', det_cbx_states )

#-----------------------------

        self.list_of_det_pars = zip(self.list_of_dets, self.list_of_types, self.det_cbx_states_list)

#-----------------------------

    def list_of_dets_selected( self ) :
        #lds = []
        #for det in self.det_name.value().split(' ') : lds.append(det)
        #for det,state in zip(self.list_of_dets,self.det_cbx_states_list) :
        #    if state.value() : lds.append(det)
        #return lds
        #return [det for det in self.det_name.value().split(' ')]
        return [det for det,state in zip(self.list_of_dets,self.det_cbx_states_list) if state.value()]


#-----------------------------

    def defineStyles( self ) :
        self.styleYellowish = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        self.stylePink      = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        self.styleYellowBkg = "background-color: rgb(255, 255, 120); color: rgb(0, 0, 0);" # YellowBkg
        self.styleGreenMy   = "background-color: rgb(150, 250, 230); color: rgb(0, 0, 0);" # My
        self.styleGray      = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        self.styleGreenish  = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        self.styleGreenPure = "background-color: rgb(150, 255, 150); color: rgb(0, 0, 0);" # Green
        self.styleBluish    = "background-color: rgb(220, 220, 250); color: rgb(0, 0, 0);" # Bluish
        self.styleWhite     = "background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);"
        self.styleRedBkgd   = "background-color: rgb(255,   0,   0); color: rgb(0, 0, 0);" # Red background
        self.styleReddish   = "background-color: rgb(220,   0,   0); color: rgb(0, 0, 0);" # Reddish background
        self.styleTransp    = "background-color: rgb(255,   0,   0, 100);"
        #self.styleDefault   = "background-color: rgb(239, 235, 231, 255); color: rgb(0, 0, 0);" # Gray bkgd
        self.styleDefault   = ""
        #self.styleTitle  = "color: rgb(150, 160, 100);"
        self.styleBlue   = "color: rgb(100, 0, 150);"
        self.styleBuriy  = "color: rgb(150, 100, 50);"
        self.styleRed    = "color: rgb(255, 0, 0);"
        self.styleGreen  = "color: rgb(0, 150, 0);"
        self.styleYellow = "color: rgb(0, 150, 150);"

        #self.styleBkgd         = self.styleGreenMy # styleYellowish
        self.styleBkgd         = self.styleDefault
        self.styleTitle        = self.styleBuriy
        self.styleLabel        = self.styleBlue
        self.styleEdit         = self.styleWhite
        self.styleEditInfo     = self.styleBkgd # self.styleGreenish
        #self.styleEditInfo     = self.styleGreenish # Bluish
        self.styleEditBad      = self.styleRedBkgd
        self.styleButton       = self.styleGray
        self.styleButtonOn     = self.styleBluish
        self.styleButtonClose  = self.stylePink
        self.styleButtonWarning= self.styleYellowBkg
        self.styleButtonGood   = self.styleGreenPure
        #self.styleButtonBad    = self.stylePink
        self.styleButtonBad    = self.styleReddish
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
        self.colorTabItem      = QtGui.QColor('white')

        self.styleTitleInFrame = self.styleWhite # self.styleDefault # self.styleWhite # self.styleGray

    def printParsDirectly( self ) :
        logger.info('Direct use of parameter:' + self.fname_ped.name() + ' ' + self.fname_ped.value(), self.name )     
        logger.info('Direct use of parameter:' + self.fname_dat.name() + ' ' + self.fname_dat.value(), self.name )    

    def close( self ) :

        if self.save_cp_at_exit.value() :
            fname = self.fname_cp
            logger.info('save configuration parameters in file: %s' % fname, __name__)
            self.saveParametersInFile( fname )

#-----------------------------

confpars = ConfigParametersForApp ()
cp = confpars

#-----------------------------

def test_ConfigParametersForApp() :
    confpars.printParameters()
    #confpars.printParsDirectly()
    confpars.saveParametersInFile()

#-----------------------------

if __name__ == "__main__" :

    test_ConfigParametersForApp()
    sys.exit (0)

#-----------------------------
