#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigFileGenerator...
#
#------------------------------------------------------------------------

"""Generates the configuration files for psana from current configuration parameters

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

from ConfigParametersCorAna import confpars as cp
from Logger                 import logger
from FileNameManager        import fnm
import GlobalUtils          as     gu


#import AppUtils.AppDataPath as apputils
import           AppDataPath as apputils # My version, added in path the '../../data:'

#-----------------------------

class ConfigFileGenerator :
    """Generates the configuration files for psana from current configuration parameters
    """

    def __init__ (self, do_test=False) :
        """
        @param path_in  path to the input psana configuration stub-file
        @param path_out path to the output psana configuration file with performed substitutions
        @param d_subs   dictionary of substitutions
        @param keys     keys from the dictionary       
        """
        self.path_in  = None 
        self.path_out = None 
        self.d_subs   = None
        self.keys     = None 
        self.do_test_print = do_test
  
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def make_psana_cfg_file_for_peds_scan (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-peds-scan.cfg').path()
        self.path_out = fnm.path_peds_scan_psana_cfg()
        self.d_subs   = {'FNAME_XTC'            : fnm.path_dark_xtc_cond(),
                         'SKIP'                 : 'IS_NOT_USED',
                         'EVENTS'               : 'FOR_ALL_EVENTS',
                         'FNAME_TIMESTAMP_LIST' : fnm.path_peds_scan_tstamp_list()
                         }

        #self.print_substitution_dict()
        #self.make_cfg_file()
        txt_cfg = self.text_for_section()
        self.save_cfg_file(txt_cfg, self.path_out)

#-----------------------------

    def make_psana_cfg_file_for_peds_aver_v1 (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-peds-aver.cfg').path()
        self.path_out = fnm.path_peds_aver_psana_cfg()
        self.d_subs   = {'FNAME_XTC'      : fnm.path_dark_xtc_cond(),
                         'SKIP'           : str( cp.bat_dark_start.value() - 1 ),
                         'EVENTS'         : str( cp.bat_dark_end.value() - cp.bat_dark_start.value() + 1 ),
                         'IMG_REC_MODULE' : str( cp.bat_img_rec_mod.value() ),
                         'DETINFO'        : str( cp.bat_det_info.value() ),
                         'FNAME_PEDS_AVE' : fnm.path_pedestals_ave(),
                         'FNAME_PEDS_RMS' : fnm.path_pedestals_rms()
                         }

        self.d_subs['FNAME_HOTPIX_MASK'   ] = fnm.path_hotpix_mask()
        self.d_subs['HOTPIX_THRESHOLD_ADU'] = str( cp.mask_hot_thr.value() )

        #if cp.mask_hot_is_used.value() : 
        #    self.d_subs['FNAME_HOTPIX_MASK'   ] = fnm.path_hotpix_mask()
        #    self.d_subs['HOTPIX_THRESHOLD_ADU'] = str( cp.mask_hot_thr.value() )
        #else :
        #    self.d_subs['FNAME_HOTPIX_MASK'   ] = ''
        #    self.d_subs['HOTPIX_THRESHOLD_ADU'] = '10000'

        #self.print_substitution_dict()
        #self.make_cfg_file()
        txt_cfg = self.text_for_section()
        self.save_cfg_file(txt_cfg, self.path_out)

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def make_psana_cfg_file_for_data_scan (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-data-scan.cfg').path()
        self.path_out = fnm.path_data_scan_psana_cfg()
        self.d_subs   = {'FNAME_XTC'                         : fnm.path_data_xtc_cond(),
                         'SKIP'                              : 'IS_NOT_USED',
                         'EVENTS'                            : 'FOR_ALL_EVENTS',
                         'FNAME_TIMESTAMP_LIST'              : fnm.path_data_scan_tstamp_list(),
                         'FNAME_INTENSITY_MONITORS_DATA'     : fnm.path_data_scan_monitors_data(),
                         'FNAME_INTENSITY_MONITORS_COMMENTS' : fnm.path_data_scan_monitors_commments()
                         }

        #self.print_substitution_dict()
        #self.make_cfg_file()
        txt_cfg = self.text_for_section()
        self.save_cfg_file(txt_cfg, self.path_out)

#-----------------------------

    def make_psana_cfg_file_for_data_aver_v1 (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-data-aver.cfg').path()
        self.path_out = fnm.path_data_aver_psana_cfg()
        self.d_subs   = {'FNAME_XTC'      : fnm.path_data_xtc_cond(),
                         'SKIP'           : str( cp.bat_data_start.value() - 1 ),
                         'EVENTS'         : str( cp.bat_data_end.value() - cp.bat_data_start.value() + 1 ),
                         'IMG_REC_MODULE' : str( cp.bat_img_rec_mod.value() ),
                         'DETINFO'        : str( cp.bat_det_info.value() ),
                         'FNAME_DATA_AVE' : fnm.path_data_raw_ave(),
                         'FNAME_DATA_RMS' : fnm.path_data_raw_rms(),
                         'SAT_THR_ADU'    : str( cp.ccdset_adcsatu.value() ),
                         'SATPIX_MASK'    : fnm.path_satpix_mask(),
                         'SATPIX_FRAC'    : fnm.path_satpix_frac(),
                         'HOTPIX_MASK'    : '', # fnm.path_hotpix_mask(),
                         'HOTPIX_FRAC'    : '' # fnm.path_hotpix_frac()
                         }
        # cp.ccdset_ccdgain.value()
        # cp.ccdset_ccdeff .value()

        #self.print_substitution_dict()
        #self.make_cfg_file()
        txt_cfg = self.text_for_section()
        self.save_cfg_file(txt_cfg, self.path_out)

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def make_psana_cfg_file_for_cora_split_v1 (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-cora-split.cfg').path()
        self.path_out = fnm.path_cora_split_psana_cfg()

        self.d_subs   = {'FNAME_XTC'       : fnm.path_data_xtc_cond(),
                         'SKIP'            : str( cp.bat_data_start.value() - 1 ),
                         'EVENTS'          : str( cp.bat_data_end.value() - cp.bat_data_start.value() + 1 ),
                         'IMG_REC_MODULE'  : str( cp.bat_img_rec_mod.value() ),
                         'DETINFO'         : str( cp.bat_det_info.value() ),
                         'PATH_PREFIX_CORA': str( fnm.path_prefix_cora() ),
                         'IMG_NPARTS'      : str( cp.bat_img_nparts.value() ),
                         'FNAME_PEDS_AVE'  : fnm.path_pedestals_ave(),
                         'FNAME_DATA_AVE'  : fnm.path_data_ave(),
                         'FNAME_DATA_RMS'  : fnm.path_data_rms()
                         }

        fname_imon_cfg = fnm.path_cora_split_imon_cfg()
        self.make_imon_cfg_file (fname_imon_cfg)
        self.d_subs['FNAME_IMON_CFG' ] = str( fname_imon_cfg )


        if cp.lld_type.value() == 'ADU' : #  ['NONE', 'ADU', 'RMS']
            self.d_subs['THRESHOLD_ADU' ] = str( cp.lld_adu.value() )
            self.d_subs['DO_CONST_THR'  ] = 'true'
            self.d_subs['THRESHOLD_NRMS'] = '0'
            self.d_subs['FNAME_PEDS_RMS'] = ''

        elif cp.lld_type.value() == 'RMS' : 
            self.d_subs['THRESHOLD_ADU' ] = '0'
            self.d_subs['DO_CONST_THR'  ] = 'false'
            self.d_subs['THRESHOLD_NRMS'] = str( cp.lld_rms.value() )
            self.d_subs['FNAME_PEDS_RMS'] = fnm.path_pedestals_rms()

        else : 
            self.d_subs['THRESHOLD_ADU' ] = '0'
            self.d_subs['DO_CONST_THR'  ] = 'false'
            self.d_subs['THRESHOLD_NRMS'] = '0'
            self.d_subs['FNAME_PEDS_RMS'] = ''


        if os.path.lexists( fnm.path_cora_split_map_static_q() ) :
            self.d_subs['FNAME_MAP_BINS' ] = fnm.path_cora_split_map_static_q()
        else :
            self.d_subs['FNAME_MAP_BINS' ] = ''            
        self.d_subs['FNAME_INT_BINS' ]     = fnm.path_cora_split_int_static_q()
        self.d_subs['NUMBER_OF_BINS' ]     = str( cp.ana_stat_part_q.value() )

        #self.print_substitution_dict()
        #self.make_cfg_file()
        txt_cfg = self.text_for_section()
        self.save_cfg_file(txt_cfg, self.path_out)

#-----------------------------

    def add_cfg_trailer_for_cora_split (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-cora-split-trailer.cfg').path()

        self.d_subs   = {'DETINFO'         : str( cp.bat_det_info.value() ),
                         'PATH_PREFIX_CORA': str( fnm.path_prefix_cora() ),
                         'IMG_NPARTS'      : str( cp.bat_img_nparts.value() ),
                         'FNAME_PEDS_AVE'  : fnm.path_pedestals_ave(),
                         'FNAME_DATA_AVE'  : fnm.path_data_ave(),
                         'FNAME_DATA_RMS'  : fnm.path_data_rms()
                         }

        fname_imon_cfg = fnm.path_cora_split_imon_cfg()
        self.make_imon_cfg_file (fname_imon_cfg)
        self.d_subs['FNAME_IMON_CFG' ] = str( fname_imon_cfg )


        if cp.lld_type.value() == 'ADU' : #  ['NONE', 'ADU', 'RMS']
            self.d_subs['THRESHOLD_ADU' ] = str( cp.lld_adu.value() )
            self.d_subs['DO_CONST_THR'  ] = 'true'
            self.d_subs['THRESHOLD_NRMS'] = '0'
            self.d_subs['FNAME_PEDS_RMS'] = ''

        elif cp.lld_type.value() == 'RMS' : 
            self.d_subs['THRESHOLD_ADU' ] = '0'
            self.d_subs['DO_CONST_THR'  ] = 'false'
            self.d_subs['THRESHOLD_NRMS'] = str( cp.lld_rms.value() )
            self.d_subs['FNAME_PEDS_RMS'] = fnm.path_pedestals_rms()

        else : 
            self.d_subs['THRESHOLD_ADU' ] = '0'
            self.d_subs['DO_CONST_THR'  ] = 'false'
            self.d_subs['THRESHOLD_NRMS'] = '0'
            self.d_subs['FNAME_PEDS_RMS'] = ''


        if os.path.lexists( fnm.path_cora_split_map_static_q() ) :
            self.d_subs['FNAME_MAP_BINS' ] = fnm.path_cora_split_map_static_q()
        else :
            self.d_subs['FNAME_MAP_BINS' ] = ''            
        self.d_subs['FNAME_INT_BINS' ]     = fnm.path_cora_split_int_static_q()
        self.d_subs['NUMBER_OF_BINS' ]     = str( cp.ana_stat_part_q.value() )

        #self.print_substitution_dict()

        self.str_of_modules += ' ImgAlgos.ImgCalib ImgAlgos.ImgIntMonCorr ImgAlgos.ImgIntForBins ImgAlgos.ImgVsTimeSplitInFiles ImgAlgos.ImgAverage'

        self.txt_cfg_body += self.text_for_section()

#-----------------------------

    def make_psana_cfg_file_for_cora_split (self) :

        self.str_of_modules = ''
        self.txt_cfg_header = '# Autogenerated config file for corana split\n'
        self.txt_cfg_body   = '\n\n'

        self.add_cfg_module_tahometer()
        self.add_cfg_module_img_producer()
        self.add_cfg_trailer_for_cora_split()
        
        self.cfg_file_header_for_data_aver()
        self.save_cfg_file(self.txt_cfg_header + self.txt_cfg_body, fnm.path_cora_split_psana_cfg())

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def make_psana_cfg_file_for_peds_aver (self) :

        self.str_of_modules = ''
        self.txt_cfg_header = '# Autogenerated config file for dark average\n' \
                            + '# Useful command:\n' \
                            + '# psana -m EventKeys -n 5 ' + fnm.path_data_xtc_cond() \
                            + '\n'
        self.txt_cfg_body   = '\n\n'

        self.add_cfg_module_tahometer()
        self.add_cfg_module_img_producer()
        self.add_cfg_module_ndarraverage_for_peds_aver() 
        self.cfg_file_header_for_peds_aver()

        self.save_cfg_file(self.txt_cfg_header + self.txt_cfg_body, fnm.path_peds_aver_psana_cfg())

#-----------------------------

    def make_psana_cfg_file_for_data_aver (self) :

        self.str_of_modules = ''
        self.txt_cfg_header = '# Autogenerated config file for data average\n'
        self.txt_cfg_body   = '\n\n'

        self.add_cfg_module_tahometer()
        self.add_cfg_module_img_producer()
        self.add_cfg_module_ndarraverage_for_data_aver() 
        self.add_cfg_module_img_mask_evaluation()
        self.cfg_file_header_for_data_aver()

        self.save_cfg_file(self.txt_cfg_header + self.txt_cfg_body, fnm.path_data_aver_psana_cfg())

#-----------------------------

    def add_cfg_module_img_producer(self) :

        det_name = cp.detector.value()

        self.source     = str( cp.bat_det_info.value() )
        #self.fname_ave  = fnm.path_pedestals_ave()
        #self.fname_rms  = fnm.path_pedestals_rms()
        #self.fname_mask = fnm.path_hotpix_mask()

        if self.do_test_print : print '\nDetector selected: %10s' % (det_name)
        #print 'Input params:', self.source, self.fname_ave, self.fname_rms, self.fname_mask

        # list_of_dets   = ['CSPAD', 'CSPAD2x2', 'Princeton', 'pnCCD', 'Tm6740', 'Opal2000', 'Opal4000'] 
        if   det_name == cp.list_of_dets[0] : self.add_cfg_module_cspad_2darr_producer('CSPadPixCoords.CSPadNDArrProducer')
        elif det_name == cp.list_of_dets[1] : self.add_cfg_module_cspad_2darr_producer('CSPadPixCoords.CSPad2x2NDArrProducer')
        elif det_name == cp.list_of_dets[2] : self.add_cfg_module_princeton_img_producer()
        elif det_name == cp.list_of_dets[3] : self.add_cfg_module_pnccd_img_producer()
        elif det_name == cp.list_of_dets[4] : self.add_cfg_module_camera_img_producer()
        elif det_name == cp.list_of_dets[5] : self.add_cfg_module_camera_img_producer()
        elif det_name == cp.list_of_dets[6] : self.add_cfg_module_camera_img_producer()
        #elif det_name == cp.list_of_dets[8] : self.print_warning()
        
        else : logger.warning('UNKNOWN DETECTOR: %s' % det_name, __name__)

#-----------------------------

    def cfg_file_header_for_peds_aver (self) :
        self.cfg_file_header(fnm.path_dark_xtc_cond(), \
                             cp.bat_dark_start.value() - 1, \
                             cp.bat_dark_end.value() - cp.bat_dark_start.value() + 1 )

#-----------------------------

    def cfg_file_header_for_data_aver (self) :
        self.cfg_file_header( fnm.path_data_xtc_cond(), \
                              cp.bat_data_start.value() - 1, \
                              cp.bat_data_end.value() - cp.bat_data_start.value() + 1 )

#-----------------------------

    def cfg_file_header (self, xtc_files, num_skip=0, num_events=1E9) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-header.cfg').path()
        self.d_subs   = {'FNAME_XTC' : xtc_files,
                         'SKIP'      : str( num_skip ),
                         'EVENTS'    : str( num_events ),
                         'MODULES'   : self.str_of_modules
                         }

        self.txt_cfg_header += self.text_for_section ()

#-----------------------------

    def add_cfg_module_tahometer (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-module-tahometer.cfg').path()
        mod = 'ImgAlgos.Tahometer'
        self.d_subs   = {
                         'MODULE'          : mod,
                         'PRINT_BITS'      : '7',
                         'EVENTS_INTERVAL' : '100'
                        }

        self.add_module_in_cfg (mod)

#-----------------------------

    def add_cfg_module_ndarraverage_for_peds_aver (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-module-ndarraverage.cfg').path()
        mod = 'ImgAlgos.NDArrAverage'
        self.d_subs   = {
                         'MODULE'               : mod,
                         'DETINFO'              : str( cp.bat_det_info.value() ),
                         'IMAGE'                : 'img',
                         'FNAME_AVE'            : fnm.path_pedestals_ave(),
                         'FNAME_RMS'            : fnm.path_pedestals_rms(),
                         'FNAME_MASK'           : fnm.path_hotpix_mask(),
                         'THR_RMS_ADU'          : str( cp.mask_hot_thr.value() ),
                         'PRINT_BITS'           : '57'
                         }

        self.add_module_in_cfg (mod)

#-----------------------------

    def add_cfg_module_ndarraverage_for_data_aver (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-module-ndarraverage.cfg').path()
        mod = 'ImgAlgos.NDArrAverage'
        self.d_subs   = {
                         'MODULE'               : mod, # str( cp.bat_img_rec_mod.value() )
                         'DETINFO'              : str( cp.bat_det_info.value() ),
                         'IMAGE'                : 'img',
                         'FNAME_AVE'            : fnm.path_data_raw_ave(),
                         'FNAME_RMS'            : fnm.path_data_raw_rms(),
                         'FNAME_MASK'           : '',
                         'THR_RMS_ADU'          : '0',
                         'PRINT_BITS'           : '57'
                         }

        self.add_module_in_cfg (mod)

#-----------------------------

    def add_cfg_module_img_mask_evaluation (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-module-img-mask-evaluation.cfg').path()
        mod = 'ImgAlgos.ImgMaskEvaluation'
        self.d_subs   = {
                         'MODULE'         : mod, # str( cp.bat_img_rec_mod.value() )
                         'DETINFO'        : str( cp.bat_det_info.value() ),
                         'KEY_IN'         : 'img',
                         'SAT_THR_ADU'    : str( cp.ccdset_adcsatu.value() ),
                         'SATPIX_MASK'    : fnm.path_satpix_mask(),
                         'SATPIX_FRAC'    : fnm.path_satpix_frac(),
                         'HOTPIX_MASK'    : '', # fnm.path_hotpix_mask(),
                         'HOTPIX_FRAC'    : '' # fnm.path_hotpix_frac()
                         }

        self.add_module_in_cfg (mod)

#-----------------------------

    def add_cfg_module_cspad_2darr_producer(self, mod) :

        self.d_subs   = {
                         'MODULE'         : mod,
                         'DETINFO'        : str( cp.bat_det_info.value() ),
                         'KEY_IN'         : '',
                         'KEY_TRANSMIT'   : 'calibrated',
                         'KEY_OUT'        : 'img',
                         'PRINT_BITS'     : '1'
                         }

        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-module-cspad-calib.cfg').path()
        self.add_module_in_cfg ('cspad_mod.CsPadCalib')

        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-module-cspad-2darr-producer.cfg').path()
        self.add_module_in_cfg (mod)

#-----------------------------

    def add_cfg_module_princeton_img_producer(self) :
        self.add_cfg_module_img_producer_universal('ImgAlgos.PrincetonImageProducer', 'CorAna/scripts/psana-module-princeton-img-producer.cfg')

#-----------------------------

    def add_cfg_module_camera_img_producer(self) :
        self.add_cfg_module_img_producer_universal('ImgAlgos.CameraImageProducer', 'CorAna/scripts/psana-module-camera-img-producer.cfg')

#-----------------------------

    def add_cfg_module_pnccd_img_producer(self) :
        self.add_cfg_module_img_producer_universal('ImgAlgos.PnccdImageProducer', 'CorAna/scripts/psana-module-pnccd-img-producer.cfg')

#-----------------------------

    def add_cfg_module_img_producer_universal(self, mod, script) :
        self.path_in  = apputils.AppDataPath(script).path()
        self.d_subs   = {
                         'MODULE'         : mod,
                         'DETINFO'        : str( cp.bat_det_info.value() ),
                         'KEY_IN'         : '',
                         'KEY_OUT'        : 'img',
                         'PRINT_BITS'     : '1'
                         }
        self.add_module_in_cfg (mod)

#-----------------------------
#-----------------------------

    def get_text_table_of_imon_pars (self) :
        text = ''
        for i, (name, ch1, ch2, ch3, ch4, norm, sele, sele_min, sele_max, norm_ave, short_name) in enumerate(cp.imon_pars_list) :
            src_imon  = ' %s' % (name.value().ljust(32))
            name_imon = ' %s' % (short_name.value().ljust(16))
            bits      = ' %d %d %d %d   %d %d' % (ch1.value(), ch2.value(), ch3.value(), ch4.value(), norm.value(), sele.value())
            vals      = '   %9.4f %9.4f %9.4f' % (sele_min.value(), sele_max.value(), norm_ave.value())
            s         = src_imon + name_imon + bits + vals

            #if norm.value() or sele.value() : text += s + '\n' # Short form of the imon_cfg file
            text += s + '\n'
        return text


    def make_imon_cfg_file (self, fname='imon_cfg.txt') :
        text_table = self.get_text_table_of_imon_pars()
        logger.info('Make intensity monitors configuration file: ' + fname + '\n' + text_table, __name__)
        fout = open(fname,'w')
        fout.write(text_table)
        fout.close() 

#-----------------------------
#-----------------------------
#------- Core methods --------
#-----------------------------
#-----------------------------

    def make_cfg_file (self) :
        """ DEPRICTED
        """
        logger.info('Make configuration file: ' + self.path_out,__name__)
        logger.debug('path_cfg_stub = ' + self.path_in)
        logger.debug('path_cfg      = ' + self.path_out)
        #print 'path_cfg      = ' + self.path_out

        self.keys   = self.d_subs.keys()

        fin  = open(self.path_in, 'r')
        fout = open(self.path_out,'w')
        for line in fin :
            line_sub = self.line_with_substitution(line)
            fout.write(line_sub)
            #logger.info(line_sub)
            #print line_sub,

        fin .close() 
        fout.close() 

#-----------------------------

    def print_substitution_dict (self) :
        msg = '\nSubstitution dictionary for %s' % self.path_in
        for k,v in self.d_subs.iteritems() :
            msg += '\n%s : %s' % (k.ljust(16), v.ljust(32))
            logger.debug(msg)
        logger.debug(msg,__name__)
        if self.do_test_print : print msg

#-----------------------------

    def field_substituted(self, field) :
        if field in self.keys : return self.d_subs[field]
        else                  : return field

#-----------------------------

    def line_with_substitution(self, line) :
        fields = line.split()
        line_sub = ''
        for field in fields :

            field_sub = self.field_substituted(field)
            line_sub += field_sub + ' '

        line_sub.rstrip(' ')
        line_sub += '\n'
        return line_sub

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def text_for_section (self) :
        """Make txt for cfg file section 
        """
        logger.debug('Make text for: ' + self.path_in,__name__)

        self.keys   = self.d_subs.keys()

        txt = ''
        fin = open(self.path_in, 'r')
        for line in fin :
            line_sub = self.line_with_substitution(line)
            txt += line_sub
        fin.close() 

        return txt

#-----------------------------

    def add_module_in_cfg (self, module_name) :
        self.print_substitution_dict()
        self.str_of_modules += ' ' + module_name
        self.txt_cfg_body += self.text_for_section() + '\n\n'
        
#-----------------------------

    def save_cfg_file (self, text, path) :
        msg = '\nSave configuration file: %s' % path
        logger.info(msg,__name__)
        if self.do_test_print : print msg
        gu.save_textfile(text, path)

#-----------------------------
#-----------------------------

cfg = ConfigFileGenerator ()

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    cp.detector.setValue('CSPAD') # 'CSPAD', 'CSPAD2x2', 'Princeton', 'pnCCD', 'Tm6740', 'Opal2000', 'Opal4000'

    cfg_test = ConfigFileGenerator (do_test=True)

    #cfg_test.make_psana_cfg_file_for_peds_scan()
    cfg_test.make_psana_cfg_file_for_peds_aver()
    #cfg_test.make_psana_cfg_file_for_data_scan()
    #cfg_test.make_psana_cfg_file_for_data_aver()
    #cfg_test.make_psana_cfg_file_for_cora_split()

    sys.exit ( 'End of test for ConfigFileGenerator' )

#-----------------------------
