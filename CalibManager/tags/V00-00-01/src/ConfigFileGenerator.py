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

from ConfigParametersForApp import cp
from Logger                 import logger
from FileNameManager        import fnm

#import AppUtils.AppDataPath as apputils
import           AppDataPath as apputils # My version, added in path the '../../data:'

#-----------------------------

class ConfigFileGenerator :
    """Generates the configuration files for psana from current configuration parameters
    """

    def __init__ (self) :
        """
        @param path_in  path to the input psana configuration stub-file
        @param path_out path to the output psana configuration file with performed substitutions
        @param d_subs   dictionary of substitutions
        @param keys     keys from the dictionary       
        """
        path_in  = None 
        path_out = None 
        d_subs   = None
        keys     = None 

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def make_psana_cfg_file_for_peds_scan (self) :
        self.path_in  = apputils.AppDataPath('CalibManager/scripts/psana-peds-scan.cfg').path()
        self.path_out = fnm.path_peds_scan_psana_cfg()
        self.d_subs   = {'FNAME_XTC'            : str(fnm.path_to_xtc_files_for_run()),
                         'SKIP'                 : 'IS_NOT_USED',
                         'EVENTS'               : 'FOR_ALL_EVENTS',
                         }

        self.print_substitution_dict()
        self.make_cfg_file()

#-----------------------------

    def make_psana_cfg_file_for_peds_aver (self) :

        if cp.blsp.list_of_sources == [] : return

        self.det_name = cp.det_name.value()

        # list_of_dets   = ['CSPAD', 'CSPAD2x2', 'Camera', 'Princeton', 'pnCCD'] 

        if   self.det_name == cp.list_of_dets[0] : self.make_psana_cfg_file_for_peds_aver_cspad(module='cspad_mod.CsPadPedestals')
        elif self.det_name == cp.list_of_dets[1] : self.make_psana_cfg_file_for_peds_aver_cspad(module='cspad_mod.CsPad2x2Pedestals')
        elif self.det_name == cp.list_of_dets[2] : self.print_worning()
        elif self.det_name == cp.list_of_dets[3] : self.make_psana_cfg_file_for_peds_aver_princeton()
        elif self.det_name == cp.list_of_dets[4] : print_worning()
        else : logger.warning('UNKNOWN DETECTOR: %s' % self.det_name, __name__)


    def print_warning (self) :
        msg = 'make_psana_cfg_file_for_peds_aver_%s - IS NOT IMPLEMENTED YET!!!' % self.det_name
        logger.warning(msg, __name__)

#-----------------------------

    def make_psana_cfg_file_for_peds_aver_cspad (self, module='cspad_mod.CsPadPedestals') :
        self.path_in  = apputils.AppDataPath('CalibManager/scripts/psana-section-psana.cfg').path()
        self.path_out = fnm.path_peds_aver_psana_cfg()

        modules = ''
        for i in range( len(cp.blsp.list_of_sources) ) : modules += '%s:%i ' % (module, i)
        #print 'List of modules: %s' % modules

        self.d_subs   = {'FNAME_XTC'        : str(fnm.path_to_xtc_files_for_run()),
                         'SKIP'             : str( cp.bat_dark_start.value() - 1 ),
                         'EVENTS'           : str( cp.bat_dark_end.value() - cp.bat_dark_start.value() + 1 ),
                         'MODULES'          : modules
                         }

        self.print_substitution_dict()
        self.make_cfg_file()

        #Add a few similar modules in loop
        self.path_in  = apputils.AppDataPath('CalibManager/scripts/psana-section-module-cspad-peds.cfg').path()

        list_of_ave = cp.blsp.get_list_of_files_for_all_sources(fnm.path_peds_ave())
        list_of_rms = cp.blsp.get_list_of_files_for_all_sources(fnm.path_peds_rms())
        
        for i, (source, fname_ave, fname_rms) in enumerate( zip(cp.blsp.list_of_sources, list_of_ave, list_of_rms) ) :
            mod = '%s:%i' % (module, i)
            #print '   Add module with pars: ', mod, source, fname_ave, fname_rms

            self.d_subs = {'MODULE'           : mod,
                           'DETINFO'          : source,
                           'FNAME_PEDS_AVE'   : fname_ave,
                           'FNAME_PEDS_RMS'   : fname_rms
                          }

            #for item in self.d_subs.items() :
            #    print '%20s : %s' % item

            self.make_cfg_file(fout_mode='a')


#-----------------------------

    def make_psana_cfg_file_for_peds_aver_princeton (self) :
        self.path_in  = apputils.AppDataPath('CalibManager/scripts/psana-peds-aver-princeton.cfg').path()
        self.path_out = fnm.path_peds_aver_psana_cfg()
        self.d_subs   = {'FNAME_XTC'      : str(fnm.path_to_xtc_files_for_run()),
                         'SKIP'           : str( cp.bat_dark_start.value() - 1 ),
                         'EVENTS'         : str( cp.bat_dark_end.value() - cp.bat_dark_start.value() + 1 ),
                         'IMG_REC_MODULE' : str( cp.bat_img_rec_mod.value() ),
                         'DETINFO'        : str( cp.bat_det_info.value() ),
                         'FNAME_PEDS_AVE' : fnm.path_peds_ave(),
                         'FNAME_PEDS_RMS' : fnm.path_peds_rms()
                         }

        self.d_subs['FNAME_HOTPIX_MASK'   ] = fnm.path_hotpix_mask()
        self.d_subs['HOTPIX_THRESHOLD_ADU'] = str( cp.mask_hot_thr.value() )

        #if cp.mask_hot_is_used.value() : 
        #    self.d_subs['FNAME_HOTPIX_MASK'   ] = fnm.path_hotpix_mask()
        #    self.d_subs['HOTPIX_THRESHOLD_ADU'] = str( cp.mask_hot_thr.value() )
        #else :
        #    self.d_subs['FNAME_HOTPIX_MASK'   ] = ''
        #    self.d_subs['HOTPIX_THRESHOLD_ADU'] = '10000'

        self.print_substitution_dict()
        self.make_cfg_file()

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def print_substitution_dict (self) :
        logger.debug('Substitution dictionary:',__name__)
        for k,v in self.d_subs.iteritems() :
            msg = '%s : %s' % (k.ljust(16), v.ljust(32))
            logger.debug(msg)


#-----------------------------

    def make_cfg_file (self, fout_mode='w') :

        logger.info('Make configuration file: ' + self.path_out,__name__)
        logger.debug('path_cfg_stub = ' + self.path_in)
        logger.debug('path_cfg      = ' + self.path_out)
        #print 'path_cfg      = ' + self.path_out

        self.keys   = self.d_subs.keys()

        fin  = open(self.path_in,  'r')
        fout = open(self.path_out, fout_mode)
        for line in fin :
            line_sub = self.line_with_substitution(line)
            fout.write(line_sub)
            #logger.info(line_sub)
            #print line_sub,

        fin .close() 
        fout.close() 

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

    def field_substituted(self, field) :
        if field in self.keys : return self.d_subs[field]
        else                  : return field

#-----------------------------

cfg = ConfigFileGenerator ()

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    #cfg.make_psana_cfg_file_for_peds()
    #cfg.make_psana_cfg_file_for_peds_scan()


    cp.blsp.parse_batch_log_peds_scan() # defines the cp.blsp.list_of_sources    
    cfg.make_psana_cfg_file_for_peds_aver()

    sys.exit ( 'End of test for ConfigFileGenerator' )

#-----------------------------
