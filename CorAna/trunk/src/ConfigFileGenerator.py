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

from ConfigParametersCorAna import confpars as cp
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
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-peds-scan.cfg').path()
        self.path_out = fnm.path_peds_scan_psana_cfg()
        self.d_subs   = {'SKIP'                 : 'IS_NOT_USED',
                         'EVENTS'               : 'FOR_ALL_EVENTS',
                         'FNAME_TIMESTAMP_LIST' : fnm.path_peds_scan_tstamp_list()
                         }

        self.print_substitution_dict()
        self.make_cfg_file()

#-----------------------------

    def make_psana_cfg_file_for_peds_aver (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-peds-aver.cfg').path()
        self.path_out = fnm.path_peds_aver_psana_cfg()
        self.d_subs   = {'SKIP'           : str( cp.bat_dark_start.value() - 1 ),
                         'EVENTS'         : str( cp.bat_dark_end.value() - cp.bat_dark_start.value() + 1 ),
                         'IMG_REC_MODULE' : str( cp.bat_img_rec_mod.value() ),
                         'DETINFO'        : str( cp.bat_det_info.value() ),
                         'FILE_AVE'       : fnm.path_pedestals_ave(),
                         'FILE_RMS'       : fnm.path_pedestals_rms()
                         }

        self.print_substitution_dict()
        self.make_cfg_file()

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def make_psana_cfg_file_for_data_scan (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-data-scan.cfg').path()
        self.path_out = fnm.path_data_scan_psana_cfg()
        self.d_subs   = {'SKIP'                              : 'IS_NOT_USED',
                         'EVENTS'                            : 'FOR_ALL_EVENTS',
                         'FNAME_TIMESTAMP_LIST'              : fnm.path_data_scan_tstamp_list(),
                         'FNAME_INTENSITY_MONITORS_DATA'     : fnm.path_data_scan_monitors_data(),
                         'FNAME_INTENSITY_MONITORS_COMMENTS' : fnm.path_data_scan_monitors_commments()
                         }

        self.print_substitution_dict()
        self.make_cfg_file()

#-----------------------------

    def make_psana_cfg_file_for_data_aver (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-data-aver.cfg').path()
        self.path_out = fnm.path_data_aver_psana_cfg()
        self.d_subs   = {'SKIP'           : str( cp.bat_data_start.value() - 1 ),
                         'EVENTS'         : str( cp.bat_data_end.value() - cp.bat_data_start.value() + 1 ),
                         'IMG_REC_MODULE' : str( cp.bat_img_rec_mod.value() ),
                         'DETINFO'        : str( cp.bat_det_info.value() ),
                         'FILE_AVE'       : fnm.path_data_ave(),
                         'FILE_RMS'       : fnm.path_data_rms()
                         }

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

    def make_cfg_file (self) :

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
    cfg.make_psana_cfg_file_for_peds_scan()

    sys.exit ( 'End of test for ConfigFileGenerator' )

#-----------------------------
