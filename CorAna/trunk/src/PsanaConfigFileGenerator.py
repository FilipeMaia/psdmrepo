#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PsanaConfigFileGenerator...
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

class PsanaConfigFileGenerator :
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

    def make_psana_cfg_file_for_pedestals (self) :
        self.path_in  = apputils.AppDataPath('CorAna/scripts/psana-pedestals.cfg').path()
        self.path_out = fnm.path_psana_cfg_pedestals()
        self.d_subs   = {'SKIP'     : '100',
                         'EVENTS'   : '1000',
                         'DETINFO'  : 'DetInfo(CxiDs1.0:Cspad.0)',
                         'FILE_AVE' : 'cspad-cxi49012-r0027-pedestals-ave.dat',
                         'FILE_RMS' : 'cspad-cxi49012-r0027-pedestals-rms.dat'
                         }
        
        self.make_psana_cfg_file ()

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

    def make_psana_cfg_file (self) :

        print 'path_psana_cfg_stub = ', self.path_in
        print 'path_psana_cfg      = ', self.path_out

        self.keys   = self.d_subs.keys()

        #for k,v in self.d_subs.iteritems() :
        #    print k, v

        #for k in self.keys :
        #    print k,
        #print '\n'

        fin  = open(self.path_in, 'r')
        fout = open(self.path_out,'w')
        for line in fin :

            line_sub = self.line_with_substitution(line)
            fout.write(line_sub)
            print line_sub,

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

pcfg = PsanaConfigFileGenerator ()

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    pcfg.make_psana_cfg_file_for_pedestals()

    sys.exit ( 'End of test for PsanaConfigFileGenerator' )

#-----------------------------
