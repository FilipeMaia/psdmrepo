#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module BatchLogScanParser...
#
#------------------------------------------------------------------------

"""Extracts required information from batch log files

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

#-----------------------------

class BatchLogScanParser :
    """Extracts EventKeys from batch log scan files
    """

    def __init__ (self) :
        """
        @param path   path to the input log file
        @param dict   dictionary of searched items and associated parameters
        """
        self.path              = None 
        self.det_name          = cp.det_name
        self.dict_of_det_types = cp.dict_of_det_types
        self.list_of_sources   = []
        self.list_of_types     = []

#-----------------------------

    def print_dict_of_det_types (self) :
        print 'List of detector names and associated types:'
        for det, type in self.dict_of_det_types.items():
            print '%10s : %s' % (det, type)

#-----------------------------

    def parse_batch_log_peds_scan (self) :
        self.path = fnm.path_peds_scan_batch_log()
        self.pattern = self.dict_of_det_types[self.det_name.value()]

        #self.print_dict_of_det_types()
        self.parse_scan_log()
        self.print_list_of_types_and_sources()
        
#-----------------------------

    def parse_scan_log (self) :
        if not os.path.lexists(self.path) :
            logger.debug('The requested scan log file: ' + self.path + ' is not available.', __name__)         
            return

        list_of_found_lines = []

        fin = open(self.path, 'r')
        for line in fin :
            if self.pattern in line :
                if line in list_of_found_lines : continue # if the line is already in the list                
                list_of_found_lines.append(line)

        fin.close() 

        self.list_of_sources = []
        self.list_of_types   = []

        for line in list_of_found_lines : 

            pos1 = line.find('type=') + 5
            wid1 = line[pos1:].find(',')
            pen1 = pos1+wid1
            type = line[pos1:pen1]
            self.list_of_types.append(type)
            #print 'pos1, wid1, type:', pos1, wid1, type
            
            pos2 = line[pen1:].find('src=') + pen1 + 4
            wid2 = line[pos2:].find(')')
            pen2 = pos2+wid2+1
            src  = line[pos2:pen2]
            self.list_of_sources.append(src)
            #print 'pos2, wid2, src :', pos2, wid2, src

#-----------------------------

    def print_list_of_types_and_sources (self) :
        msg = 'In log file: %s\nsearch pattern: %s for detector: %s' % (self.path, self.pattern, self.det_name.value())

        if self.list_of_sources == [] :
            msg += '\nLIST OF SOURCES IS EMPTY !!!' 
            logger.warning(msg, __name__)         
            return

        for type, src in zip(self.list_of_types, self.list_of_sources) :
            msg += '\n    %30s : %s' % (type, src)
        #print msg
        logger.info(msg, __name__)         

#-----------------------------

    def get_list_of_sources (self) :
        #if self.list_of_sources == [] : return None
        return self.list_of_sources

    def get_list_of_types (self) :
        #if self.list_of_types == [] : return None
        return self.list_of_types



#-----------------------------

    def get_list_of_files_for_all_sources(self, path1='work/file.dat') :
        """From pattern of the path it makes a list of files with indexes for sources.
        For example, for path1='work/file.dat', it returns [work/file.0.dat, work/file.1.dat, ...]
        """
        list_of_files = []
        len_of_list = len(self.list_of_sources) 
        if len_of_list == 0 : return [path1]

        name, ext = os.path.splitext(path1)
        for i in range(len_of_list) :
            name_i = '%s.%i%s' % (name, i, ext)
            list_of_files.append(name_i)
        return list_of_files

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

blsp = BatchLogScanParser ()

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    blsp.parse_batch_log_peds_scan()

    sys.exit ( 'End of test for BatchLogScanParser' )

#-----------------------------  
