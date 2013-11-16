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
import GlobalUtils          as     gu

#-----------------------------

class BatchLogScanParser :
    """Extracts EventKeys from batch log scan files
    """

    def __init__ (self) :
        """
        @param path   path to the input log file
        @param dict   dictionary of searched items and associated parameters
        """
        self.det_name               = cp.det_name
        self.dict_of_det_data_types = cp.dict_of_det_data_types
        self.list_of_dets_selected  = cp.list_of_dets_selected # reference to method
        self.list_of_sources        = []
        self.list_of_types          = []

        self.pattern                = 'N/A'
        self.det_names_parsed       = None 
        self.path                   = None 
        self.is_parsed              = False 


#-----------------------------

    def print_dict_of_det_data_types (self) :
        print 'List of detector names and associated types:'
        for det, type in self.dict_of_det_data_types.items():
            print '%10s : %s' % (det, type)

#-----------------------------

    def parse_batch_log_peds_scan (self) :

        if  self.is_parsed \
        and self.path == fnm.path_peds_scan_batch_log() \
        and self.det_names_parsed == self.det_name.value(): return

        #if self.det_name.value == self.det_name.value_def() : return

        self.path = fnm.path_peds_scan_batch_log()

        if not os.path.lexists(self.path) :
            logger.info('\nThe requested scan log file: ' + self.path + '\nIS NOT AVAILABLE!', __name__)         
            return

        self.list_of_detinfo_sources = []
        self.list_of_sources         = []
        self.list_of_types           = []

        for det_name in self.list_of_dets_selected() :

            self.pattern = self.dict_of_det_data_types[det_name]

            # print 'Parse file: %s for detector: %s and pattern: %s' % (self.path, det_name, self.pattern)

            #self.print_dict_of_det_data_types()
            self.parse_scan_log()
        #self.print_list_of_types_and_sources()
        self.is_parsed = True
        self.det_names_parsed = self.det_name.value()
        
#-----------------------------

    def parse_scan_log (self) :

        list_of_found_lines  = []

        fin = open(self.path, 'r')
        for line in fin :
            if self.pattern in line :
                if line in list_of_found_lines : continue # if the line is already in the list                
                list_of_found_lines.append(line)

        fin.close() 

        for line in list_of_found_lines : 

            pos1 = line.find('type=Psana::') + 12
            wid1 = line[pos1:].find(',')
            pen1 = pos1+wid1
            type = line[pos1:pen1]
            self.list_of_types.append(type)
            #print 'pos1, wid1, type:', pos1, wid1, type
            
            pos2 = line[pen1:].find('src=') + pen1 + 4
            wid2 = line[pos2:].find(')')
            pen2 = pos2+wid2+1
            detinfo_src = line[pos2:pen2]
            self.list_of_detinfo_sources.append(detinfo_src)
            #print 'pos2, wid2, detinfo_src:', pos2, wid2, detinfo_src

            pos3 = line[pos2:].find('(') + pos2 + 1
            wid3 = line[pos3:].find(')')
            pen3 = pos3+wid3
            src  = line[pos3:pen3]
            self.list_of_sources.append(src)
            #print 'pos3, wid3, pen3, src:', pos3, wid3, pen3, src
            

#-----------------------------

    def print_list_of_types_and_sources (self) :
        txt = self.txt_list_of_types_and_sources()
        logger.info(txt, __name__)         
        #print txt

#-----------------------------

    def txt_list_of_types_and_sources (self) :        
        self.parse_batch_log_peds_scan()
        msg   = 'log file: %s \nfor detector(s): %s' % (self.path, self.det_name.value())
        state = 'Sources found in scan:' 
        if self.list_of_sources == [] :
            msg += '\nLIST OF SOURCES IS EMPTY !!!'
            return msg

        for type, src in zip(self.list_of_types, self.list_of_sources) :
            line  = '\n    %30s : %s' % (type, src)
            msg   += line
            state += line
        return msg

#-----------------------------

    def get_list_of_detinfo_sources (self) :
        #if self.list_of_sources == [] : return None
        self.parse_batch_log_peds_scan()        
        return self.list_of_detinfo_sources

    def get_list_of_sources (self) :
        #if self.list_of_sources == [] : return None
        self.parse_batch_log_peds_scan()        
        return self.list_of_sources

    def get_list_of_types (self) :
        #if self.list_of_types == [] : return None
        self.parse_batch_log_peds_scan()
        return self.list_of_types

#-----------------------------

    def get_list_of_files_for_all_sources(self, path1='work/file.dat') :
        """From pattern of the path it makes a list of files with indexes for all sources."""
        #self.print_list_of_types_and_sources()
        return gu.get_list_of_files_for_list_of_insets( path1, self.get_list_of_sources() )

        #len_of_list = len(self.list_of_sources)
        ##print 'len_of_list =', len_of_list
        #return gu.get_list_of_enumerated_file_names(path1, len_of_list)

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

blsp = BatchLogScanParser ()
cp.blsp = blsp

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    #blsp.parse_batch_log_peds_scan()

    blsp.print_list_of_types_and_sources ()

    sys.exit ( 'End of test for BatchLogScanParser' )

#-----------------------------  
