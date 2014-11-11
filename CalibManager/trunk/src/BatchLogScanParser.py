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

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
from time import sleep

from ConfigParametersForApp import cp
from Logger                 import logger
from FileNameManager        import fnm
import GlobalUtils          as     gu
import RegDBUtils           as     ru

#-----------------------------

class BatchLogScanParser :
    """Extracts EventKeys from batch log scan files
    """

    dict_exists = {True  : 'is available', 
                   False : 'is NOT available'}


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

#-----------------------------

    def parse_batch_log_peds_scan (self) :
        """Psrses log file for dark run scan and makes lists:
           self.list_of_types and self.list_of_sources for all psana data types in file.
        """

        if  self.path == fnm.path_peds_scan_batch_log() : return
        #and self.det_names_parsed == self.det_name.value(): return
        #self.det_names_parsed = self.det_name.value()


        #if self.det_name.value == self.det_name.value_def() : return

        self.list_of_detinfo_sources = []
        self.list_of_sources         = []
        self.list_of_types           = []

        if not self.make_set_of_lines_from_file_for_pattern() : return
        self.make_list_of_types_and_sources()

        #print 'List of types and sources:'
        #for type, src in zip(self.list_of_types, self.list_of_sources) :
        #    msg = '    %30s : %s' % (type, src)
        #    print msg
    

        #for det_name in self.list_of_dets_selected() :
            #self.pattern = self.dict_of_det_data_types[det_name]
            # print 'Parse file: %s for detector: %s and pattern: %s' % (self.path, det_name, self.pattern)
            #cp.print_dict_of_det_data_types()
            #self.parse_list_of_lines()

#-----------------------------

    def make_set_of_lines_from_file_for_pattern(self, pattern = 'EventKey(type=Psana' ) :
        """Makse self.list_of_found_lines - a set of lines from file specified containing pattern
        """

        self.path = fnm.path_peds_scan_batch_log()

        if not os.path.lexists(self.path) :
            logger.info('\nThe requested file: ' + self.path + '\nIS NOT AVAILABLE!', __name__)
            return False

        self.list_of_found_lines  = []

        fin = open(self.path, 'r')
        for line in fin :
            if pattern in line :
                line_st = line.rstrip('\n').strip(' ')               
                if line_st in self.list_of_found_lines : continue # if the line is already in the list
                self.list_of_found_lines.append(line_st)
                #print 'found line:', line_st

        fin.close() 

        return True

#-----------------------------

    def make_list_of_types_and_sources (self) :

        for line in self.list_of_found_lines : 

            pos1 = line.find('type=Psana::') + 12
            wid1 = line[pos1:].find(',')
            pen1 = pos1+wid1
            type = line[pos1:pen1]
            if type.find('ConfigV') != -1 : continue # remove ConfigV from lists
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
        #if self.path is None or not os.path.exists(self.path) :
        #    msg = 'log file: %s IS NOT AVAILABLE' % (self.path)
        #    return msg

        self.parse_batch_log_peds_scan()
        msg   = 'log file: %s \nExpecting data for detector(s): %s' % (self.path, self.det_name.value())
        state = 'Sources found in scan:' 
        if self.list_of_sources == [] :
            msg += '\nLog file %s' % self.dict_exists[self.scan_log_exists()] # is available or not
            msg += '\nLIST OF SOURCES IS EMPTY !!!'

            return msg

        for type, src in zip(self.list_of_types, self.list_of_sources) :
            line  = '\n    %30s : %s' % (type, src)
            msg   += line
            state += line
        return msg


    def txt_of_sources_in_run(self) :
        return ru.txt_of_sources_in_run(cp.instr_name.value(), cp.exp_name.value(), int(cp.str_run_number.value()))

#-----------------------------
#-----------------------------

    def scan_log_exists (self) :
        return os.path.exists(fnm.path_peds_scan_batch_log())


    def get_list_of_sources (self) :
        if self.scan_log_exists() :
            self.parse_batch_log_peds_scan()        
            return self.list_of_sources
        # Use RegDB
        return ru.list_of_sources_in_run(cp.instr_name.value(), cp.exp_name.value(), int(cp.str_run_number.value()))


    def get_list_of_types (self) :
        self.parse_batch_log_peds_scan()
        return self.list_of_types


    def get_list_of_type_sources (self) :
        self.parse_batch_log_peds_scan()
        return zip(self.list_of_types, self.list_of_sources)


    def list_of_types_and_sources_for_detector (self, det_name) :

        if not self.scan_log_exists() : # Use RegDB
            ins, exp, run_number = cp.instr_name.value(), cp.exp_name.value(), int(cp.str_run_number.value())
            lst_srcs = ru.list_of_sources_in_run_for_selected_detector(ins, exp, run_number, det_name)
            dtype = cp.dict_of_det_data_types[det_name]
            ctype = cp.dict_of_det_calib_types[det_name]
            lst_dtypes = [dtype for src in lst_srcs]
            lst_ctypes = [ctype for src in lst_srcs]
            #print 'lst_ctypes ::: ', lst_ctypes
            #print 'lst_dtypes ::: ', lst_dtypes
            #print 'lst_srcs   ::: ', lst_srcs
            return lst_dtypes, lst_srcs, lst_ctypes

        pattern_det = det_name.lower() + '.'
        pattern_type = self.dict_of_det_data_types[det_name]
        #print 'pattern_det, pattern_type', pattern_det, pattern_type

        list_of_ctypes_for_det=[]
        list_of_dtypes_for_det=[]
        list_of_srcs_for_det=[]
        for type,src in self.get_list_of_type_sources() :
            #print '  type, src: %24s  %s' % (type,src)
            if type.find(pattern_type)       == -1 : continue
            if src.lower().find(pattern_det) == -1 : continue
            list_of_ctypes_for_det.append(cp.dict_of_det_calib_types[det_name])
            list_of_dtypes_for_det.append(type)
            list_of_srcs_for_det.append(src)
        #print 'list of types and sources for detector %s:\n  %s\n  %s' \
        #      % (det_name, str(list_of_types_for_det), str(list_of_srcs_for_det))  
        return list_of_dtypes_for_det, list_of_srcs_for_det, list_of_ctypes_for_det



    def list_of_types_and_sources_for_selected_detectors (self) :
        """Returns the list of data types, sources, and calib types in run for selected detector.
        For example, for CSPAD returns
        ['CsPad::DataV2',    'CsPad::DataV2'],
        ['CxiDs1.0:Cspad.0', 'CxiDs2.0:Cspad.0']
        ['CsPad::CalibV1',   'CsPad::CalibV1'],
        """
        lst_ctypes = []
        lst_types  = []
        lst_srcs   = []

        for det_name in cp.list_of_dets_selected() :
            lst_t, lst_s, lst_c = self.list_of_types_and_sources_for_detector(det_name)

            #print 'lst_t: ', lst_t
            #print 'lst_s: ', lst_s
            #print 'lst_c: ', lst_c

            lst_ctypes += lst_c
            lst_types += lst_t
            lst_srcs += lst_s

        return lst_types, lst_srcs, lst_ctypes



    def list_of_sources_for_selected_detectors (self) :
        """Returns the list of sources in run for selected detectors.
        """
        lst_types, lst_srcs, lst_ctypes = self.list_of_types_and_sources_for_selected_detectors()
        return lst_srcs


#-----------------------------
#-----------------------------

blsp = BatchLogScanParser ()
cp.blsp = blsp

#-----------------------------
#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    #cp.blsp.parse_batch_log_peds_scan()

    cp.blsp.print_list_of_types_and_sources ()

    sys.exit ( 'End of test for BatchLogScanParser' )

#-----------------------------  
