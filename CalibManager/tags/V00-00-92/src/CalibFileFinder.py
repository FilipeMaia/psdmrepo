#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CalibFileFinder...
#
#------------------------------------------------------------------------

"""Python analog of the module PSCalib/CalibFileFinder

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

from Logger                   import logger
#import GlobalUtils            as     gu

#-----------------------------

class CalibFile() :
    """Calib file name, run range, etc
    """
    max_run_number = 9999

    def __init__(self, path) :
        """Constructor.
        @param path path to the calibration file
        """
        self.is_calibfile = True
        self.path = path
        self.basename = os.path.basename(path)
        fname, ext = os.path.splitext(self.basename)
        if ext != '.data'  :
            self.is_calibfile = False
            logger.debug('File %s extension is not ".data"' % self.basename, __name__)
            return 
        if not '-' in fname :
            self.is_calibfile = False
            logger.debug('File %s basename does not have "-"' % self.basename, __name__)
            return None

        str_begin, str_end = fname.split('-')    
        self.begin = int(str_begin)
        self.end = self.max_run_number
        if str_end != 'end' : self.end = int(str_end)

#----------------------------------

    def is_calib_file(self) : return self.is_calibfile

    def get_path(self) : return self.path

    def get_basename(self) : return self.basename

    def get_begin(self) : return self.begin

    def get_end(self) : return self.end

#----------------------------------

    def run_is_in_range(self, runnum) :
        if self.begin <= runnum and runnum <= self.end : return True
        else                                           : return False

#----------------------------------

    def __cmp__(self, other) :
        """Method for sorted()"""
        if self.begin  < other.begin : return -1
        if self.begin  > other.begin : return  1
        if self.begin == other.begin : 
            if self.end  < other.end : return  1 # inverse comparison for end
            if self.end  > other.end : return -1 
            if self.end == other.end : return  0 

#----------------------------------

    def print_member_data(self) :
        if self.is_calibfile : 
            print 'path %s,  run range: %04d-%04d' % (self.path.rjust(14), self.begin, self.end)          
        else :
            print 'IS NOT A CALIBRATION FILE: %s' % self.path
                    
#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------

class CalibFileFinder() :
    """Calib file name, run range, etc
    """
    max_run_number = 9999

    def __init__(self, path_to_calib_types, calib_type='pedestals') :
        """Constructor.
        @param path_to_calib_types - path to the directory with calibration types
        @param calib_type - calibration type, for example "pedestals", "comm_mode", "pixel_status", etc.
        """
        self.path_to_calib_types = path_to_calib_types
        self.type = calib_type
        self.path = os.path.join(path_to_calib_types, calib_type)

        if not os.path.exists(self.path) :
            msg = 'Path %s DOES NOT EXIST' % self.path
            print msg
            logger.error(msg, __name__)
            return None


#----------------------------------

    def print_member_data(self) :
        print 'path_to_calib_types : %s' % self.path_to_calib_types         
        print 'type                : %s' % self.type          
        print 'path                : %s' % self.path          

#----------------------------------

    def find_calib_file(self, runnum=0) :
        """returns string path to found calibration file of empty string
        """
        cfile = find_calib_file_in_list_for_run(self.list_of_sorted_calib_files(), runnum)
        #cfile.print_member_data()
        if cfile is None : return ''
        else : return os.path.join(self.path, cfile.get_basename())

#----------------------------------

    def list_of_sorted_calib_files(self) :

        list_of_fnames = os.listdir(self.path)
        if list_of_fnames == [] :
            logger.warning('Directory %s IS EMPTY!' % self.path, __name__)
            return []
        return list_of_sorted_calib_files_from_list_of_files(list_of_fnames)

#----------------------------------

def list_of_sorted_calib_files_from_list_of_files(list_of_files) :
    """Returns the list of CalibFile objects for specified list of files or [] 
    """    
    list_of_calib_files = []
    for file in list_of_files :
        cfile = CalibFile(str(file))
        if cfile.is_calib_file() :
            list_of_calib_files.append(cfile)
            #cfile.print_member_data()

    return sorted(list_of_calib_files) # sorted() uses reimplemented method CalibFile.__cmp__()

#----------------------------------

def find_calib_file_in_list_for_run(list_sorted_cfiles, runnum=0) :
    """Returns CalibFile object or None 
    """
    for cfile in list_sorted_cfiles[-1::-1] : # reverse iterator
        #cfile.print_member_data()
        if cfile.run_is_in_range(runnum) : return cfile
    return None

#----------------------------------

def dict_calib_file_actual_run_range(list_of_cfiles) :

    list_of_ends   = [cfile.get_begin()-1 for cfile in list_of_cfiles]
    #list_of_ends.append(0)                              # add minimal end=0
    list_of_ends.append(CalibFileFinder.max_run_number) # add maximal end=9999
    if len(list_of_cfiles)>0 :
        list_of_ends.append(list_of_cfiles[-1].get_end())   # end of the last file
    
    # Fill dictionary with begin-end from file name
    #dict_fname_range = { cfile.get_basename():[cfile.get_begin(), cfile.get_end()] for cfile in list_of_cfiles }
    dict_fname_range = { cfile.get_basename():[-1, -1] for cfile in list_of_cfiles }

    # Substitute ends from file finder
    for end in list_of_ends :
        cfile = find_calib_file_in_list_for_run(list_of_cfiles, runnum=end)
        if cfile is None : continue
        dict_fname_range[cfile.get_basename()] = [cfile.get_begin(), end]
    
    return dict_fname_range

#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------
#
#  Test of class CalibFile
#
if __name__ == "__main__" :

    list_of_files = ['220-230.data', '220-end.data', '221-240.data', '528-end.data', '222-end.data', '659-800.data', '373-end.data', '79-end.data', '45-end.data'] 


    print '\n\nTest class CalibFile'
    print '\nShafled list of calibration files'
    list_of_calib_files = []
    for file in list_of_files :
        calib_file = CalibFile(file)
        list_of_calib_files.append(calib_file)
        calib_file.print_member_data()


    print '\nSorted list of calibration files'
    for cfile in sorted(list_of_calib_files) :
        cfile.print_member_data()



    print '\n\nTest class CalibFileFinder'
    #cff = CalibFileFinder("/reg/d/psdm/CXI/cxitut13/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/", "offset_corr")
    cff = CalibFileFinder("/reg/d/psdm/MEC/meca1113/calib/CsPad::CalibV1/MecTargetChamber.0:Cspad.0", "pedestals")
    cff.print_member_data()
    runnum = 232
    print 'For run %d: %s' % (runnum, cff.find_calib_file(runnum))



    print '\n\nTest methods for run ranges:'
    list_of_cfiles = list_of_sorted_calib_files_from_list_of_files(list_of_files)
    dict_fname_range = dict_calib_file_actual_run_range(list_of_cfiles)

    for cfile in list_of_cfiles :
        #print cfile.get_basename()
        fname = cfile.get_basename()
        range = dict_fname_range[fname]

        txt = '%s  run range %04d - %04d' % (fname.rjust(14), range[0], range[1])
        if range[0] == -1 : txt = '%s  file is not used' % fname.rjust(14)

        print txt

    sys.exit ( "End of test" )

#----------------------------------
    
