#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module calib...
#
#------------------------------------------------------------------------

"""Classes and functions to deal with calibration data

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andy Salnikov
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
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc

#----------------------------------
# Local non-exported definitions --
#----------------------------------

def _detInfo2Addr(dinfo):
    """Converts DetInfo object into address string"""
    detector = str(dinfo.detector()).split('.')[1]
    device = str(dinfo.device()).split('.')[1]
    return "%s.%d:%s.%d" % (detector, dinfo.detId(), device, dinfo.devId())

def _parseFile(name):
    """Parses file name which should have format "run1-run2.data"
    and returns tuple (run1, run2, name). Run2 can be string "end",
    for that number 2^32-1 will be returned.
    """

    base = os.path.basename(name)
    base, ext = os.path.splitext(base)
    if ext != ".data":
        logging.debug("skipping file: %s, wrong extension", name)
        return

    runs = base.split('-')
    if len(runs) != 2:
        logging.debug("skipping file: %s, incorrect number of dashes", name)
        return

    begin, end = runs
    if end == "end":
        end = 0xffffffff
    try:
        begin = int(begin)
        end = int(end)
    except ValueError, ex:
        logging.debug("skipping file: %s, incorrect number format", name)
        return

    return (begin, end, name)

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------

class CalibFileFinder(object):
    """Utility class for finding files in calibration directory. This
    implementation is based on C++ class PSCalib/CalibFileFinder."""
    
    def __init__(self, calibDir, className):
        """
        CalibFileFinder(calibDir: string, typeGroupName: string) -> object
        
        Make a finder instance. Takes two string parameters, calibration directory 
        location for current experiment (use environment class to obtain it), and
        calibration class name such as "CsPad::CalibV1" or "CsPad2x2::CalibV1".
        """

        self.m_calibDir = calibDir
        self.m_className = className

    def findCalibFile(self, src, dataType, run):
        """
        self.findCalibFile(src, dataType: string, run: int) -> string
        
        Finds calibration file for given parameter set and returns its name or None.
        First parameter is the data source which can be either string (format is
        "Detector-N|Device-M" or "Detector.N:Device.M") or DetInfo object.
        *dataType* is the type of calibration such as "pedestals", "common_mode", etc.
        If no file found then None is returned.
        """
        
        if not self.m_calibDir: return
        
        if type(src) == xtc.DetInfo:
            addr = _detInfo2Addr(src)
        else:
            # must be a string, need canonical format
            addr = src.replace('-', '.').replace('|', ':')
        
        dir = os.path.join(self.m_calibDir, self.m_className, addr, dataType)
        logging.debug('CalibFileFinder: directory = %s', dir)

        # parse all file names, skip unparseable names
        files = [file for file in [_parseFile(os.path.join(dir, file)) for file in os.listdir(dir)] if file]
        
        # sort in reverse
        files.sort()
        files.reverse()
        
        # find latest file containing this run
        for f in files:
            logging.debug("CalibFileFinder: trying file %s for run %d", f[2], run)
            if f[0] <= run <= f[1]: 
                return f[2] 
                 
        logging.debug("CalibFileFinder: no match found for run %d", run)
            

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
