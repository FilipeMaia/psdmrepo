#!/usr/bin/env python
#------------------------------
""":py:class:`PSCalib.CalibFileFinder` is a python version of CalibFileFinder.cpp - finds calibration file

Usage::

    from PSCalib.CalibFileFinder import CalibFileFinder

    cdir  = '/reg/d/psdm/CXI/cxi83714/calib/'
    group = 'CsPad::CalibV1'
    src   = 'CxiDs1.0:Cspad.0'
    type  = 'pedestals'
    rnum  = 137

    cff = CalibFileFinder(cdir, group, pbits=0377)
    fname = cff.findCalibFile(src, type, rnum)


This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

Revision: $Revision$

@version $Id: CalibFileFinder.py 8469 2014-06-24 22:55:21Z dubrovin@SLAC.STANFORD.EDU $

@author Mikhail S. Dubrovin
"""
#--------------------------------
__version__ = "$Revision$"
#--------------------------------

import os
import sys

#------------------------------

class CalibFile :

    rnum_max = 9999

    def __init__(self, path='', pbits=1) :
        self.path = path
        self.pbits = pbits
        
        fname = os.path.basename(path)
        basename = os.path.splitext(fname)[0]

        if not ('-' in basename) :
            if self.pbits & 1  : print 'WARNING! MISSING DASH IN FILENAME "%s"' % basename
            return
            
        begin, end = basename.split('-')
        self.begin = int(begin)
        self.end   = int(end) if end != 'end' else self.rnum_max

    def get_path(self) :
        return self.path

    def get_begin(self) :
        return self.begin

    def get_end(self) :
        return self.end

    def __cmp__(self, other) :        
        #if self.begin != other.begin : return self.begin < other.begin
        #return self.end > other.end

        if   self.begin < other.begin : return -1
        elif self.begin > other.begin : return  1
        else :
            if   self.end > other.end : return -1
            elif self.end < other.end : return  1
            else : return 0

    def str_attrs(self) : 
        return 'begin: %4d  end: %4d  path: %s' % (self.begin, self.end, self.path)

#------------------------------

class CalibFileFinder :

    def __init__(self, cdir='', group='', pbits=1) :
        self.cdir  = cdir
        self.group = group
        self.pbits = pbits


    def findCalibFile(self, src, type, rnum0) :
        """Find calibration file
        """
        rnum = rnum0 if rnum0 <= CalibFile.rnum_max else CalibFile.rnum_max

        if self.cdir == '' :
            if self.pbits & 1 : print 'WARNING! CALIBRATION DIRECTORY IS EMPTY'
            return ''

        dir_name = os.path.join(self.cdir, self.group, src, type)
        if not os.path.exists(dir_name) :
            if self.pbits & 1  : print 'WARNING! NON-EXISTENT DIR: %s' % dir_name
            return ''

        fnames = os.listdir(dir_name)
        files = [os.path.join(dir_name,fname) for fname in fnames]
        return self.selectCalibFile(files, rnum) 


    def selectCalibFile(self, files, rnum) :
        """Selects calibration file from a list of file names
        """

        if self.pbits & 2 : print '\nUnsorted list of *.data files in the calib directory:'
        list_cf = []
        for path in files : 
           fname = os.path.basename(path)

           if fname is 'HISTORY' : continue
           if os.path.splitext(fname)[1] != '.data' : continue

           cf = CalibFile(path)
           if self.pbits & 2 : print cf.str_attrs()
           list_cf.append(cf)
           
        # sotr list
        list_cf_ord = sorted(list_cf)
        
        # print entire sorted list
        if self.pbits & 4 :
            print '\nSorted list of *.data files in the calib directory:'
            for cf in list_cf_ord[::-1] :
                if self.pbits & 4 : print cf.str_attrs()

        # search for the calibration file
        for cf in list_cf_ord[::-1] :
            if cf.get_begin() <= rnum and rnum <= cf.get_end() :
                if self.pbits & 8 :
                    print 'Select calib file: %s' % cf.get_path()
                return cf.get_path()

        # if no matching found
        return ''

#----------------------------------------------

if __name__ == "__main__" :

    # assuming /reg/d/psdm/CXI/cxid2714/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/pedestals/15-end.data

    #cdir  = '/reg/d/psdm/CXI/cxid2714/calib/'
    #cdir  = '/reg/d/psdm/CXI/cxi80410/calib/'
    cdir  = '/reg/d/psdm/CXI/cxi83714/calib/'

    group = 'CsPad::CalibV1'
    src   = 'CxiDs1.0:Cspad.0'
    type  = 'pedestals'
    #rnum  = 137
    rnum  = 123456789

    print 'Finding calib file for\n  dir = %s\n  grp = %s\n  src = %s\n  type= %s\n  run = %d' % \
          (cdir, group, src, type, rnum)

    cff = CalibFileFinder(cdir, group, 0377) # 0377)
    fname = cff.findCalibFile(src, type, rnum)

    sys.exit('End of %s' % sys.argv[0])

#----------------------------------------------
