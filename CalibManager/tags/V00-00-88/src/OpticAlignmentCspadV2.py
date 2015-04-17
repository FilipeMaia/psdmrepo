#!/usr/bin/env python

#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#------------------------------------------------------------------------
""" Processing of optical measurements for XPP-CSPAD (fixed geometry)

@see OpticAlignmentCspadMethods.py

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
__version__ = "$Revision$"
# $Source$
#----------------------------------
#import os
#import sys
#import numpy
#import numpy as np
#import math
#from time import localtime, gmtime, strftime, clock, time, sleep

#import matplotlib.pyplot as plt
#import matplotlib.lines  as lines

from CalibManager.OpticAlignmentCspadMethods import *
#from OpticAlignmentCspadMethods import *

#----------------------------------
#  Numeration of quads in the metrology file should be consistent with numeration in DAQ.
#  Orientation of CSPAD for
#
#  n90=0,          n90=1, etc.
#  ^               ^           
#  | Q0  Q1        | Q1  Q2    
#  |               |           
#  | Q3  Q2        | Q0  Q3    
#  +-------->      +-------->  
#
#----------------------------------

class OpticAlignmentCspadV2 (OpticAlignmentCspadMethods) :
    """OpticAlignmentCspadV2"""

    # Numeration of 2x1 corners in XPP-CSPAD optical measurements depending on quad orientation
    quad_r090 = [0,  32,29,30,31,  28,25,26,27,      4,1,2,3,      8,5,6,7,  16,13,14,15,   12,9,10,11,  20,17,18,19,  24,21,22,23]
    quad_r000 = [0,      1,2,3,4,      5,6,7,8,   9,10,11,12,  13,14,15,16,  17,18,19,20,  21,22,23,24,  25,26,27,28,  29,30,31,32]
    quad_r270 = [0,   10,11,12,9,  14,15,16,13,  22,23,24,21,  18,19,20,17,  26,27,28,25,  30,31,32,29,      6,7,8,5,      2,3,4,1]
    quad_r180 = [0,  23,24,21,22,  19,20,17,18,  31,32,29,30,  27,28,25,26,      7,8,5,6,      3,4,1,2,  15,16,13,14,   11,12,9,10]


    quad_n90_in_det = [1,0,3,2]

    def __init__(self, fname=None, path='calib-tmp', save_calib_files=True, print_bits=07777, plot_bits=0377, exp='Any', det='CSPAD-XPP', n90=0):
        """Constructor."""

        if print_bits &  1 : print 'Start OpticAlignmentCspadV2'

        if fname is not None : self.fname = fname
        else                 : self.fname = '/reg/neh/home1/dubrovin/LCLS/CSPadMetrologyProc/metrology_standard.txt'

        if not os.path.lexists(self.fname) : 
            if print_bits &  1 : print 'Non-available input file: ' + self.fname
            return

        self.path             = path
        self.save_calib_files = save_calib_files
        self.print_bits       = print_bits
        self.plot_bits        = plot_bits
        self.exp              = exp
        self.det              = det

        self.fname_center_um  = os.path.join(self.path, 'center_global_um-0-end.data')
        self.fname_center     = os.path.join(self.path, 'center_global-0-end.data')
        self.fname_tilt       = os.path.join(self.path, 'tilt-0-end.data')
        self.fname_geometry   = os.path.join(self.path, 'geometry-0-end.data')

        self.fname_plot_det   = os.path.join(self.path, 'metrology_standard_det.png')

        self.readOpticalAlignmentFile()
        self.changeNumerationToQuadsV1(n90)
        self.evaluate_deviation_from_flatness()
        self.evaluate_center_coordinates()
        self.evaluate_length_width_angle(n90)

        self.present_results()


#----------------------------------

    def present_results(self): 

        if self.print_bits & 2 : print '\n' + self.txt_deviation_from_flatness()
        if self.print_bits & 4 : print '\nQuality check in XY plane:\n', self.txt_qc_table_xy() 
        if self.print_bits & 8 : print '\nQuality check in Z:\n', self.txt_qc_table_z()

        center_txt_um  = self.txt_center_um_formatted_array (format='%6i  ')
        center_txt_pix = self.txt_center_pix_formatted_array(format='%7.2f  ')
        tilt_txt       = self.txt_tilt_formatted_array(format='%8.5f  ')
        geometry_txt   = self.txt_geometry()

        if self.print_bits &  16 : print 'X, Y, and Z coordinates of the 2x1 center_global (um):\n' + center_txt_um
        if self.print_bits &  32 : print '\nCalibration type "center_global" in pixels:\n' + center_txt_pix
        if self.print_bits &  64 : print '\nCalibration type "tilt" - degree:\n' + tilt_txt
        if self.print_bits & 128 : print '\nCalibration type "geometry"\n%s' % geometry_txt

        if self.save_calib_files :
            self.create_directory(self.path)
            self.save_text_file(self.fname_center_um, center_txt_um)
            self.save_text_file(self.fname_center, center_txt_pix)
            self.save_text_file(self.fname_tilt, tilt_txt)
            self.save_text_file(self.fname_geometry, geometry_txt)

        if self.plot_bits & 1 :
            self.arr = self.arr_opt
            print 'Draw array from metrology file'
            self.drawOpticalAlignmentFile()

        if self.plot_bits & 2 :
            self.arr = self.arr_renum
            print 'Draw array with re-numerated points for quads'
            self.drawOpticalAlignmentFile()

#----------------------------------

    def readOpticalAlignmentFile(self): 
        """Reads the metrology.txt file with original optical measurements.
           The numereation of points is changed since 2012-02-26.
        """
        if self.print_bits & 1 : print 'readOpticalAlignmentFile()'

                                 # quad 0:3
                                   # point 1:32
                                      # record: point, X, Y, Z 0:3
        self.arr_opt = numpy.zeros( (self.nquads, self.npoints+1, 4), dtype=numpy.int32 )

        #infile = './2012-01-12-Run5-DSD-Metrology-corrected.txt'
        file = open(self.fname, 'r')
        # Print out 7th entry in each line.
        for linef in file:

            line = linef.strip('\n')

            #if len(line) == 1 : continue # ignore empty lines
            #print len(line),  ' Line: ', line
            if not line : continue   # discard empty strings

            list_of_fields = line.split()

            if list_of_fields[0] == 'Quad' : # Treat quad header lines
                self.quad = int(list_of_fields[1])
                if self.print_bits & 256 : print 'Stuff for quad', self.quad  
                continue

            if list_of_fields[0] == 'Sensor' : # Treat the title lines
                if self.print_bits & 256 : print 'Comment line:', line  
                continue
            
            if len(list_of_fields) != 4 : # Ignore lines with non-expected number of fields
                if self.print_bits & 256 : print 'len(list_of_fields) =', len(list_of_fields),
                if self.print_bits & 256 : print 'RECORD IS IGNORED due to unexpected format of the line:',line
                continue              

            point, X, Y, Z = [int(v) for v in list_of_fields]
            
            #record = [point, X, Y, Z, Title]
            if self.print_bits & 256 : print 'ACCEPT RECORD:', point, X, Y, Z #, Title

            self.arr_opt[self.quad,point,:] = [point, X, Y, Z]

        file.close()

        if self.print_bits & 256 : print 'Array of alignment info:\n', self.arr_opt


#----------------------------------

    def arrNumTransformationForN90(self, n90=0): 
        
        if   n90 == 0 : return np.array([self.quad_r090, self.quad_r000, self.quad_r270, self.quad_r180])
        elif n90 == 1 : return np.array([self.quad_r000, self.quad_r270, self.quad_r180, self.quad_r090])
        elif n90 == 2 : return np.array([self.quad_r270, self.quad_r180, self.quad_r090, self.quad_r000]) 
        elif n90 == 3 : return np.array([self.quad_r180, self.quad_r090, self.quad_r000, self.quad_r270])
        else        :
            print 'arrNumTransformationForN90: WRONG n90=%d, use n90 = 0, 1, 2, or 3' % n90
            sys.exit('Exit on warning')


#----------------------------------

    def changeNumerationToQuadsV1(self, n90=0): 
        """The numereation of points is changed since 2012-02-26.
           Bring the numeration of points to old-standard.
        """
        if self.print_bits & 256 : print 'changeNumerationToQuadsV1()'

                                       # quad 0:3
                                         # point 1:32
                                            # record: point, X, Y, Z 0:3
        self.arr_renum = numpy.zeros( (self.nquads, self.npoints+1, 4), dtype=numpy.int32 )

        # Table of new in positions of old-standard
        num_conv = self.arrNumTransformationForN90(n90)

        for quad in range(self.nquads) :
            for point in range(1,self.npoints+1,1) :
                #print 'quad, point=', quad, point
                point_optical = num_conv[quad][point] # INDEXES for a python list
                self.arr_renum[quad,point,...] = self.arr_opt[quad,point_optical,...]
                self.arr_renum[quad,point,0]   = point # change the point number from optical to standard

        if self.print_bits & 256 : print 'Array with standard (per quad) numeration of points:\n', self.arr_renum

        self.arr = self.arr_renum


#----------------------------------

    def txt_geometry_segments(self) :
        txt = ''        
        name_segm   = 'SENS2X1:V1'
        segm_index  = -1
        name_parent = 'CSPAD:V2'
        name_index  = 0
        rotXZ, rotYZ = 0,0
        for quad in range(self.nquads) :
            for segm in range(self.nsegms) :
                segm_index += 1 
                txt += self.str_fmt() % \
                       (name_parent.ljust(12), name_index, name_segm.ljust(12), segm_index, \
                       self.arrXmu[quad][segm], \
                       self.arrYmu[quad][segm], \
                       self.arrZmu[quad][segm], \
                       self.rotXYDegree[quad][segm], \
                       rotXZ, \
                       rotYZ, \
                       self.tiltXYDegree[quad][segm], \
                       self.tiltXZDegree[quad][segm], \
                       self.tiltYZDegree[quad][segm])
            txt += '\n' 
        return txt

#----------------------------------
 
    def txt_geometry_det_ip(self) :
        txt = ''
        name_object = 'CSPAD:V2'
        name_parent = 'IP'
        num_parent, num_object, x0, y0, z0, rotXY, rotXZ, rotYZ, tiltXY, tiltXZ, tiltYZ = 0,0,0,0,1e6,0,0,0,0,0,0
        txt += self.str_fmt() % \
            (name_parent.ljust(12), num_parent, name_object.ljust(12), num_object, \
            x0, y0, z0, rotXY, rotXZ, rotYZ, tiltXY, tiltXZ, tiltYZ)

        return txt + '\n' 


#----------------------------------
 
    def txt_geometry(self) :
        return self.txt_geometry_header() + \
               self.txt_geometry_segments() + \
               self.txt_geometry_det_ip()

#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------

def main():

    #fname = '2012-02-26-CSPAD-XPP-Metrology.txt'
    #fname = '2013-01-24-CSPAD-XPP-Metrology.txt'
    #fname = '2013-01-29-CSPAD-XPP-Metrology.txt'      wrong numeration of quads
    #fname = '2013-01-29-CSPAD-XPP-Metrology-corr.txt' wrong numeration of quads
    fname = '2013-10-09-CSPAD-XPP-Metrology.txt'

    base_dir = '/reg/neh/home1/dubrovin/LCLS/CSPadMetrologyProc/'

    (opts, args) = input_option_parser(base_dir, fname)
    path_metrol = os.path.join(opts.dir, opts.fname)

    OpticAlignmentCspadV2(path_metrol, print_bits=opts.pbits, plot_bits=opts.gbits, n90=0)
    sys.exit()

if __name__ == '__main__':
    main()

#----------------------------------
