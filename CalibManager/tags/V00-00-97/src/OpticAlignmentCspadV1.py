#!/usr/bin/env python

#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#------------------------------------------------------------------------
""" Processing of optical measurements for CXI-CSPAD (moving quads geometry)

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

class OpticAlignmentCspadV1 (OpticAlignmentCspadMethods) :
    """OpticAlignmentCspadV1"""

    sensor_rotation = [0,0,270,270,180,180,270,270]

    points_for_quadrants = [ [ 6, 2,14,10,18,22,30,26],
                             [ 6, 2,14,10,18,22,30,26],
                             [ 6, 2,14,10,18,22,30,26],
                             [ 6, 2,14,10,18,22,30,26] ]
    #Base index for 2x1:
    #        0  1   2  3   4   5   6   7
    ibase = [5, 1, 13, 9, 17, 21, 29, 25] 

    quad_n90_in_det = [0,0,0,0]

    pixelSize = 109.92

    def __init__(self, fname=None, path='calib-tmp', save_calib_files=True, print_bits=0377, plot_bits=0377, exp='Any', det='CSPAD-CXI'):
        """Constructor."""
        if print_bits &  1 : print 'Start OpticAlignmentCspadV1'

        if fname is not None : self.fname = fname
        else                 : self.fname = '/reg/neh/home1/dubrovin/LCLS/CSPadMetrologyProc/metrology_standard.txt'

        if not os.path.lexists(self.fname) : 
            if print_bits &  1 : print 'Non-available input file: ' + self.fname
            return

        self.path           = path
        self.save_calib_files = save_calib_files
        self.print_bits     = print_bits
        self.plot_bits      = plot_bits
        self.exp            = exp
        self.det            = det

        self.fname_center_um  = os.path.join(self.path, 'center_um-0-end.data')
        self.fname_center     = os.path.join(self.path, 'center-0-end.data')
        self.fname_tilt       = os.path.join(self.path, 'tilt-0-end.data')
        self.fname_geometry   = os.path.join(self.path, 'geometry-0-end.data')
 
        self.fname_plot_quads = os.path.join(self.path, 'metrology_standard_quads.png')
        self.fname_plot_det   = os.path.join(self.path, 'metrology_standard_det.png')

        self.read_optical_alignment_file()
        self.evaluate_deviation_from_flatness()
        self.evaluate_center_coordinates()
        #self.evaluate_length_width_angle_v0()
        self.evaluate_length_width_angle()

        self.present_results()

    #-------------------
    # methods --
    #-------------------

    def present_results(self): 

        if self.print_bits & 2 : print '\n' + self.txt_deviation_from_flatness()
        if self.print_bits & 4 : print '\nQuality check in XY plane:\n', self.txt_qc_table_xy() 
        if self.print_bits & 8 : print '\nQuality check in Z:\n', self.txt_qc_table_z()

        center_txt_um  = self.txt_center_um_formatted_array (format='%6i  ')
        center_txt_pix = self.txt_center_pix_formatted_array(format='%7.2f  ')
        tilt_txt       = self.txt_tilt_formatted_array(format='%8.5f  ')
        geometry_txt   = self.txt_geometry()
        
        if self.print_bits &  16 : print 'X, Y, and Z coordinates of the 2x1 CENTER (um):\n' + center_txt_um
        if self.print_bits &  32 : print '\nCalibration type "center" in pixels:\n' + center_txt_pix
        if self.print_bits &  64 : print '\nCalibration type "tilt" - degree:\n' + tilt_txt
        if self.print_bits & 128 : print '\nCalibration type "geometry"\n' + geometry_txt

        if self.save_calib_files :
            self.create_directory(self.path)
            self.save_text_file(self.fname_center_um, center_txt_um)
            self.save_text_file(self.fname_center, center_txt_pix)
            self.save_text_file(self.fname_tilt, tilt_txt)
            self.save_text_file(self.fname_geometry, geometry_txt)

        if self.plot_bits & 1 : self.drawOpticalAlignmentFile()
        if self.plot_bits & 2 : self.drawQuadsSeparately()


    def read_optical_alignment_file(self): 
        if self.print_bits & 256 : print 'read_optical_alignment_file()'

                                 # quad 0:3
                                   # point 1:32
                                      # record: point, X, Y, Z 0:3
        self.arr = numpy.zeros( (self.nquads, self.npoints+1, 4), dtype=numpy.int32 )

        file = open(self.fname, 'r')
        # Print out 7th entry in each line.
        for line in file:

            if len(line) == 1 : continue # ignore empty lines
            if self.print_bits & 256 : print len(line),  ' Line: ', line

            list_of_fields = line.split()

            if list_of_fields[0] == 'Quad' : # Treat quad header lines
                self.quad = int(list_of_fields[1])
                if self.print_bits & 256 : print 'Stuff for quad', self.quad  
                continue

            if list_of_fields[0] == 'Sensor' or list_of_fields[0] == 'Point' : # Treat the title lines
                if self.print_bits & 256 : print 'Comment line:', line  
                continue
            
            if len(list_of_fields) != 4 : # Ignore lines with non-expected number of fields
                if self.print_bits & 256 : print 'len(list_of_fields) =', len(list_of_fields),
                if self.print_bits & 256 : print 'RECORD IS IGNORED due to unexpected format of the line:',line
                continue              

            #point = int(list_of_fields[0])
            #X = int(list_of_fields[1])
            #Y = int(list_of_fields[2])
            #Z = int(list_of_fields[3])
            #Title = list_of_fields[4]
            
            factor = 1
            #factor = 1000
            point = int(list_of_fields[0])
            X = int(float(list_of_fields[1]) * factor)
            Y = int(float(list_of_fields[2]) * factor)
            Z = int(float(list_of_fields[3]) * factor)
            ##Title = list_of_fields[4]
            
            #record = [point, X, Y, Z, Title]
            if self.print_bits & 256 : print 'ACCEPT RECORD:', point, X, Y, Z #, Title

            self.arr[self.quad,point,0] = point
            self.arr[self.quad,point,1] = X
            self.arr[self.quad,point,2] = Y
            self.arr[self.quad,point,3] = Z

        file.close()


#----------------------------------

    def evaluate_length_width_angle_v0(self) :
        """ DEPRICATED in favor of OpticAlignmentCspadMethods.evaluate_length_width_angle()
        """

        self.S1  = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.S2  = numpy.zeros( (4,8), dtype=numpy.int32 )

        self.dS1 = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dS2 = numpy.zeros( (4,8), dtype=numpy.int32 )

        self.L1  = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.L2  = numpy.zeros( (4,8), dtype=numpy.int32 )

        self.dL1 = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dL2 = numpy.zeros( (4,8), dtype=numpy.int32 )

        self.dZS1 = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dZS2 = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dZL1 = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dZL2 = numpy.zeros( (4,8), dtype=numpy.int32 )

        self.tiltXYDegree = numpy.zeros( (4,8), dtype=numpy.float32 )
        self.tiltXZDegree = numpy.zeros( (4,8), dtype=numpy.float32 )
        self.tiltYZDegree = numpy.zeros( (4,8), dtype=numpy.float32 )

        self.D1  = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.D2  = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dD  = numpy.zeros( (4,8), dtype=numpy.int32 )

        self.ddS = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.ddL = numpy.zeros( (4,8), dtype=numpy.int32 )

        self.ddZS = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.ddZL = numpy.zeros( (4,8), dtype=numpy.int32 )

        self.dZSA = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dZLA = numpy.zeros( (4,8), dtype=numpy.int32 )

        self.SA  = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.LA  = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dSA = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dLA = numpy.zeros( (4,8), dtype=numpy.int32 )

        self.XSize = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.YSize = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dZX   = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.dZY   = numpy.zeros( (4,8), dtype=numpy.int32 )

        ix = 1
        iy = 2
        iz = 3

        for quad in range(4) :

            for segm in range(8) :

                icor1 = self.ibase[segm]   
                icor2 = self.ibase[segm] + 1
                icor3 = self.ibase[segm] + 2
                icor4 = self.ibase[segm] + 3

                if segm == 0 or  segm == 1 or  segm == 4 or  segm == 5 :
                    # for horizontal 2x1

                    self. S1[quad][segm]  = self.arr[quad,icor2,iy] - self.arr[quad,icor1,iy]
                    self. S2[quad][segm]  = self.arr[quad,icor3,iy] - self.arr[quad,icor4,iy]

                    self.dS1[quad][segm]  = self.arr[quad,icor4,iy] - self.arr[quad,icor1,iy]
                    self.dS2[quad][segm]  = self.arr[quad,icor3,iy] - self.arr[quad,icor2,iy]

                    self. L1[quad][segm]  = self.arr[quad,icor4,ix] - self.arr[quad,icor1,ix]
                    self. L2[quad][segm]  = self.arr[quad,icor3,ix] - self.arr[quad,icor2,ix]

                    self.dL1[quad][segm]  = self.arr[quad,icor2,ix] - self.arr[quad,icor1,ix]
                    self.dL2[quad][segm]  = self.arr[quad,icor3,ix] - self.arr[quad,icor4,ix]

                    self.dZS1[quad][segm] = self.arr[quad,icor2,iz] - self.arr[quad,icor1,iz]
                    self.dZS2[quad][segm] = self.arr[quad,icor3,iz] - self.arr[quad,icor4,iz]
                    self.dZL1[quad][segm] = self.arr[quad,icor4,iz] - self.arr[quad,icor1,iz]
                    self.dZL2[quad][segm] = self.arr[quad,icor3,iz] - self.arr[quad,icor2,iz]

                    self.evaluateSLAverage(quad,segm)

                    self.XSize[quad][segm] = self.LA[quad][segm]
                    self.YSize[quad][segm] = self.SA[quad][segm]
                    self.dZX  [quad][segm] = self.dZLA[quad][segm] 
                    self.dZY  [quad][segm] = self.dZSA[quad][segm]  

                else:
                    # for vertical 2x1

                    self. S1[quad][segm]  =   self.arr[quad,icor4,ix] - self.arr[quad,icor1,ix]
                    self. S2[quad][segm]  =   self.arr[quad,icor3,ix] - self.arr[quad,icor2,ix]
                                                                                           
                    self.dS1[quad][segm]  = -(self.arr[quad,icor2,ix] - self.arr[quad,icor1,ix]) # sign is chosen 
                    self.dS2[quad][segm]  = -(self.arr[quad,icor3,ix] - self.arr[quad,icor4,ix]) # for positive phi

                    self. L1[quad][segm]  =   self.arr[quad,icor2,iy] - self.arr[quad,icor1,iy]
                    self. L2[quad][segm]  =   self.arr[quad,icor3,iy] - self.arr[quad,icor4,iy]
                                                                                           
                    self.dL1[quad][segm]  =   self.arr[quad,icor4,iy] - self.arr[quad,icor1,iy]
                    self.dL2[quad][segm]  =   self.arr[quad,icor3,iy] - self.arr[quad,icor2,iy]

                    self.dZS1[quad][segm] =   self.arr[quad,icor4,iz] - self.arr[quad,icor1,iz]
                    self.dZS2[quad][segm] =   self.arr[quad,icor3,iz] - self.arr[quad,icor2,iz]
                    self.dZL1[quad][segm] =   self.arr[quad,icor2,iz] - self.arr[quad,icor1,iz]
                    self.dZL2[quad][segm] =   self.arr[quad,icor3,iz] - self.arr[quad,icor4,iz]

                    self.evaluateSLAverage(quad,segm)

                    self.XSize[quad][segm] = self.SA[quad][segm]
                    self.YSize[quad][segm] = self.LA[quad][segm]
                    self.dZX  [quad][segm] = self.dZSA[quad][segm] 
                    self.dZY  [quad][segm] = self.dZLA[quad][segm]  

                diag1x = float(self.arr[quad,icor1,ix] - self.arr[quad,icor3,ix])
                diag2x = float(self.arr[quad,icor2,ix] - self.arr[quad,icor4,ix])
                diag1y = float(self.arr[quad,icor1,iy] - self.arr[quad,icor3,iy])
                diag2y = float(self.arr[quad,icor2,iy] - self.arr[quad,icor4,iy])

                self.D1[quad][segm] = int( math.sqrt(diag1x*diag1x + diag1y*diag1y) )
                self.D2[quad][segm] = int( math.sqrt(diag2x*diag2x + diag2y*diag2y) )
                self.dD[quad][segm] = self.D1[quad][segm] - self.D2[quad][segm]

                self.ddS[quad][segm] = self.dS1[quad][segm] - self.dS2[quad][segm]
                self.ddL[quad][segm] = self.dL1[quad][segm] - self.dL2[quad][segm]

                self.ddZS[quad][segm] = self.dZS1[quad][segm] - self.dZS2[quad][segm]
                self.ddZL[quad][segm] = self.dZL1[quad][segm] - self.dZL2[quad][segm]

                #ang1 = ang2 = 0
                #if self.L1[quad][segm] != 0 : ang1 = float(self.dS1[quad][segm]) / self.L1[quad][segm]
                #if self.L2[quad][segm] != 0 : ang2 = float(self.dS2[quad][segm]) / self.L2[quad][segm]
                #angXY = (ang1 + ang2) * 0.5

                tiltXY = float(self.dSA[quad][segm]) / self.LA[quad][segm]    if self.LA[quad][segm]    != 0 else 0
                tiltXZ = float(self.dZX[quad][segm]) / self.XSize[quad][segm] if self.XSize[quad][segm] != 0 else 0
                tiltYZ = float(self.dZY[quad][segm]) / self.YSize[quad][segm] if self.YSize[quad][segm] != 0 else 0

                self.tiltXYDegree[quad][segm] = self.rad_to_deg * tiltXY
                self.tiltXZDegree[quad][segm] = self.rad_to_deg * tiltXZ
                self.tiltYZDegree[quad][segm] = self.rad_to_deg * tiltYZ


#----------------------------------
 
    def txt_geometry_quads(self) :
        txt = ''        
        name_segm   = 'QUAD:V1'
        name_parent = 'CSPAD:V1'
        num_parent, x0, y0, z0, rotXZ, rotYZ, tiltXY, tiltXZ, tiltYZ = 0,0,0,0,0,0,0,0,0
        q_rot = [90,0,270,180]
        q_x0  = [-4500,-4500, 4500, 4500]
        q_y0  = [-4500, 4500, 4500,-4500]
        for quad in range(self.nquads) :
            txt += self.str_fmt() % \
                (name_parent.ljust(12), num_parent, name_segm.ljust(12), quad, \
                 q_x0[quad], q_y0[quad], z0, q_rot[quad], rotXZ, rotYZ, tiltXY, tiltXZ, tiltYZ)
        return txt + '\n' 

#----------------------------------
 
    def txt_geometry_det_ip(self) :
        txt = ''
        name_object = 'CSPAD:V1'
        name_parent = 'RAIL'
        num_parent, num_object, x0, y0, z0, rotXY, rotXZ, rotYZ, tiltXY, tiltXZ, tiltYZ = 0,0,0,0,1e6,0,0,0,0,0,0
        txt += self.str_fmt() % \
            (name_parent.ljust(12), num_parent, name_object.ljust(12), num_object, \
            x0, y0, z0, rotXY, rotXZ, rotYZ, tiltXY, tiltXZ, tiltYZ)

        name_object = 'RAIL'
        name_parent = 'IP'
        z0          = 0
        txt += self.str_fmt() % \
            (name_parent.ljust(12), num_parent, name_object.ljust(12), num_object, \
            x0, y0, z0, rotXY, rotXZ, rotYZ, tiltXY, tiltXZ, tiltYZ)

        return txt + '\n' 

#----------------------------------
 
    def txt_geometry(self) :
        return self.txt_geometry_header() + \
               self.txt_geometry_segments() + \
               self.txt_geometry_quads() + \
               self.txt_geometry_det_ip()

#----------------------------------

def main():

    #fname = '2011-03-29-CSPAD2-Alignment-PostRun3.txt'
    #fname = '2011-06-20-CSPAD2-Alignment-Before-Run4.txt'
    #fname = '2011-08-10-Metrology.txt'
    #fname = '2011-08-DD-Run4-DSD-Metrology.txt'
    #fname = '2012-01-10-Run5-DSD-Metrology.txt'
    #fname = '2012-01-12-Run5-DSD-Metrology-corrected.txt'
    #fname = '2012-11-08-Run6-DSD-Metrology-standard.txt'
    #fname = 'metrology_renumerated.txt'
    #fname = 'metrology_standard.txt'
    # New life
    #fname = '2013-12-12-CSPAD-CXI-DSD-Metrology.txt'
    #fname = '2013-12-20-CSPAD-CXI-DS1-Metrology-corr.txt'
    #fname = '2014-03-19-CSPAD-CXI-DS1-Metrology-corr.txt'
    fname = '2014-05-15-CSPAD-CXI-DS1-Metrology-corr.txt'
    #fname = '2014-05-15-CSPAD-CXI-DS2-Metrology.txt'

    base_dir = '/reg/neh/home1/dubrovin/LCLS/CSPadMetrologyProc/'

    (opts, args) = input_option_parser(base_dir, fname)
    path_metrol = os.path.join(opts.dir, opts.fname)

    OpticAlignmentCspadV1(path_metrol, print_bits=opts.pbits, plot_bits=opts.gbits)
    sys.exit()

if __name__ == '__main__':
    main()

#----------------------------------
