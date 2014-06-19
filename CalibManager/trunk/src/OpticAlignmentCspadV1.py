#!/usr/bin/env python

#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Template...
#
#------------------------------------------------------------------------
"""
@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$


#----------------------------------
import os
import sys
import numpy
import numpy as np
import math
from time import localtime, gmtime, strftime, clock, time, sleep

#from PyQt4 import QtGui, QtCore

import matplotlib.pyplot as plt
import matplotlib.lines  as lines

#----------------------------------

class OpticAlignmentCspadV1 :
    """OpticAlignmentCspadV1"""

    sensor_rotation = [0,0,270,270,180,180,270,270]

    points_for_quadrants = [ [ 6, 2,14,10,18,22,30,26],
                             [ 6, 2,14,10,18,22,30,26],
                             [ 6, 2,14,10,18,22,30,26],
                             [ 6, 2,14,10,18,22,30,26] ]
    #Base index for 2x1:
    #        0  1   2  3   4   5   6   7
    ibase = [5, 1, 13, 9, 17, 21, 29, 25] 

    pixelSize = 109.92

    def __init__(self, fname=None, path='calib-tmp', save_calib_files=True, print_bits=0377, plot_bits=0377, exp='', det=''):
        """Constructor."""
        if print_bits &  1 : print 'Start OpticAlignmentCspadV1'

        if fname != None : self.fname = fname
        else             : self.fname = '/reg/neh/home1/dubrovin/LCLS/CSPadMetrologyProc/metrology_standard.txt'

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
        if self.print_bits & 1 : print 'read_optical_alignment_file()'

                                 # quad 0:3
                                   # point 1:32
                                      # record: point, X, Y, Z 0:3
        self.arr = numpy.zeros( (4,33,4), dtype=numpy.int32 )

        file = open(self.fname, 'r')
        # Print out 7th entry in each line.
        for line in file:

            if len(line) == 1 : continue # ignore empty lines
            if self.print_bits & 1 : print len(line),  ' Line: ', line

            list_of_fields = line.split()

            if list_of_fields[0] == 'Quad' : # Treat quad header lines
                self.quad = int(list_of_fields[1])
                if self.print_bits & 1 : print 'Stuff for quad', self.quad  
                continue

            if list_of_fields[0] == 'Sensor' or list_of_fields[0] == 'Point' : # Treat the title lines
                if self.print_bits & 1 : print 'Comment line:', line  
                continue
            
            if len(list_of_fields) != 4 : # Ignore lines with non-expected number of fields
                if self.print_bits & 1 : print 'len(list_of_fields) =', len(list_of_fields),
                if self.print_bits & 1 : print 'RECORD IS IGNORED due to unexpected format of the line:',line
                continue              

            point = int(list_of_fields[0])
            X = int(list_of_fields[1])
            Y = int(list_of_fields[2])
            Z = int(list_of_fields[3])
            #Title = list_of_fields[4]
            
            #factor = 1000
            #point = int(list_of_fields[0])
            #X = int(float(list_of_fields[1]) * factor)
            #Y = int(float(list_of_fields[2]) * factor)
            #Z = int(float(list_of_fields[3]) * factor)
            ##Title = list_of_fields[4]
            
            #record = [point, X, Y, Z, Title]
            if self.print_bits & 1 : print 'ACCEPT RECORD:', point, X, Y, Z #, Title

            self.arr[self.quad,point,0] = point
            self.arr[self.quad,point,1] = X
            self.arr[self.quad,point,2] = Y
            self.arr[self.quad,point,3] = Z

        file.close()


    def evaluate_deviation_from_flatness(self) :

        ix = 1
        iy = 2
        iz = 3

        self.arr_dev_um = numpy.zeros( (4,8), dtype=numpy.double )

        for quad in range(4) :
           for segm in range(8) :

               icor1 = self.ibase[segm]   
               icor2 = self.ibase[segm] + 1
               icor3 = self.ibase[segm] + 2
               icor4 = self.ibase[segm] + 3

               v21 = ( self.arr[quad,icor2,ix] - self.arr[quad,icor1,ix],
                       self.arr[quad,icor2,iy] - self.arr[quad,icor1,iy],
                       self.arr[quad,icor2,iz] - self.arr[quad,icor1,iz] )

               v31 = ( self.arr[quad,icor3,ix] - self.arr[quad,icor1,ix],
                       self.arr[quad,icor3,iy] - self.arr[quad,icor1,iy],
                       self.arr[quad,icor3,iz] - self.arr[quad,icor1,iz] )

               v41 = ( self.arr[quad,icor4,ix] - self.arr[quad,icor1,ix],
                       self.arr[quad,icor4,iy] - self.arr[quad,icor1,iy],
                       self.arr[quad,icor4,iz] - self.arr[quad,icor1,iz] )

               #print v21, v31, v41, 

               vort = np.array(np.cross(v21, v41), dtype=np.double) # vort = [v21 x v41]        - vector product
               norm = math.sqrt(np.sum(vort*vort))                  # norm = |vort|             - length of the vector vort
               vort_norm = vort / norm                              # vort_norm = vort / |vort| - normalized vector orthogonal to the plane
               dev = np.sum(v31*vort_norm)                          # dev = (v31 * vort_norm)   - scalar product

               self.arr_dev_um[quad,segm] = dev

               #print '  vort_norm=', vort_norm, '  norm =', norm, '  dev =', dev
               #print '  vort_norm=', vort_norm, '  norm =', norm, '  dev =', dev
               #print 'quad:%1d, segm:%2d,  dz3[um]: %8.3f\n' % (quad, segm, self.arr_dev_um[quad,segm])


    def txt_deviation_from_flatness(self) :
        txt = 'Deviation from flatness [um] for segments:\n'
        for quad in range(4) :
           for segm in range(8) :
               txt += 'quad:%1d, segm:%2d,  dz3[um]: %8.3f\n' % (quad, segm, self.arr_dev_um[quad,segm])
        txt += 'Mean and Standard deviation for dz3[um]: %8.3f +- %8.3f\n' % (np.mean(self.arr_dev_um), np.std(self.arr_dev_um))
        return txt


    def evaluate_center_coordinates(self) :

        self.arrXmu = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.arrYmu = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.arrZmu = numpy.zeros( (4,8), dtype=numpy.int32 )
        self.arrX   = numpy.zeros( (4,8), dtype=numpy.float )
        self.arrY   = numpy.zeros( (4,8), dtype=numpy.float )
        self.arrZ   = numpy.zeros( (4,8), dtype=numpy.float )

        ix = 1
        iy = 2
        iz = 3

        for quad in range(4) :
            for segm in range(8) :

                icor1 = self.ibase[segm]   
                icor2 = self.ibase[segm] + 1
                icor3 = self.ibase[segm] + 2
                icor4 = self.ibase[segm] + 3

                X = 0.25 * ( self.arr[quad,icor1,ix]
                           + self.arr[quad,icor2,ix]
                           + self.arr[quad,icor3,ix]
                           + self.arr[quad,icor4,ix] )

                Y = 0.25 * ( self.arr[quad,icor1,iy]
                           + self.arr[quad,icor2,iy]
                           + self.arr[quad,icor3,iy]
                           + self.arr[quad,icor4,iy] ) 

                Z = 0.25 * ( self.arr[quad,icor1,iz]
                           + self.arr[quad,icor2,iz]
                           + self.arr[quad,icor3,iz]
                           + self.arr[quad,icor4,iz] ) 

                Xmy, Ymy, Zmy = X, Y, Z

                #print 'quad:%1d, segm:%2d,  X:%7d  Y:%7d, Z:%3d' % (quad, segm, Xmy, Ymy, Zmy)

                self.arrXmu[quad][segm] = Xmy
                self.arrYmu[quad][segm] = Ymy
                self.arrZmu[quad][segm] = Zmy

                self.arrX[quad][segm] = float(Xmy) / self.pixelSize
                self.arrY[quad][segm] = float(Ymy) / self.pixelSize
                self.arrZ[quad][segm] = float(Zmy) / self.pixelSize

#----------------------------------

    def txt_center_um_formatted_array(self, format='%6i  ') :
        return self.get_formatted_array(self.arrXmu, format=format)+'\n' \
             + self.get_formatted_array(self.arrYmu, format=format)+'\n' \
             + self.get_formatted_array(self.arrZmu, format=format)

#----------------------------------

    def txt_center_pix_formatted_array(self, format='%7.2f  ') :
        return self.get_formatted_array(self.arrX, format=format)+'\n' \
             + self.get_formatted_array(self.arrY, format=format)+'\n' \
             + self.get_formatted_array(self.arrZ, format=format)

#----------------------------------

    def txt_tilt_formatted_array(self, format='%8.5f  ') :
        return self.get_formatted_array(self.tiltXYDegree, format)

#----------------------------------

    def get_formatted_array(self, arr, format='%7.2f,') :
        txt = ''
        for row in range(4) :
            for col in range(8) :
                txt += format % (arr[row][col])
            txt += '\n'
        return txt

#----------------------------------

    def create_directory(self, dir) : 
        if os.path.exists(dir) :
            if self.print_bits & 1 : print 'Directory exists:' + dir
        else :
            os.makedirs(dir)
            if self.print_bits & 1 : print 'Directory created:' + dir 

#----------------------------------

    def save_text_file(self, fname, text) :
        if self.print_bits & 256 : print 'Save text file: ' + fname
        f=open(fname,'w')
        f.write( text )
        f.close() 

#----------------------------------

    def evaluate_length_width_angle(self) :

        rad_to_deg = 180/3.1415927

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

                self.tiltXYDegree[quad][segm] = rad_to_deg * tiltXY
                self.tiltXZDegree[quad][segm] = rad_to_deg * tiltXZ
                self.tiltYZDegree[quad][segm] = rad_to_deg * tiltYZ


#----------------------------------

    def evaluateSLAverage(self,quad,segm) :
        self.dZSA[quad][segm] = 0.5 * (self.dZS1[quad][segm] + self.dZS2[quad][segm])
        self.dZLA[quad][segm] = 0.5 * (self.dZL1[quad][segm] + self.dZL2[quad][segm])
        self.SA[quad][segm]   = 0.5 * (self.S1[quad][segm]   + self.S2[quad][segm])
        self.LA[quad][segm]   = 0.5 * (self.L1[quad][segm]   + self.L2[quad][segm])
        self.dSA[quad][segm]  = 0.5 * (self.dS1[quad][segm]  + self.dS2[quad][segm])
        self.dLA[quad][segm]  = 0.5 * (self.dL1[quad][segm]  + self.dL2[quad][segm])

#----------------------------------

    def txt_qc_table_xy(self) :
        txt = 'segm:        S1      S2     dS1     dS2        L1      L2     dL1     dL2    angle(deg)   D1      D2      dD   d(dS)   d(dL)'
        for quad in range(4) :
            txt += '\nQuad  %d\n' % quad
            for segm in range(8) :
                txt += 'segm: %1d  %6d  %6d  %6d  %6d    %6d  %6d  %6d  %6d   %8.5f  %6d  %6d  %6d  %6d  %6d\n' % \
                    (segm, self.S1[quad][segm], self.S2[quad][segm], self.dS1[quad][segm], self.dS2[quad][segm], \
                           self.L1[quad][segm], self.L2[quad][segm], self.dL1[quad][segm], self.dL2[quad][segm], \
                           self.tiltXYDegree[quad][segm], \
                           self.D1[quad][segm], self.D2[quad][segm], self.dD[quad][segm], self.ddS[quad][segm], self.ddL[quad][segm] )
        return txt

#----------------------------------

    def txt_qc_table_z(self) :
        txt = 'segm:        SA      LA   XSize   YSize    dZS1  dZS2  dZL1  dZL2    dZSA  dZLA  ddZS  ddZL     dZX   dZY   angXZ(deg) angYZ(deg) dz3(um)'
        for quad in range(4) :
            txt += '\nQuad  %d\n' % quad
            for segm in range(8) :
                txt += 'segm: %1d  %6d  %6d  %6d  %6d   %5d %5d %5d %5d   %5d %5d %5d %5d   %5d %5d  %8.5f   %8.5f   %8.3f\n' % \
                    (segm, self.SA[quad][segm],   self.LA[quad][segm],   self.XSize[quad][segm], self.YSize[quad][segm], \
                           self.dZS1[quad][segm], self.dZS2[quad][segm], self.dZL1[quad][segm],  self.dZL2[quad][segm], \
                           self.dZSA[quad][segm], self.dZLA[quad][segm], self.ddZS[quad][segm],  self.ddZL[quad][segm], \
                           self.dZX[quad][segm],  self.dZY[quad][segm],  self.tiltXZDegree[quad][segm], self.tiltYZDegree[quad][segm], \
                           self.arr_dev_um[quad,segm])
        return txt #+'\n' 

#----------------------------------
 
    def txt_geometry_header(self) :
        txt = '# TITLE      Geometry parameters of CSPAD with moving quads' \
            + '\n# CREATED    %s' % strftime('%Y-%m-%d %H:%M:%S %Z', localtime()) \
            + '\n# CREATOR    %s' % os.environ['LOGNAME'] \
            + '\n# EXPERIMENT %s' % self.exp \
            + '\n# DETECTOR   %s' % self.det \
            + '\n# CALIB_TYPE geometry' \
            + '\n# COMMENT                Table contains the list of geometry parameters for alignment of 2x1 sensors in quads and quads in CSPAD' \
            + '\n# PARAMETER PARENT     - name and version of the parent object; all translation and rotation pars are defined w.r.t. parent Cartesian frame' \
            + '\n# PARAMETER PARENT_IND - index of the parent object' \
            + '\n# PARAMETER OBJECT     - name and version of the new object' \
            + '\n# PARAMETER OBJECT_IND - index of the new object' \
            + '\n# PARAMETER X0         - x-coordinate [um] of the new object origin in the parent frame' \
            + '\n# PARAMETER Y0         - y-coordinate [um] of the new object origin in the parent frame' \
            + '\n# PARAMETER Z0         - z-coordinate [um] of the new object origin in the parent frame' \
            + '\n# PARAMETER ROTATION_Z - new object design rotation angle [deg] around Z axis of the parent frame' \
            + '\n# PARAMETER ROTATION_Y - new object design rotation angle [deg] around Y axis of the parent frame' \
            + '\n# PARAMETER ROTATION_X - new object design rotation angle [deg] around X axis of the parent frame' \
            + '\n# PARAMETER TILT_Z     - new object tilt angle [deg] around Z axis of the parent frame' \
            + '\n# PARAMETER TILT_Y     - new object tilt angle [deg] around Y axis of the parent frame' \
            + '\n# PARAMETER TILT_X     - new object tilt angle [deg] around X axis of the parent frame' \
            + '\n\n# PARENT   IND  OBJECT     IND    X0[um]  Y0[um]  Z0[um]   ROT-Z ROT-Y ROT-X     TILT-Z   TILT-Y   TILT-X'
        return txt + '\n\n'

#----------------------------------
 
    def txt_geometry_segments(self) :
        txt = ''        
        name_segm   = 'SENS2X1:V1'
        name_parent = 'QUAD:V1'
        rotXZ, rotYZ = 0,0
        for quad in range(4) :
            for segm in range(8) :
                txt += '%s %3d  %s %3d   %7d %7d %7d   %5d %5d %5d   %8.5f %8.5f %8.5f \n' % \
                       (name_parent.ljust(10), quad, name_segm.ljust(10), segm, \
                       self.arrXmu[quad][segm], \
                       self.arrYmu[quad][segm], \
                       self.arrZmu[quad][segm], \
                       self.sensor_rotation[segm], \
                       rotXZ, \
                       rotYZ, \
                       self.tiltXYDegree[quad][segm], \
                       self.tiltXZDegree[quad][segm], \
                       self.tiltYZDegree[quad][segm])
            txt += '\n' 
        return txt

#----------------------------------
 
    def txt_geometry_quads(self) :
        txt = ''        
        name_segm   = 'QUAD:V1'
        name_parent = 'CSPAD:V1'
        num_parent, x0, y0, z0, rotXZ, rotYZ, tiltXY, tiltXZ, tiltYZ = 0,0,0,0,0,0,0,0,0
        quad_rotation = [270,0,90,180]
        for quad in range(4) :
            txt += '%s %3d  %s %3d   %7d %7d %7d   %5d %5d %5d   %8.5f %8.5f %8.5f \n' % \
                (name_parent.ljust(10), num_parent, name_segm.ljust(10), quad, x0, y0, z0, quad_rotation[quad], rotXZ, rotYZ, tiltXY, tiltXZ, tiltYZ)
        return txt + '\n' 

#----------------------------------
 
    def txt_geometry(self) :
        return self.txt_geometry_header() + \
               self.txt_geometry_segments() + \
               self.txt_geometry_quads()

#----------------------------------

    def drawOpticalAlignmentFile(self): 
        print 'drawOpticalAlignmentFile()'

        sizex, sizey = shape = (100,100)
        #arr   = np.arange(sizex*sizey)
        #arr.shape = shape
        #arr   = np.zeros(shape)
        fig   = plt.figure(figsize=(10,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)
        axes  = fig.add_subplot(111)        
        axes.set_xlim((-50,1750))
        axes.set_ylim((-50,1750))
        #axes1 = plt.imshow(arr, origin='lower', interpolation='nearest',extent=ax_range) 

        for quad in range(4) :
            #print '\nQuad:', quad
            self.drawOneQuad(quad,axes)

        plt.show()
        fig.savefig(self.fname_plot_det)
        print 'Image saved in file:', self.fname_plot_det


    def drawOneQuad(self,quad,axes):
        print 'drawOneQuad(' + str(quad) + ')'

        line_point = 0
        self.xlp = [0,0,0,0,0]
        self.ylp = [0,0,0,0,0]
        for point in range(1,33) :
            N = self.arr[quad,point,0]
            X = self.arr[quad,point,1]
            Y = self.arr[quad,point,2]
            Z = self.arr[quad,point,3]                
            #print 'N,X,Y =', N,X,Y

            x = self.xlp[line_point] = X / self.pixelSize
            y = self.ylp[line_point] = Y / self.pixelSize
            plt.text(x, y, str(N), fontsize=7, color='k', ha='left', rotation=45)

            if N==1 :
                x, y = self.xlp[0] + 100, self.ylp[0] + 100
                plt.text(x, y, 'Quad:'+str(quad), fontsize=12, color='k', ha='left', rotation=0)

            if line_point == 3 :
                #print 'Add new line:'
                #print 'x=',self.xlp                   
                #print 'y=',self.ylp
                self.xlp[4] = self.xlp[0]
                self.ylp[4] = self.ylp[0]
                line = lines.Line2D(self.xlp, self.ylp, linewidth=1, color='r')        
                axes.add_artist(line)
                line_point = -1
                self.xlp = [0,0,0,0,0]
                self.ylp = [0,0,0,0,0]
            line_point += 1

#----------------------------------

    def drawQuadsSeparately(self): 
        print 'drawQuadsSeparately()'

        sizex, sizey = shape = (100,100)
        fig   = plt.figure(figsize=(10,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)

        quadlims = (-50,870)
        
        for quad in range(4) :
            axes = fig.add_subplot(221+quad)
            axes.set_xlim(quadlims)
            axes.set_ylim(quadlims)
            self.drawOneQuad(quad,axes)

        plt.show()
        fig.savefig(self.fname_plot_quads)
        print 'Image saved in file:', self.fname_plot_quads

#----------------------------------

def main():

    #fname = '2011-03-29-CSPAD2-Alignment-PostRun3.txt'
    #fname = '2011-06-20-CSPAD2-Alignment-Before-Run4.txt'
    #fname = '2011-08-10-Metrology.txt'
    #fname = '2011-08-DD-Run4-DSD-Metrology.txt'
    #fname = '2012-01-10-Run5-DSD-Metrology.txt'
    #fname = '2012-01-12-Run5-DSD-Metrology-corrected.txt'
    #fname = '2012-02-26-CSPAD-XPP-Metrology.txt'
    #fname = '2012-11-08-Run6-DSD-Metrology-standard.txt'
    #fname = '2013-01-24-CSPAD-XPP-Metrology-standard.txt'
    #fname = 'metrology_renumerated.txt'
    fname = 'metrology_standard.txt'

    base_dir = '/reg/neh/home1/dubrovin/LCLS/CSPadMetrologyProc/'
    path_metrol = os.path.join(base_dir,fname)

    OpticAlignmentCspadV1(path_metrol, print_bits=0777, plot_bits=0377)
    sys.exit()

if __name__ == '__main__':
    main()

#----------------------------------
