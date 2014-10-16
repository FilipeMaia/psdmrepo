#!/usr/bin/env python

#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#------------------------------------------------------------------------
""" Common methods for CSPAD alignment

@see OpticAlignmentCspadV1.py, OpticAlignmentCspadV2.py

@version $Id$

@author Mikhail S. Dubrovin
"""

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

import matplotlib.pyplot as plt
import matplotlib.lines  as lines

from optparse import OptionParser

#----------------------------------

class OpticAlignmentCspadMethods :
    """OpticAlignmentCspadMethods"""

    pixelSize = 109.92
    PI = 3.14159265359
    rad_to_deg = 180/PI

    nquads  = 4
    nsegms  = 8
    npoints = 32

    #Base index of 2x1s in the quad:
    #        0  1   2  3   4   5   6   7
    ibase = [5, 1, 13, 9, 17, 21, 29, 25] 

    #Segment origin index
    iorgn = [6, 2, 15,11, 20, 24, 31, 27]

    #sensor_rotation = [0,0,270,270,180,180,270,270]
    sensor_n90_in_quad = [0,0,3,3,2,2,3,3]


    def evaluate_deviation_from_flatness(self) :

        if self.print_bits & 256 : print '\nevaluate_deviation_from_flatness():'

        ix = 1
        iy = 2
        iz = 3
        nquads = self.nquads
        nsegms = self.nsegms

        self.arr_dev_um = numpy.zeros( (nquads,nsegms), dtype=numpy.double )

        for quad in range(nquads) :
           for segm in range(nsegms) :

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
               if self.print_bits & 256 : print 'quad:%1d, segm:%2d,  dz3[um]: %8.3f' % (quad, segm, self.arr_dev_um[quad,segm])


#----------------------------------

    def txt_deviation_from_flatness(self) :
        txt = 'Deviation from flatness [um] for segments:\n'
        for quad in range(self.nquads) :
           for segm in range(self.nsegms) :
               txt += 'quad:%1d, segm:%2d,  dz3[um]: %8.3f\n' % (quad, segm, self.arr_dev_um[quad,segm])
        txt += 'Mean and Standard deviation for dz3[um]: %8.3f +- %8.3f\n' % (np.mean(self.arr_dev_um), np.std(self.arr_dev_um))
        return txt


#----------------------------------

    def evaluate_center_coordinates(self) :

        if self.print_bits & 256 : print '\nevaluate_center_coordinates():'

        nquads = self.nquads
        nsegms = self.nsegms

        self.arrXmu = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.arrYmu = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.arrZmu = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.arrX   = numpy.zeros( (nquads, nsegms), dtype=numpy.float )
        self.arrY   = numpy.zeros( (nquads, nsegms), dtype=numpy.float )
        self.arrZ   = numpy.zeros( (nquads, nsegms), dtype=numpy.float )

        ix = 1
        iy = 2
        iz = 3

        for quad in range(nquads) :
            for segm in range(nsegms) :

                icor1, icor2, icor3, icor4 = [self.ibase[segm] + i for i in range(4)]
                
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

                if self.print_bits & 256 : print 'quad:%1d, segm:%2d,  X:%7d  Y:%7d, Z:%3d' % (quad, segm, X, Y, Z)

                self.arrXmu[quad][segm] = X
                self.arrYmu[quad][segm] = Y
                self.arrZmu[quad][segm] = Z

                self.arrX[quad][segm] = float(X) / self.pixelSize
                self.arrY[quad][segm] = float(Y) / self.pixelSize
                self.arrZ[quad][segm] = float(Z) / self.pixelSize

#----------------------------------

    def get_vectors_from_origin(self, quad, segm) :
        iorgn = self.iorgn[segm]
        icorn = [icor1, icor2, icor3, icor4] = [self.ibase[segm] + i for i in range(4)]
        #print '  quad:%d  segm:%d  orgn:%d    -> icorn: %s' % (quad, segm, iorgn, icorn) 

        dic_v = {}
        for i in icorn :
            if i == iorgn : continue
            v = self.arr[quad,i,1:] - self.arr[quad,iorgn,1:]
            vlen = math.sqrt(np.sum(np.square(v)))
            #print 'v.shape: %s, v: %s, vlen:%f ' % (v.shape, v, vlen)            
            dic_v[vlen] = v

        list_v_keys = sorted(dic_v.keys())
        #print '   sorted(list_v_keys) = ', list_v_keys
        vS1, vL1, vD1 = [dic_v[k] for k in list_v_keys]
        vS2 = vD1 - vL1 
        vL2 = vD1 - vS1 
        vD2 = vL1 - vS1 
        #print 'vS1, vS2, vL1, vL2, vD1, vD2 = ', vS1, vS2, vL1, vL2, vD1, vD2

        return vS1, vS2, vL1, vL2, vD1, vD2

#----------------------------------

    def evaluate_length_width_angle(self, n90=0) :

        if self.print_bits & 256 : print '\nevaluate_length_width_angle(n90=%d):' % n90

        nquads = self.nquads
        nsegms = self.nsegms

        self.S1  = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.S2  = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        self.dS1 = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dS2 = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        self.L1  = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.L2  = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        self.dL1 = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dL2 = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        self.dZS1 = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dZS2 = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dZL1 = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dZL2 = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        self.rotXYDegree = numpy.zeros( (nquads, nsegms), dtype=numpy.float32 )
        self.rotXZDegree = numpy.zeros( (nquads, nsegms), dtype=numpy.float32 )
        self.rotYZDegree = numpy.zeros( (nquads, nsegms), dtype=numpy.float32 )

        self.tiltXYDegree = numpy.zeros( (nquads, nsegms), dtype=numpy.float32 )
        self.tiltXZDegree = numpy.zeros( (nquads, nsegms), dtype=numpy.float32 )
        self.tiltYZDegree = numpy.zeros( (nquads, nsegms), dtype=numpy.float32 )

        self.D1  = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.D2  = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dD  = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        self.ddS = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.ddL = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        self.ddZS = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.ddZL = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        self.dZSA = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dZLA = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        self.SA  = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.LA  = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dSA = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dLA = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        self.XSize = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.YSize = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dZX   = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )
        self.dZY   = numpy.zeros( (nquads, nsegms), dtype=numpy.int32 )

        ix = 0
        iy = 1
        iz = 2

        for quad in range(nquads) :

            for segm in range(nsegms) :

                segm_n90 = n90 + self.quad_n90_in_det[quad] + self.sensor_n90_in_quad[segm]
                segm_n90 = segm_n90 % 4

                vS1, vS2, vL1, vL2, vD1, vD2 = self.get_vectors_from_origin(quad, segm)

                if segm_n90 == 0 or segm_n90 == 2 :
                    # horizontal 2x1

                    self. S1[quad][segm]  = vS1[iy]
                    self. S2[quad][segm]  = vS2[iy]

                    self.dS1[quad][segm]  = vL1[iy]
                    self.dS2[quad][segm]  = vL2[iy]

                    self. L1[quad][segm]  = vL1[ix]
                    self. L2[quad][segm]  = vL2[ix]

                    self.dL1[quad][segm]  = vS1[ix]
                    self.dL2[quad][segm]  = vS2[ix]

                    self.dZS1[quad][segm] = vS1[iz]
                    self.dZS2[quad][segm] = vS2[iz]
                    self.dZL1[quad][segm] = vL1[iz]
                    self.dZL2[quad][segm] = vL2[iz]

                    self.evaluateSLAverage(quad,segm)

                    self.XSize[quad][segm] = math.fabs(self.LA[quad][segm])
                    self.YSize[quad][segm] = math.fabs(self.SA[quad][segm])
                    self.dZX  [quad][segm] = self.dZLA[quad][segm] 
                    self.dZY  [quad][segm] = self.dZSA[quad][segm]  


                if segm_n90 == 1 or segm_n90 == 3  :
                    # vertical 2x1

                    self. S1[quad][segm]  = vS1[ix]
                    self. S2[quad][segm]  = vS2[ix]
                                                   
                    self.dS1[quad][segm]  = vL1[ix]
                    self.dS2[quad][segm]  = vL2[ix]
                                                   
                    self. L1[quad][segm]  = vL1[iy]
                    self. L2[quad][segm]  = vL2[iy]
                                                   
                    self.dL1[quad][segm]  = vS1[iy]
                    self.dL2[quad][segm]  = vS2[iy]
                                                   
                    self.dZS1[quad][segm] = vS1[iz]
                    self.dZS2[quad][segm] = vS2[iz]
                    self.dZL1[quad][segm] = vL1[iz]
                    self.dZL2[quad][segm] = vL2[iz]

                    self.evaluateSLAverage(quad,segm)

                    self.XSize[quad][segm] = math.fabs(self.SA[quad][segm])
                    self.YSize[quad][segm] = math.fabs(self.LA[quad][segm])
                    self.dZX  [quad][segm] = self.dZSA[quad][segm] 
                    self.dZY  [quad][segm] = self.dZLA[quad][segm]  


                self.D1[quad][segm] = math.sqrt(np.sum(np.square(vD1)))
                self.D2[quad][segm] = math.sqrt(np.sum(np.square(vD2)))
                self.dD[quad][segm] = self.D1[quad][segm] - self.D2[quad][segm]

                self.ddS[quad][segm] = self.dS1[quad][segm] - self.dS2[quad][segm]
                self.ddL[quad][segm] = self.dL1[quad][segm] - self.dL2[quad][segm]

                self.ddZS[quad][segm] = self.dZS1[quad][segm] - self.dZS2[quad][segm]
                self.ddZL[quad][segm] = self.dZL1[quad][segm] - self.dZL2[quad][segm]

                self.rotXYDegree[quad][segm] = segm_n90 * 90
                #self.rotXZDegree[quad][segm] = 0
                #self.rotYZDegree[quad][segm] = 0

                vLA = 0.5*(vL1+vL2)
                vSA = 0.5*(vS1+vS2)

                tiltXY = math.atan2(vLA[iy], vLA[ix])
                tiltXZ = math.atan2(self.dZX[quad][segm], self.XSize[quad][segm]) 
                tiltYZ = math.atan2(self.dZY[quad][segm], self.YSize[quad][segm]) 

                #vLlen = math.sqrt(np.sum(np.square(vLA)))
                #vSlen = math.sqrt(np.sum(np.square(vSA)))
                #tiltXZ = math.atan2(vLA[iz], vLlen) 
                #tiltYZ = math.atan2(vSA[iz], vSlen)  

                if abs(tiltXY)>0.1 and tiltXY<0 : tiltXY += 2*self.PI # move angle in range [0,2pi]

                self.tiltXYDegree[quad][segm] = self.rad_to_deg * tiltXY - self.rotXYDegree[quad][segm]
                self.tiltXZDegree[quad][segm] = self.rad_to_deg * tiltXZ - self.rotXZDegree[quad][segm]
                self.tiltYZDegree[quad][segm] = self.rad_to_deg * tiltYZ - self.rotYZDegree[quad][segm]

                #print 'rotXY : %f' % self.rotXYDegree[quad][segm]
                #print 'tiltXY: %f' % self.tiltXYDegree[quad][segm]

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
        for quad in range(self.nquads) :
            txt += '\nQuad  %d\n' % quad
            for segm in range(self.nsegms) :
                txt += 'segm: %1d  %6d  %6d  %6d  %6d    %6d  %6d  %6d  %6d   %8.5f  %6d  %6d  %6d  %6d  %6d\n' % \
                    (segm, self.S1[quad][segm], self.S2[quad][segm], self.dS1[quad][segm], self.dS2[quad][segm], \
                           self.L1[quad][segm], self.L2[quad][segm], self.dL1[quad][segm], self.dL2[quad][segm], \
                           self.tiltXYDegree[quad][segm], \
                           self.D1[quad][segm], self.D2[quad][segm], self.dD[quad][segm], self.ddS[quad][segm], self.ddL[quad][segm] )
        return txt

#----------------------------------

    def txt_qc_table_z(self) :
        txt = 'segm:        SA      LA   XSize   YSize    dZS1  dZS2  dZL1  dZL2    dZSA  dZLA  ddZS  ddZL     dZX   dZY   angXZ(deg) angYZ(deg) dz3(um)'
        for quad in range(self.nquads) :
            txt += '\nQuad  %d\n' % quad
            for segm in range(self.nsegms) :
                txt += 'segm: %1d  %6d  %6d  %6d  %6d   %5d %5d %5d %5d   %5d %5d %5d %5d   %5d %5d  %8.5f   %8.5f   %8.3f\n' % \
                    (segm, self.SA[quad][segm],   self.LA[quad][segm],   self.XSize[quad][segm], self.YSize[quad][segm], \
                           self.dZS1[quad][segm], self.dZS2[quad][segm], self.dZL1[quad][segm],  self.dZL2[quad][segm], \
                           self.dZSA[quad][segm], self.dZLA[quad][segm], self.ddZS[quad][segm],  self.ddZL[quad][segm], \
                           self.dZX[quad][segm],  self.dZY[quad][segm],  self.tiltXZDegree[quad][segm], self.tiltYZDegree[quad][segm], \
                           self.arr_dev_um[quad,segm])
        return txt #+'\n' 

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
        for row in range(self.nquads) :
            for col in range(self.nsegms) :
                txt += format % (arr[row][col])
            txt += '\n'
        return txt

#----------------------------------

    def create_directory(self, dir) : 
        if os.path.exists(dir) :
            if self.print_bits & 1 : print 'Directory exists: %s' % dir
        else :
            os.makedirs(dir)
            if self.print_bits & 1 : print 'Directory created: %s' % dir 

#----------------------------------

    def save_text_file(self, fname, text) :
        if self.print_bits & 256 : print 'Save text file: %s' % fname
        f=open(fname,'w')
        f.write( text )
        f.close() 

#----------------------------------
#----------------------------------
#---------- GRAPHICS --------------
#----------------------------------
#----------------------------------

    def drawOpticalAlignmentFile(self): 
        print 'drawOpticalAlignmentFile()'

        sizex, sizey = shape = (100,100)
        #arr   = np.arange(sizex*sizey)
        #arr.shape = shape
        #arr   = np.zeros(shape)
        fig   = plt.figure(figsize=(10,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)
        axes  = fig.add_subplot(111)        

        if self.nquads == 1 and self.nsegms == 2:
            axes.set_xlim((-50,450))
            axes.set_ylim((-50,450))
        else :
            axes.set_xlim((-50,1750))
            axes.set_ylim((-50,1750))
        #axes1 = plt.imshow(arr, origin='lower', interpolation='nearest',extent=ax_range) 

        for quad in range(self.nquads) :
            #print '\nQuad:', quad
            self.drawOneQuad(quad,axes)

        plt.show()
        fig.savefig(self.fname_plot_det)
        print 'Image saved in file:', self.fname_plot_det


#----------------------------------

    def drawOneQuad(self,quad,axes):
        print 'drawOneQuad(' + str(quad) + ')'

        line_point = 0
        self.xlp = [0,0,0,0,0]
        self.ylp = [0,0,0,0,0]
        for point in range(1,self.npoints+1) :
            N = self.arr[quad,point,0]
            X = self.arr[quad,point,1]
            Y = self.arr[quad,point,2]
            Z = self.arr[quad,point,3]                
            #print 'N,X,Y =', N,X,Y

            x = self.xlp[line_point] = X / self.pixelSize
            y = self.ylp[line_point] = Y / self.pixelSize
            plt.text(x, y, str(N), fontsize=7, color='k', ha='left', rotation=45)

            if N==1 :
                #x, y = self.xlp[0] + 100, self.ylp[0] + 100
                x = 0.5*(self.arr[quad,point,1]+self.arr[quad,point+2,1])/ self.pixelSize - 70
                y = 0.5*(self.arr[quad,point,2]+self.arr[quad,point+2,2])/ self.pixelSize 
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
        
        for quad in range(self.nquads) :
            axes = fig.add_subplot(221+quad)
            axes.set_xlim(quadlims)
            axes.set_ylim(quadlims)
            self.drawOneQuad(quad,axes)

        plt.show()
        fig.savefig(self.fname_plot_quads)
        print 'Image saved in file:', self.fname_plot_quads


#----------------------------------
 
    def txt_geometry_header(self) :
        txt = '# TITLE      Geometry parameters of %s' % self.det \
            + '\n# DATE_TIME  %s' % strftime('%Y-%m-%d %H:%M:%S %Z', localtime()) \
            + '\n# METROLOGY  %s' % self.fname \
            + '\n# AUTHOR     %s' % os.environ['LOGNAME'] \
            + '\n# EXPERIMENT %s' % self.exp \
            + '\n# DETECTOR   %s' % self.det \
            + '\n# CALIB_TYPE geometry' \
            + '\n# COMMENT:01 Table contains the list of geometry parameters for alignment of 2x1 sensors, quads, CSPAD, etc' \
            + '\n# COMMENT:02 All translation and rotation pars of the object are defined w.r.t. parent object Cartesian frame' \
            + '\n# PARAM:01 PARENT     - name and version of the parent object' \
            + '\n# PARAM:02 PARENT_IND - index of the parent object' \
            + '\n# PARAM:03 OBJECT     - name and version of the object' \
            + '\n# PARAM:04 OBJECT_IND - index of the new object' \
            + '\n# PARAM:05 X0         - x-coordinate [um] of the object origin in the parent frame' \
            + '\n# PARAM:06 Y0         - y-coordinate [um] of the object origin in the parent frame' \
            + '\n# PARAM:07 Z0         - z-coordinate [um] of the object origin in the parent frame' \
            + '\n# PARAM:08 ROT_Z      - object design rotation angle [deg] around Z axis of the parent frame' \
            + '\n# PARAM:09 ROT_Y      - object design rotation angle [deg] around Y axis of the parent frame' \
            + '\n# PARAM:10 ROT_X      - object design rotation angle [deg] around X axis of the parent frame' \
            + '\n# PARAM:11 TILT_Z     - object tilt angle [deg] around Z axis of the parent frame' \
            + '\n# PARAM:12 TILT_Y     - object tilt angle [deg] around Y axis of the parent frame' \
            + '\n# PARAM:13 TILT_X     - object tilt angle [deg] around X axis of the parent frame' \
            + '\n\n# HDR PARENT IND        OBJECT IND     X0[um]   Y0[um]   Z0[um]   ROT-Z ROT-Y ROT-X     TILT-Z   TILT-Y   TILT-X'

        return txt + '\n\n'

#----------------------------------

    def str_fmt(self) :
        return '%s %3d  %s %3d   %8d %8d %8d   %5d %5d %5d   %8.5f %8.5f %8.5f \n'

#----------------------------------

    def txt_geometry_segments(self, name_segm='SENS2X1:V1', name_parent='QUAD:V1') :
        txt = ''        
        rotXZ, rotYZ = 0,0
        for quad in range(self.nquads) :
            for segm in range(self.nsegms) :
                txt += self.str_fmt() % \
                       (name_parent.ljust(12), quad, name_segm.ljust(12), segm, \
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
def input_option_parser(dir_def, fname_def) :

    parser = OptionParser(description='Optional input parameters.', usage ='usage: %prog [options] args')
    parser.add_option('-d', '--dir',   dest='dir',    default=dir_def,   action='store', type='string', help='directory for metrology files')
    parser.add_option('-f', '--fnm',   dest='fname',  default=fname_def, action='store', type='string', help='metrology file name')
    parser.add_option('-p', '--pbits', dest='pbits',  default=0377,      action='store', type='int',    help='print control bitword')
    parser.add_option('-g', '--gbits', dest='gbits',  default=0377,      action='store', type='int',    help='graphics control bitword')
    #parser.add_option('-v', '--verbose',      dest='verb',    default=True, action='store_true',           help='allows print on console')
    #parser.add_option('-q', '--quiet',        dest='verb',                  action='store_false',          help='supress print on console')

    (opts, args) = parser.parse_args()

    print 'opts:\n', opts
    print 'args:\n', args

    return (opts, args)

#----------------------------------
 
