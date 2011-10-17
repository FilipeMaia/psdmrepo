#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CSPadImageProducer...
#
#------------------------------------------------------------------------

"""This module provides access to the calibration parameters

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 2008-09-27$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#----------
#  Imports 
#----------

import sys
import os
import math
import numpy as np
import scipy.ndimage as spi # rotate(...)

import CalibPars as calp
import CSPadConfigPars as ccp

import GlobalGraphics as gg # For test purpose in main only
import HDF5Methods    as hm # For test purpose in main only

#---------------------
#  Class definition --
#---------------------

class CSPadImageProducer (object) :
    """This class produces the device dependent CSPad image"""

#---------------------

    def getImageArrayForPair( self, arr1ev, pairNum=None ):
        """Returns the image array for pair of ASICs"""
        if pairNum == None :
            self.pair = ccp.cspadconfig.cspadPair
        else :
            self.pair = pairNum

        asic2x1 = arr1ev[self.pair,...]
        asics   = np.hsplit(asic2x1,2)
        arrgap = np.zeros( (185,3), dtype=np.int16 )
        arr2d  = np.hstack((asics[0],arrgap,asics[1]))
        return arr2d

#---------------------

    def getImageArrayForQuad( self, arr1ev, quadNum=None ):
        """Returns the image array for one quad"""

        if ccp.cspadconfig.isCSPad2x2 : # For 2x2 Mini
            return self.getImageArrayForCSPadMiniElement( arr1ev )

        if quadNum == None :
            self.quad = ccp.cspadconfig.cspadQuad
        else :
            self.quad = quadNum            

        indPairsInQuads  = ccp.cspadconfig.indPairsInQuads
        pairInQaudOriInd = ccp.cspadconfig.pairInQaudOriInd

        pairXInQaud = calp.calibpars.getCalibPars ('center')[0]
        pairYInQaud = calp.calibpars.getCalibPars ('center')[1]
        dXInQaud    = calp.calibpars.getCalibPars ('center_corr')[0]
        dYInQaud    = calp.calibpars.getCalibPars ('center_corr')[1]
        offsetX     = calp.calibpars.getCalibPars ('marg_gap_shift')[0][0]
        offsetY     = calp.calibpars.getCalibPars ('marg_gap_shift')[1][0]
        dPhi        = calp.calibpars.getCalibPars ('tilt')

        #arr2dquad = np.zeros( (850,850), dtype=np.float32 ) # np.int16
        arr2dquad = np.zeros( (ccp.cspadconfig.quadDimX,ccp.cspadconfig.quadDimY), dtype=np.float32 ) # dtype=np.int16 
        #print 'arr2dquad.shape=',arr2dquad.shape

#       for ind in xrange(1): # loop over ind = 0,1,2,...,7
        for ind in xrange(8): # loop over ind = 0,1,2,...,7
            pair = indPairsInQuads[self.quad][ind]
            #print 'quad,ind,pair=', self.quad, ind, pair
            if pair == -1 : continue

            asic2x1 = self.getImageArrayForPair( arr1ev, pair )
            rotarr2d_0 = np.rot90(asic2x1,pairInQaudOriInd[self.quad][ind])
            #print 'rotarr2d_0.shape=',rotarr2d_0.shape
            #print 'rotarr2d.base is asic2x1 ? ',rotarr2d.base is asic2x1 

            rotarr2d = rotarr2d_0

            ixOff  = offsetX + pairXInQaud[self.quad][ind] + dXInQaud[self.quad][ind] 
            iyOff  = offsetY + pairYInQaud[self.quad][ind] + dYInQaud[self.quad][ind]
            #print 'ixOff, iyOff :', ixOff, iyOff

            # 0:185, 0:388 -> 185x391
            rot_index = pairInQaudOriInd[self.quad][ind] 

            offS = 0.5*185
            offL = 0.5*(388+3)
            #print 'offS, offL :', offS, offL

            if rot_index == 0 or rot_index == 2 :
                self.lx0 = offS  
                self.ly0 = offL  
            else :
                self.lx0 = offL  
                self.ly0 = offS  

            ixOff -= self.lx0  
            iyOff -= self.ly0  

            #-------- Apply tilt angle of 2x1 sensors
            if True :
            #if ccp.confpars.cspadApplyTiltAngle :

                r0      = math.sqrt( self.lx0*self.lx0 + self.ly0*self.ly0 )
                sinPhi  = self.ly0 / r0
                cosPhi  = self.lx0 / r0

                angle  = dPhi[self.quad][ind]
                rotarr2d = spi.rotate(rotarr2d_0, angle, reshape=True, output=np.float32 )
                dimX0,dimY0 = rotarr2d_0.shape

                rdphi = r0 * abs(math.radians(angle))
                #print 'rdphi :',rdphi

                ixOff -= rdphi * sinPhi
                iyOff -= rdphi * cosPhi

                #print 'Tilt offset dx, dy=', rdphi * sinPhi, rdphi * cosPhi

            #-------- 

            ixOff = int( ixOff )
            iyOff = int( iyOff )

            dimX, dimY = rotarr2d.shape
            #print 'ixOff, iyOff =', ixOff, iyOff,           
            #print ' dimX,  dimY =', dimX, dimY           
            
            arr2dquad[ixOff:dimX+ixOff, iyOff:dimY+iyOff] += rotarr2d[0:dimX, 0:dimY]

        #print 'arr2dquad=\n', arr2dquad
        return arr2dquad


#---------------------

    def getImageArrayForCSPadElement( self, arr1ev ):
        """Returns the image array for the CSPad detector for dataset CSPadElement"""

        quadInDetOriInd = ccp.cspadconfig.quadInDetOriInd
        quadNumsInEvent = ccp.cspadconfig.quadNumsInEvent

        margX    = calp.calibpars.getCalibPars ('marg_gap_shift')[0][1]
        margY    = calp.calibpars.getCalibPars ('marg_gap_shift')[1][1]
        gapX     = calp.calibpars.getCalibPars ('marg_gap_shift')[0][2]
        gapY     = calp.calibpars.getCalibPars ('marg_gap_shift')[1][2]
        shiftX   = calp.calibpars.getCalibPars ('marg_gap_shift')[0][3]
        shiftY   = calp.calibpars.getCalibPars ('marg_gap_shift')[1][3]
        offX     = calp.calibpars.getCalibPars ('offset')[0] + calp.calibpars.getCalibPars ('offset_corr')[0] + margX
        offY     = calp.calibpars.getCalibPars ('offset')[1] + calp.calibpars.getCalibPars ('offset_corr')[1] + margY

        #self.arr2dCSpad = np.zeros( (1710,1710), dtype=np.int16 )
        #self.arr2dCSpad = np.zeros( (1750,1750), dtype=np.int16 )
        #self.arr2dCSpad = np.zeros( (1765,1765), dtype=np.float32 )
        self.arr2dCSpad = np.zeros( (ccp.cspadconfig.detDimX,ccp.cspadconfig.detDimY), dtype=np.float32 )

        quadXOffset = [offX[0]-gapX+shiftX, offX[1]-gapX-shiftX, offX[2]+gapX-shiftX, offX[3]+gapX+shiftX]
        quadYOffset = [offY[0]-gapY-shiftY, offY[1]+gapY-shiftY, offY[2]+gapY+shiftY, offY[3]-gapY+shiftY]

        #for quad in range(4) :
        for quad in range(len(quadNumsInEvent)) :
            arr2dquad = self.getImageArrayForQuad(arr1ev, quad)
            rotarr2d = np.rot90(arr2dquad,quadInDetOriInd[quad])
            #print 'rotarr2d.shape=',rotarr2d.shape
            dimX,dimY = rotarr2d.shape

            ixOff = quadXOffset[quad]
            iyOff = quadYOffset[quad]

            self.arr2dCSpad[ixOff:dimX+ixOff, iyOff:dimY+iyOff] += rotarr2d[0:dimX, 0:dimY]

        return self.arr2dCSpad

#---------------------

    def getImageArrayForDet( self, arr1ev ):
        """Returns the image array for entire CSpad detector"""       

        if cp.confpars.eventCurrent == self.eventWithAlreadyGeneratedCSpadDetImage :
            #print 'Use already generated image for CSpad and save time'
            return self.arr2dCSpad

        if ccp.cspadconfig.isCSPad2x2 : # For 2x2 Mini
            self.arr2dCSpad = self.getImageArrayForCSPadMiniElement( arr1ev )

        else : # For regular CSPad detector
            self.arr2dCSpad = self.getImageArrayForCSPadElement( arr1ev )

        self.eventWithAlreadyGeneratedCSpadDetImage = cp.confpars.eventCurrent

        if cp.confpars.bkgdSubtractionIsOn : self.arr2dCSpad -= cp.confpars.arr_bkgd
        if cp.confpars.gainCorrectionIsOn  : self.arr2dCSpad *= cp.confpars.arr_gain

        return self.arr2dCSpad

#---------------------

    def getImageArrayForMiniElementPair( self, arr1ev, pairNum=None ):
        """Returns the image array for pair of ASICs"""
        if pairNum == None :
            self.pair = ccp.cspadconfig.cspadPair
        else :
            self.pair = pairNum

        #arr2x1 = arr1ev[0:185,0:388,self.pair]
        arr2x1 = arr1ev[:,:,self.pair]
        asics  = hsplit(arr2x1,2)
        arrgap = zeros ((185,3), dtype=np.float32)
        arr2d  = hstack((asics[0],arrgap,asics[1]))
        return arr2d

#---------------------

    def getImageArrayForCSPadMiniElement( self, arr1ev ):
        """Returns the image array for the CSpadMiniElement or CSpad2x2"""       

        arr2x1Pair0 = self.getImageArrayForMiniElementPair(arr1ev,0)
        arr2x1Pair1 = self.getImageArrayForMiniElementPair(arr1ev,1)
        wid2x1      = arr2x1Pair0.shape[0]
        len2x1      = arr2x1Pair0.shape[1]

        arrgapV = zeros( (20,len2x1), dtype=np.float ) # dtype=np.int16 
        arr2d   = vstack((arr2x1Pair0, arrgapV, arr2x1Pair1))

        #print 'arr2d.shape=', arr2d.shape
        #print 'arr2d=',       arr2d
        return arr2d

#---------------------

    def __init__ (self) :
        #print 'CSPadImageProducer(): Initialization'
        pass

#----------------------------------------------

def main() :

    print 'Start test'
    calp.calibpars.setCalibPars( run      = 9,
                                 calibdir = '/reg/d/psdm/CXI/cxi35711/calib',
                                 group    = 'CsPad::CalibV1',
                                 source   = 'CxiDs1.0:Cspad.0' )

    print 'CSPadImageProducer() object initialization'
    cspadimg = CSPadImageProducer()

    print 'Get one raw CSPad event: ',   
    ds1ev = hm.getOneCSPadEventForTest( fname  = '/reg/d/psdm/CXI/cxi35711/hdf5/cxi35711-r0009.h5',
                                        dsname = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDs1.0:Cspad.0/data',
                                        event  = 1 )
    print 'ds1ev.shape = ',ds1ev.shape

    print 'Make the CSPad image from raw event array'
    #arr = cspadimg.getImageArrayForPair( ds1ev, pairNum=3 )
    #arr = cspadimg.getImageArrayForQuad( ds1ev, quadNum=2 )
    arr = cspadimg.getImageArrayForCSPadElement( ds1ev )
    #print 'arr = \n',arr

    print 'Plot CSPad image'
    gg.plotImage(arr,range=(0,2000),figsize=(11.6,10))
    gg.move(200,100)
    #gg.plotImageAndSpectrum(arr,range=(1,2001))
    gg.plotSpectrum(arr,range=(10,2000))
    gg.move(50,50)
    gg.show()

#---------------------

if __name__ == "__main__" :

    main()
    sys.exit ( 'End of test.' )

#----------------------------------------------
