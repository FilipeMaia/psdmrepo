#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GlobalExternal...
#
#------------------------------------------------------------------------

"""Contains global external methods

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 

@author Marcin Sikorski
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

import numpy as np
import math
from math import sqrt, cos


def map_image(s,beam0, pixel_size, sample_detector,energy,det_pos,sense):
	# s - data image size
	# beam0[x0,y0] - coordinates of direct beam (pix)
	# pixel_size[x,y] - pixel size in x and y direction (mm)
	# sample to detector - sample to detector distance  (mm)
	# energy - energy of x-rays                         (keV)
	# det_pos - detector position (CCDx and CCDy motor values) 
	#	det_pos[ [CCDx,CCDy],		- for direct beam measurements (mm)
	#	         [CCDx,CCDy] ]          - for data acquisition         (mm)
	# sense[a,b] - a=-1,1 and b=-1,1   this variable takes into account rotation of the camera (during mounting at the detector stage)


	# q_map - matrix used to store wavevector values
	q_map = np.zeros(s,np.float32) 
	wavelength = 1.236/energy
	# xDBpix, yDBpix -  direct beam coordinates for detector at "data acquisition"        
        xDBpix = beam0[0] + sense[0]*(det_pos[1][0] - det_pos[0][0]) / pixel_size[0]
        yDBpix = beam0[1] + sense[1]*(det_pos[1][1] - det_pos[0][1]) / pixel_size[1]
	pre = 4*(math.pi/wavelength)
	for i in range(0,s[0]):
		for j in range(0, s[1]):
			 R = sqrt((pixel_size[0]**2)*(yDBpix-i)**2 + (pixel_size[1]**2)*(xDBpix-j)**2)
			 q_map[i,j] = np.sin(0.5*np.arctan(R/sample_detector));
	
	q_map = q_map*pre
	return q_map





def map_image_refl(s, alpha, beam0, beam_s, pixel_size, sample_detector, energy, det_pos, sense = [1,1]):
	#  s: frame size
	#  alpha: incident angle in degree (nominal value)
	#  beam0[x0,y0]: direct beam position ([x0,y0]) (pix)
	#  beam_s[xs,ys]: specular beam position (pix)
	#  pixel_size                                   (mm/pix)
	#  sample_detector: sample to detector distance (mm)
	#  energy:  energy of x-rays [keV]
	#  det_pos: 3x2 array of detector postion during direct beam measurements, specular beam measurements, spekcle pattern collection (mm)
    	#  sense[a,b] - a=-1,1 and b=-1,1   this variable takes into account rotation of the camera (during mounting at the detector stage)
         
     alpha = (math.pi/180.0)*alpha
     q_map = np.zeros(s,np.float64)
     wavelength = (1.236)/energy  #   [nm] are desired units  should be: 1.23984
     xmm = np.zeros(s,np.float64)
     ymm = np.zeros(s,np.float64)
     d2Beam0 = np.zeros(s,np.float64)
  
     # direct beam -  direct beam coordinates for detector at "data acquisition"  	
     xDBpix = beam0[0] + sense[0]*(det_pos[2][0] - det_pos[0][0]) / pixel_size[0]
     yDBpix = beam0[1] + sense[1]*(det_pos[2][1] - det_pos[0][1]) / pixel_size[1]	

     # specular beam -  specular beam coordinates for detector at "data acquisition"  
     xRBpix = beam_s[0] + sense[0]*(det_pos[2][0] - det_pos[1][0]) / pixel_size[0]
     yRBpix = beam_s[1] + sense[1]*(det_pos[2][1] - det_pos[1][1]) / pixel_size[1]

     for i in range(s[0]):
        for j in range(s[1]):
           xmm[i,j] = j - xDBpix
           ymm[i,j] = i - yDBpix
   
     # d2Beam0 - distance to direct beam	
     d2Beam0 = np.sqrt(np.add((pixel_size[0]**2)*xmm**2,(pixel_size[1]**2)*ymm**2))	
   
     # distance from direct beam to specular beam [mm]
     xDB2RB = (xRBpix - xDBpix) * pixel_size[0]                                    
     yDB2RB = (yRBpix - yDBpix) * pixel_size[1]
     dDB2RB = sqrt( xDB2RB**2 + yDB2RB**2)	
   
     # calculate true incident angle
     dDB2RB = sqrt( xDB2RB**2 + yDB2RB**2)	                                   
     true_alpha = 0.5*math.atan( dDB2RB / sample_detector)
     alpha = true_alpha
      	
     # determine sample tilt in horizontal plane
     if ( yDB2RB != 0 ):
        tilt = math.atan( xDB2RB / yDB2RB )                                     
     else:
        #tilt = signum(xDB2RB) * math.pi / 2    
        tilt = math.copysign(1,xDB2RB) * math.pi / 2    
                                   
     # projected distance of each pixel to Plane Of Reflection [POR])
     # [positive means above the streak, negative below the streak]
     d2POR = np.zeros(s, np.float32)  
     d2POR = xmm* math.sin(math.pi/2 - tilt)	                                             
     v = np.where(ymm != 0)
     d2POR[v] =  np.multiply(d2Beam0[v],np.sin(np.arctan(np.divide(xmm[v],1.0*ymm[v])) - tilt))            

     v = np.where(xmm > 0)
     v2 = np.where(ymm == 0)
     r = [i for i in v if i in v2]
     t = np.ones_like(xmm)
     t[xmm<0]  = 0
     t[ymm!=0]  = 0

     v = np.where(t ==1)  
     d2POR[v] =  d2Beam0[v]* math.sin(-math.pi/2 - tilt)
     v = np.where(xmm < 0)
     v2 = np.where(ymm == 0)
  
     t = np.ones_like(xmm)
     t[xmm>0]  = 0
     t[ymm!=0]  = 0		
     v = np.where(t ==1)	
     d2POR[v] =  d2Beam0[v]* math.sin(math.pi/2 - tilt)

     v = np.where(ymm == 0)
     v2 = np.where(xmm == 0)
     r = [i for i in v if i in v2]
     t = np.ones_like(xmm)
     t[xmm!=0]  = 0
     t[ymm!=0]  = 0
     v = np.where(t ==1)  
     d2POR[v] =  d2Beam0[v]* math.sin(-tilt)
     # in plane exit angle of each pixel (not true exit angle)
     
     inPlaneExitAngle =   np.arctan(np.sqrt(np.subtract((d2Beam0)**2,d2POR**2))/sample_detector)- alpha
    
     # distance of projected point (PPT) to sample
     dPPt2Sample = np.sqrt(np.subtract(d2Beam0**2,d2POR**2)+ sample_detector**2)
     # out of plane angle
     outOfPlaneAngle = np.arctan(np.divide(d2POR,np.multiply(dPPt2Sample,np.cos(inPlaneExitAngle))))   
     # true exit angle
     exitAngle = np.multiply(np.sign(inPlaneExitAngle), np.arccos(np.divide(np.sqrt(np.add(d2POR**2,np.multiply(dPPt2Sample,np.cos(inPlaneExitAngle))**2)), np.sqrt(d2Beam0**2 + sample_detector**2))))
    
     #qz = 2*math.pi/wavelength* np.add(np.sin(alpha), sin(exitAngle))
     qx = 2*math.pi/wavelength * np.subtract( cos(alpha), np.multiply(np.cos(exitAngle),np.cos(outOfPlaneAngle))) 
     qy = 2*math.pi/wavelength * np.multiply( np.cos(exitAngle),np.sin(outOfPlaneAngle))              
     qp   = np.sqrt(qx**2 + qy**2)                 
     return qp    


#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
# Test
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

import sys
from time import localtime, gmtime, strftime, clock, time, sleep

from PyQt4 import QtGui, QtCore
import PlotImgSpe as pis
from ConfigParametersCorAna import confpars as cp

#-----------------------------


def get_q_map_transition() :
    s               = (1300, 1340)
    beam0           = [300.,  400.]
    pixel_size      = [0.1, 0.1]
    sample_detector = 2000.
    energy          = 7.
    det_pos         = [[0,0],[0,0]]
    sense           = [1,1]
    t_start_sec = time()
    q_map = map_image(s,beam0, pixel_size, sample_detector, energy, det_pos, sense)
    print 'Time (sec) to produce q-map: ', time() - t_start_sec
    return q_map

#def map_image(s,beam0, pixel_size, sample_detector,energy,det_pos,sense):
	# s - data image size
	# beam0[x0,y0] - coordinates of direct beam (in pix)
	# pixel_size[x,y] - pixel size in x and y direction
	# sample to detector - sample to detector distance
	# energy - energy of x-rays
	# det_pos - detector position (CCDx and CCDy motor values) 
	#	det_pos[ [CCDx,CCDy],		- for direct beam measurements
	#	         [CCDx,CCDy] ]          - for data acquisition
	# sense[a,b] - a=-1,1 and b=-1,1   this variable takes into account rotation of the camera (during mounting at the detector stage)



def get_q_map_reflection() :
    s               = (1300, 1340)
    alpha           = 1
    beam0           = [300.,  400.]
    beam_s          = [300.,  500.]
    pixel_size      = [0.1, 0.1]
    sample_detector = 100.
    energy          = 7.
    det_pos         = [[0,0],[0,0],[0,0]]
    sense           = [1,1]
    t_start_sec = time()
    q_map = map_image_refl(s, alpha, beam0, beam_s, pixel_size, sample_detector, energy, det_pos, sense)
    print 'Time (sec) to produce q-map: ', time() - t_start_sec
    return q_map

#def map_image_refl(s, alpha, beam0, beam_s, pixel_size, sample_detector, energy, det_pos, sense = [1,1]):
	#  s: frame size
	#  alpha: incident angle in [degree] (nominal value)
	#  beam0[x0,y0]: direct beam position ([x0,y0])
	#  beam_s[xs,ys]: specular beam position
	#  pixel_size
	#  sample_detector: sample to detector distance
	#  energy:  energy of x-rays [keV]
	#  det_pos: 3x2 array of detector postion during direct beam measurements, specular beam measurements, spekcle pattern collection
    	#  sense[a,b] - a=-1,1 and b=-1,1   this variable takes into account rotation of the camera (during mounting at the detector stage)

#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25

    shape = (rows, cols) = (1300, 1340)
    size = rows * cols
    #arr = mu + sigma*np.random.standard_normal(size=2400)
    arr = 100*np.random.standard_exponential(size=size)
    #arr = np.arange(2400)
    arr.shape = shape
    return arr

#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)
    #w  = pis.PlotImgSpe(None, get_array2d_for_test())
    #w  = pis.PlotImgSpe(None, get_q_map_transition())
    w  = pis.PlotImgSpe(None, get_q_map_reflection())
    #w  = PlotImgSpe(None)
    #w.set_image_array( get_array2d_for_test() )
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------







