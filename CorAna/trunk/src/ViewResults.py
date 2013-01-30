#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ViewResults...
#
#------------------------------------------------------------------------

"""First look at results

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id:$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import math
import numpy as np

from ConfigParametersCorAna   import confpars as cp
from Logger                   import logger
from FileNameManager          import fnm
import GlobalUtils            as     gu
from PlotImgSpe               import *

#-----------------------------

def cart2polar(x, y) :
    r = np.sqrt(x*x + y*y)
    theta = np.rad2deg(np.arctan2(y, x))
    return r, theta

def cart2r(x, y) :
    return np.sqrt(x*x + y*y)

def cart2theta(x, y) :
    return np.rad2deg(np.arctan2(y, x))

def polar2cart(r, theta) :
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

#-----------------------------

class ViewResults :
    """First look at results.
    """

    def __init__(sp, fname=None) :
        """
        @param fname the file name with results
        """

        sp.set_file_name(fname)
        sp.set_parameters()
        sp.evaluate_parameters()
        sp.print_parameters()

        sp.ccd_pixel_coordinates()
        #q_map = sp.q_map_for_direct_beam_data()

#-----------------------------

    def set_parameters(sp) :

        sp.rows            = cp.bat_img_rows.value()
        sp.cols            = cp.bat_img_cols.value()
        sp.size            = cp.bat_img_size.value()
        sp.col_begin       = cp.col_begin   .value()    
        sp.col_end         = cp.col_end     .value()    
        sp.row_begin       = cp.row_begin   .value()    
        sp.row_end         = cp.row_end     .value()

        sp.x_coord_beam0   = cp.x_coord_beam0.value()
        sp.y_coord_beam0   = cp.y_coord_beam0.value()
        sp.x0_pos_in_beam0 = cp.x0_pos_in_beam0.value()
        sp.y0_pos_in_beam0 = cp.y0_pos_in_beam0.value()

        sp.x_coord_spec    = cp.x_coord_specular.value()
        sp.y_coord_spec    = cp.y_coord_specular.value()
        sp.x0_pos_spec     = cp.x0_pos_in_specular.value()
        sp.y0_pos_spec     = cp.y0_pos_in_specular.value()

        sp.x0_pos_in_data  = cp.x0_pos_in_data .value()
        sp.y0_pos_in_data  = cp.y0_pos_in_data .value()

        sp.ccd_orient      = cp.ccd_orient.value()
        sp.ccd_pixsize     = cp.ccdset_pixsize.value()

        sp.photon_energy   = cp.photon_energy.value() # [keV]
        sp.nom_angle       = cp.nominal_angle.value()
        sp.real_angle      = cp.real_angle.value()
        sp.distance        = cp.sample_det_dist.value()

#-----------------------------

    def evaluate_parameters(sp) :
        sp.wavelength   = 1.23984/sp.photon_energy # 1.23984 ? [nm]
        sp.factor       = 4*(math.pi/sp.wavelength)
        sp.distance_pix = sp.distance / sp.ccd_pixsize 
        
#-----------------------------

    def transform_rc_to_xy(sp) :
        path

#-----------------------------

    def ccd_pixel_coordinates(sp) :
        ir = np.arange(sp.rows)
        ic = np.arange(sp.cols)

        if   sp.ccd_orient == '0' :
            sp.x_ccd_pix = ic
            sp.y_ccd_pix = sp.rows - ir
            sp.X_ccd_pix, sp.Y_ccd_pix = np.meshgrid(sp.x_ccd_pix, sp.y_ccd_pix)
            
        elif sp.ccd_orient == '90' :
            sp.x_ccd_pix = ir
            sp.y_ccd_pix = ic
            sp.Y_ccd_pix, sp.X_ccd_pix = np.meshgrid(sp.y_ccd_pix, sp.x_ccd_pix)

        elif sp.ccd_orient == '180' :
            sp.x_ccd_pix = sp.cols - ic
            sp.y_ccd_pix = ir
            sp.X_ccd_pix, sp.Y_ccd_pix = np.meshgrid(sp.x_ccd_pix, sp.y_ccd_pix)

        elif sp.ccd_orient == '270' :
            sp.x_ccd_pix = sp.rows - ir
            sp.y_ccd_pix = sp.cols - ic
            sp.Y_ccd_pix, sp.X_ccd_pix = np.meshgrid(sp.y_ccd_pix, sp.x_ccd_pix)

        else :
            logger.error('Non-existent CCD orientation: ' + str(sp.ccd_orient), __name__)            

        #sp.x_ccd = sp.x_ccd_pix * sp.ccd_pixsize
        #sp.y_ccd = sp.y_ccd_pix * sp.ccd_pixsize

        #print 'ir:', ir
        #print 'ic:', ic
        #print 'x_ccd:', sp.x_ccd
        #print 'y_ccd:', sp.y_ccd

        #print 'X_ccd_pix.shape =', sp.X_ccd_pix.shape
        #print 'Y_ccd_pix.shape =', sp.Y_ccd_pix.shape
        #print 'X_ccd_pix:\n', sp.X_ccd_pix
        #print 'Y_ccd_pix:\n', sp.Y_ccd_pix

#-----------------------------

    def xy_maps_for_direct_beam_data(sp) :
        t0 = gu.get_time_sec()
        x_db_pix = sp.x_coord_beam0 + (sp.x0_pos_in_data - sp.x0_pos_in_beam0) / sp.ccd_pixsize
        y_db_pix = sp.y_coord_beam0 + (sp.y0_pos_in_data - sp.y0_pos_in_beam0) / sp.ccd_pixsize
        sp.X = sp.X_ccd_pix - x_db_pix
        sp.Y = sp.Y_ccd_pix - y_db_pix
        #print 'xy_maps_... consumed time: ', gu.get_time_sec()-t0 # < 0.02sec for 1300x1340 img  
        return sp.X, sp.Y  

#-----------------------------

    def rtheta_maps_for_direct_beam_data(sp) :
        x_map, y_map = sp.xy_maps_for_direct_beam_data()
        sp.R, sp.Theta = cart2polar(x_map, y_map)
        return sp.R, sp.Theta
  
#-----------------------------

    def r_map_for_direct_beam_data(sp) :
        x_map, y_map = sp.xy_maps_for_direct_beam_data()
        t0 = gu.get_time_sec()
        sp.R  = cart2r(x_map, y_map)
        #print 'r_map_... consumed time: ', gu.get_time_sec()-t0 # < 0.04sec for 1300x1340 img  
        return sp.R
  
#-----------------------------

    def q_map_for_direct_beam_data(sp) :
        t0 = gu.get_time_sec()
        sp.R = sp.r_map_for_direct_beam_data()
        sp.Q = sp.factor * np.sin(0.5*np.arctan2(sp.R, sp.distance_pix))
        #print 'q_map_... consumed time: ', gu.get_time_sec()-t0 # < 0.4sec for 1300x1340 img  
        return sp.Q
  
#-----------------------------

    def theta_map_for_direct_beam_data(sp) :
        x_map, y_map = sp.xy_maps_for_direct_beam_data()
        t0 = gu.get_time_sec()
        sp.Theta  = cart2theta(x_map, y_map)
        #print 'theta_map_... consumed time: ', gu.get_time_sec()-t0 # < 0.02sec for 1300x1340 img  
        return sp.Theta
  
#-----------------------------

    def print_parameters(sp) :

        logger.info('rows            = ' + str(sp.rows           ), __name__)
        logger.info('cols            = ' + str(sp.cols           ), __name__)
        logger.info('size            = ' + str(sp.size           ), __name__)
        logger.info('col_begin       = ' + str(sp.col_begin      ), __name__)
        logger.info('col_end         = ' + str(sp.col_end        ), __name__)
        logger.info('row_begin       = ' + str(sp.row_begin      ), __name__)
        logger.info('row_end         = ' + str(sp.row_end        ), __name__)

        logger.info('x_coord_beam0   = ' + str(sp.x_coord_beam0  ), __name__)
        logger.info('y_coord_beam0   = ' + str(sp.y_coord_beam0  ), __name__)
        logger.info('x0_pos_in_beam0 = ' + str(sp.x0_pos_in_beam0), __name__)
        logger.info('y0_pos_in_beam0 = ' + str(sp.y0_pos_in_beam0), __name__)
                                                                   
        logger.info('x_coord_spec    = ' + str(sp.x_coord_spec   ), __name__)
        logger.info('y_coord_spec    = ' + str(sp.y_coord_spec   ), __name__)
        logger.info('x0_pos_spec     = ' + str(sp.x0_pos_spec    ), __name__)
        logger.info('y0_pos_spec     = ' + str(sp.y0_pos_spec    ), __name__)
                                                                   
        logger.info('x0_pos_in_data  = ' + str(sp.x0_pos_in_data ), __name__)
        logger.info('y0_pos_in_data  = ' + str(sp.y0_pos_in_data ), __name__)
                                                                   
        logger.info('ccd_orient      = ' + str(sp.ccd_orient     ), __name__)
        logger.info('ccd_pixsize     = ' + str(sp.ccd_pixsize    ), __name__)

        logger.info('photon_energy   = ' + str(sp.photon_energy  ), __name__)
        logger.info('nom_angle       = ' + str(sp.nom_angle      ), __name__)
        logger.info('real_angle      = ' + str(sp.real_angle     ), __name__)
        logger.info('distance        = ' + str(sp.distance       ), __name__)
                                                                   
        logger.info('Evaluated parameters:', __name__)

        logger.info('sp.wavelength   = ' + str(sp.wavelength     ), __name__)
        logger.info('sp.factor       = ' + str(sp.factor         ), __name__)
#        logger.info('' + str(), __name__)

#-----------------------------

    def set_file_name(sp, fname=None) :
        if fname == None : sp.fname = cp.res_fname.value()
        else :             sp.fname = fname

#-----------------------------

    def get_cor_array_from_text_file(sp) :
        logger.info('get_cor_array_from_text_file: ' + sp.fname, __name__)
        #return np.loadtxt(fname, dtype=np.float32)


    def get_cor_array_from_binary_file(sp) :
        logger.info('get_cor_array_from_binary_file: ' + sp.fname, __name__)
        sp.arr = np.fromfile(sp.fname, dtype=np.float32)

        nptau = sp.arr.shape[0]/cp.bat_img_size.value()/3
        sp.arr.shape = (nptau, 3, sp.rows, sp.cols)
        logger.info('Set arr.shape = ' + str(sp.arr.shape), __name__)
        return sp.arr

#-----------------------------

    def get_img_array_for_dynamic_partition(sp) :
        logger.info('get_img_array_for_dynamic_partition: ' + sp.fname, __name__)
        #arr = mu + sigma*np.random.standard_normal(size=2400)

        #arr = np.arange(2400)
        arr = 100*np.random.standard_exponential(sp.size)
        arr.shape = (sp.rows,sp.cols)
        return arr

#-----------------------------

    def get_list_of_tau_from_file(sp, fname_tau) :
        #fname_tau = fnm.path_cora_merge_tau()
        logger.info('get_list_of_tau_from_file: ' + fname_tau, __name__)
        return np.loadtxt(fname_tau, dtype=np.uint16)

#-----------------------------

