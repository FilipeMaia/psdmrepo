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
    phi = np.rad2deg(np.arctan2(y, x)) # arctan2 returns angle in range [-180,180]
    return r, phi

def cart2r(x, y) :
    return np.sqrt(x*x + y*y)

def cart2phi(x, y) :
    return np.rad2deg(np.arctan2(y, x)) # arctan2 returns angle in range [-180,180]

def polar2cart(r, phi) :
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def valueToIndex(V,VRange) :
    Vmin, Vmax, Nbins = VRange
    factor = float(Nbins) / float(Vmax-Vmin)
    return np.uint32( factor * (V-Vmin) )

def valueToIndexProtected(V,VRange) :
    Vmin, Vmax, Nbins = VRange
    Nbins1 = int(Nbins)-1
    factor = float(Nbins) / float(Vmax-Vmin)
    indarr = np.int32( factor * (V-Vmin) )
    #return np.select([V<Vmin,V>Vmax], [0,Nbins-1], default=indarr)
    return np.select([indarr<0, indarr>Nbins1], [0,Nbins1], default=indarr)

def q_map_partitions(map, nbins) :
    q_min = map.min()
    q_max = map.max()
    return valueToIndexProtected(map, [q_min, q_max, nbins])

def phi_map_partitions(map, nbins) :
    phi_min = -180.
    phi_max =  180.
    return valueToIndexProtected(map, [phi_min, phi_max, nbins])
 
#-----------------------------

class ViewResults :
    """First look at results.
    """

    def __init__(sp, fname=None) :
        """
        @param fname the file name with results
        """

        sp.x_map = None
        sp.y_map = None
        sp.r_map = None
        sp.q_map = None
        sp.phi_map = None

        sp.q_map_stat = None
        sp.phi_map_stat = None
        sp.q_phi_map_stat = None

        sp.q_map_dyna = None
        sp.phi_map_dyna = None
        sp.q_phi_map_dyna = None

        sp.counts_stat = None
        sp.counts_dyna = None
        sp.q_average_dyna = None

        sp.cor_arr = None
        sp.g2_vs_itau_arr = None
        sp.mask_total = None

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

        sp.ana_stat_meth_q   = cp.ana_stat_meth_q  .value()
        sp.ana_stat_meth_phi = cp.ana_stat_meth_phi.value()
        sp.ana_dyna_meth_q   = cp.ana_dyna_meth_q  .value()
        sp.ana_dyna_meth_phi = cp.ana_dyna_meth_phi.value()

        if  sp.ana_stat_meth_q   == cp.ana_stat_meth_q  .value_def() : # 'evenly_spaced'
            sp.ana_stat_part_q   = int(cp.ana_stat_part_q  .value())

        if  sp.ana_stat_meth_phi == cp.ana_stat_meth_phi.value_def() :
            sp.ana_stat_part_phi = int(cp.ana_stat_part_phi.value())                           

        if  sp.ana_dyna_meth_q   == cp.ana_dyna_meth_q  .value_def() :
            sp.ana_dyna_part_q   = int(cp.ana_dyna_part_q  .value())

        if  sp.ana_dyna_meth_phi == cp.ana_dyna_meth_phi.value_def() :
            sp.ana_dyna_part_phi = int(cp.ana_dyna_part_phi.value())

        sp.npart_stat = sp.ana_stat_part_q * sp.ana_stat_part_phi
        sp.npart_dyna = sp.ana_dyna_part_q * sp.ana_dyna_part_phi
        


#-----------------------------

    def evaluate_parameters(sp) :
        sp.wavelength   = 1.23984/sp.photon_energy # 1.23984 ? [nm]
        sp.factor       = 4*(math.pi/sp.wavelength)
        sp.distance_pix = sp.distance / sp.ccd_pixsize 
        
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
                                                                   
        logger.info('ana_stat_meth_q   = ' + str(sp.ana_stat_meth_q  ), __name__)
        logger.info('ana_stat_meth_phi = ' + str(sp.ana_stat_meth_phi), __name__)
        logger.info('ana_dyna_meth_q   = ' + str(sp.ana_dyna_meth_q  ), __name__)
        logger.info('ana_dyna_meth_phi = ' + str(sp.ana_dyna_meth_phi), __name__)
                                                                                                        
        logger.info('ana_stat_part_q   = ' + str(sp.ana_stat_part_q  ), __name__)
        logger.info('ana_stat_part_phi = ' + str(sp.ana_stat_part_phi), __name__)
        logger.info('ana_dyna_part_q   = ' + str(sp.ana_dyna_part_q  ), __name__)
        logger.info('ana_dyna_part_phi = ' + str(sp.ana_dyna_part_phi), __name__)

        logger.info('npart_stat        = ' + str(sp.npart_stat   ), __name__)
        logger.info('npart_dyna        = ' + str(sp.npart_dyna   ), __name__)
                                                                   
        logger.info('Evaluated parameters:', __name__)
        logger.info('sp.wavelength   = ' + str(sp.wavelength     ), __name__)
        logger.info('sp.factor       = ' + str(sp.factor         ), __name__)
#        logger.info('' + str(), __name__)

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

        sp.x_map, sp.y_map = sp.get_xy_maps()

#-----------------------------

    def get_xy_maps(sp) :
        """Set map x, y for direct beam or reflected beam modes"""
        # MAKE HERE SELECTION OF THE X,Y MAPS FOR MODE !!!
        if sp.x_map != None and sp.y_map != None : return sp.x_map, sp.y_map 
        x_map, y_map = sp.get_xy_maps_for_direct_beam_data()
        #sp.x_map, sp.y_map = sp.get_xy_maps_for_reflected_beam_data()
        return x_map, y_map

#-----------------------------

    def get_xy_maps_for_direct_beam_data(sp) :
        x_db_pix = sp.x_coord_beam0 + (sp.x0_pos_in_data - sp.x0_pos_in_beam0) / sp.ccd_pixsize
        y_db_pix = sp.y_coord_beam0 + (sp.y0_pos_in_data - sp.y0_pos_in_beam0) / sp.ccd_pixsize
        return sp.X_ccd_pix - x_db_pix, sp.Y_ccd_pix - y_db_pix  

#-----------------------------

    def get_x_map(sp) :
        return sp.x_map

#-----------------------------

    def get_y_map(sp) :
        return sp.y_map

#-----------------------------

    def get_rphi_maps(sp) :
        if sp.r_map != None and sp.phi_map != None : return sp.r_map, sp.phi_map 
        sp.r_map, sp.phi_map = cart2polar(sp.x_map, sp.y_map)
        return sp.r_map, sp.phi_map
  
#-----------------------------

    def get_r_map(sp) :
        if sp.r_map != None : return sp.r_map
        sp.r_map = cart2r(sp.x_map, sp.y_map)
        return sp.r_map
  
#-----------------------------

    def get_q_map(sp) :
        if sp.q_map != None : return sp.q_map
        r_map = sp.get_r_map()
        sp.q_map = sp.factor * np.sin(0.5*np.arctan2(r_map, sp.distance_pix))
        return sp.q_map
  
#-----------------------------

    def get_phi_map(sp) :
        if sp.phi_map != None : return sp.phi_map
        sp.phi_map = cart2phi(sp.x_map, sp.y_map)
        return sp.phi_map
  
#-----------------------------

    def get_q_map_for_stat_bins(sp) :
        if sp.q_map_stat != None : return sp.q_map_stat
        sp.q_map_stat = q_map_partitions(sp.get_q_map(), sp.ana_stat_part_q)
        return sp.q_map_stat


    def get_phi_map_for_stat_bins(sp) :
        if sp.phi_map_stat != None : return sp.phi_map_stat
        sp.phi_map_stat = phi_map_partitions(sp.get_phi_map(), sp.ana_stat_part_phi)
        return sp.phi_map_stat


    def get_q_phi_map_for_stat_bins(sp) :
        if sp.q_phi_map_stat != None : return sp.q_phi_map_stat
        sp.q_phi_map_stat = sp.get_q_map_for_stat_bins() * sp.ana_stat_part_phi \
                          + sp.get_phi_map_for_stat_bins()#* sp.ana_stat_part_q 
        return sp.q_phi_map_stat


    def get_counts_for_stat_bins(sp) :
        if sp.counts_stat != None : return sp.counts_stat
        weights = sp.get_mask_total() # Apply mask for bin counts
        sp.counts_stat = sp.bincount(sp.get_q_phi_map_for_stat_bins(), weights, length=sp.npart_stat)
        return sp.counts_stat

#-----------------------------

    def get_q_map_for_dyna_bins(sp) :
        if sp.q_map_dyna != None : return sp.q_map_dyna
        sp.q_map_dyna = q_map_partitions(sp.get_q_map(), sp.ana_dyna_part_q)
        return sp.q_map_dyna


    def get_phi_map_for_dyna_bins(sp) :
        if sp.phi_map_dyna != None : return sp.phi_map_dyna
        sp.phi_map_dyna = phi_map_partitions(sp.get_phi_map(), sp.ana_dyna_part_phi)
        return sp.phi_map_dyna

  
    def get_q_phi_map_for_dyna_bins(sp) :
        if sp.q_phi_map_dyna != None : return sp.q_phi_map_dyna
        sp.q_phi_map_dyna = sp.get_q_map_for_dyna_bins() * sp.ana_dyna_part_phi \
                          + sp.get_phi_map_for_dyna_bins() # * sp.ana_dyna_part_q
        return sp.q_phi_map_dyna


    def get_counts_for_dyna_bins(sp) :
        if sp.counts_dyna != None : return sp.counts_dyna
        weights = sp.get_mask_total() # Apply mask for bin counts
        sp.counts_dyna = sp.bincount(sp.get_q_phi_map_for_dyna_bins(), weights, length=sp.npart_dyna)
        return sp.counts_dyna


    def get_q_average_for_dyna_bins(sp) :
        if sp.q_average_dyna != None : return sp.q_average_dyna

        q_map_masked      = sp.get_q_map() * sp.get_mask_total()
        sum_q_dyna        = sp.bincount(sp.get_q_phi_map_for_dyna_bins(), q_map_masked, length=sp.npart_dyna)
        counts_dyna       = sp.get_counts_for_dyna_bins()
        counts_dyna_prot  = np.select([counts_dyna<=0.], [-1.], counts_dyna)
        sp.q_average_dyna = np.select([counts_dyna_prot<=0.], [0.], default=sum_q_dyna/counts_dyna)
        print 'get_q_average_for_dyna_bins():\n', sp.q_average_dyna
        return sp.q_average_dyna

#-----------------------------

    def get_1oIp_map_for_stat_bins_itau(sp, itau) :
        sp.Ip_normf_map = sp.get_norm_factor_map_masked_for_stat_bins_itau(sp.get_Ip_for_itau(itau))
        return sp.Ip_normf_map

    def get_1oIf_map_for_stat_bins_itau(sp, itau) :
        sp.If_normf_map = sp.get_norm_factor_map_masked_for_stat_bins_itau(sp.get_If_for_itau(itau))
        return sp.If_normf_map

#-----------------------------

    def get_norm_factor_map_masked_for_stat_bins_itau(sp, intens_map) :
        """Apply mask to the input and output mask to get correct normalization and output, respectively."""
        intens_map_masked = intens_map * sp.get_mask_total() # Apply mask for bin counts
        norm_factor_map = sp.get_norm_factor_map_for_stat_bins_itau(intens_map_masked)
        return norm_factor_map * sp.get_mask_total() # Apply mask for intensity map

#-----------------------------

    def get_norm_factor_map_for_stat_bins_itau(sp, intens_map) :
        q_phi_map_stat = sp.get_q_phi_map_for_stat_bins()
        counts = sp.get_counts_for_stat_bins()

        intens = sp.bincount(q_phi_map_stat, intens_map, sp.npart_stat)
        intens_prot = np.select([intens<=0.], [-1.], default=intens)
        normf = np.select([intens_prot<=0.], [0.], default=counts/intens_prot)

        #norm_facotr_map = np.choose(q_phi_map_stat, normf, mode='clip') # DOES NOT WORK!
        #norm_facotr_map = q_phi_map_stat.choose(normf, mode='clip')     # DOES NOT WORK!
        #norm_facotr_map = np.array(map(lambda i : normf[i], q_phi_map_stat)) # 0.26sec
        norm_facotr_map = np.array([normf[i] for i in q_phi_map_stat]) # WORKS! # 0.24sec
        norm_facotr_map.shape = (sp.rows,sp.cols)        
        return norm_facotr_map # sp.get_random_img()

#-----------------------------

    def get_g2_map_for_itau(sp, itau) :
        Ip_normf_map = sp.get_1oIp_map_for_stat_bins_itau(itau)
        If_normf_map = sp.get_1oIp_map_for_stat_bins_itau(itau)
        I2_map       = sp.get_I2_for_itau(itau)
        sp.g2_map = I2_map * Ip_normf_map * If_normf_map # mask is already applied to normf
        return sp.g2_map


    def get_g2_for_dyna_bins_itau(sp, itau) :
        q_phi_map_dyna = sp.get_q_phi_map_for_dyna_bins()
        g2_map         = sp.get_g2_map_for_itau(itau)
        intens_dyna    = sp.bincount(q_phi_map_dyna, g2_map, sp.npart_dyna)
        counts         = sp.get_counts_for_dyna_bins()
        counts_prot    = np.select([counts==0], [-1], default=counts) 
        sp.g2_for_dyna_bins = np.select([counts_prot<0], [0], default=intens_dyna/counts_prot)
        return sp.g2_for_dyna_bins


    def get_g2_map_for_dyna_bins_itau(sp, itau) :
        q_phi_map_dyna          = sp.get_q_phi_map_for_dyna_bins()
        g2_for_dyna_bins        = sp.get_g2_for_dyna_bins_itau(itau)
        sp.g2_map_for_dyna_bins = np.array([g2_for_dyna_bins[i] for i in q_phi_map_dyna])
        sp.g2_map_for_dyna_bins*= sp.get_mask_total() # Apply mask to the map
        return sp.g2_map_for_dyna_bins
        #return sp.get_random_img()


    def get_g2_vs_itau_arr(sp) :
        if sp.g2_vs_itau_arr != None : return sp.g2_vs_itau_arr        
        sp.list_of_tau = sp.get_list_of_tau_from_file(fnm.path_cora_merge_tau())
        #print 'sp.list_of_tau = ', sp.list_of_tau

        g2_vs_itau = []
        for itau, tau in enumerate(sp.list_of_tau) :
            g2_for_dyna_bins = sp.get_g2_for_dyna_bins_itau(itau)
            g2_vs_itau.append(g2_for_dyna_bins)

            msg = 'get_g2_vs_itau_arr: itau=' + str(itau) + \
                  '  tau='                    + str(tau) + \
                  '  <g2>='                   + str(g2_for_dyna_bins.mean()) 
            logger.info(msg, __name__)
            print msg
        
        sp.g2_vs_itau_arr = np.array(g2_vs_itau)

        return sp.g2_vs_itau_arr

#-----------------------------

    def bincount(sp, map_bins, map_weights=None, length=None) :
        if map_weights == None : weights = None
        else                   : weights = map_weights.flatten() 

        bin_count = np.bincount(map_bins.flatten(), weights, length)
        #print 'bin_count:\n',      bin_count
        #print 'bin_count.shape =', bin_count.shape
        return bin_count

#-----------------------------

    def set_file_name(sp, fname=None) :
        sp.cor_arr = None
        sp.g2_vs_itau_arr = None

        if fname == None : sp.fname = cp.res_fname.value()
        else :             sp.fname = fname

#-----------------------------

    def get_cor_array_from_text_file(sp) :
        logger.info('get_cor_array_from_text_file: ' + sp.fname, __name__)
        #return np.loadtxt(fname, dtype=np.float32)


    def get_cor_array_from_binary_file(sp) :
        if sp.cor_arr != None : return sp.cor_arr
        logger.info('get_cor_array_from_binary_file: ' + sp.fname, __name__)

        sp.cor_arr = np.fromfile(sp.fname, dtype=np.float32)
        nptau = sp.cor_arr.shape[0]/cp.bat_img_size.value()/3
        sp.cor_arr.shape = (nptau, 3, sp.rows, sp.cols)
        logger.info('Set arr.shape = ' + str(sp.cor_arr.shape), __name__)
        return sp.cor_arr

#-----------------------------

    def get_Ip_for_itau(sp,itau) :
        cor_array = sp.get_cor_array_from_binary_file()
        return cor_array[itau, 0,...]

#-----------------------------

    def get_If_for_itau(sp,itau) :
        cor_array = sp.get_cor_array_from_binary_file()
        return cor_array[itau, 1,...]

#-----------------------------

    def get_I2_for_itau(sp,itau) :
        cor_array = sp.get_cor_array_from_binary_file()
        return cor_array[itau, 2,...]

#-----------------------------

    def get_g2_raw_for_itau(sp,itau) :
        """Returns g2 raw map without normalization for preliminary plot"""
        cor_arr = sp.get_cor_array_from_binary_file()
        Ip = cor_arr[itau, 0,...] 
        If = cor_arr[itau, 1,...] 
        I2 = cor_arr[itau, 2,...] 
        sp.g2_raw_for_itau = I2/Ip/If
        return sp.g2_raw_for_itau

#-----------------------------

    def get_random_img(sp) :
        logger.info('get_random_img(): standard_exponential', __name__)
        #arr = mu + sigma*np.random.standard_normal(size=2400)
        #arr = np.arange(2400)
        sp.arr2d = 100*np.random.standard_exponential(sp.size)
        sp.arr2d.shape = (sp.rows,sp.cols)
        return sp.arr2d


    def get_random_binomial_img(sp) :
        logger.info('get_random_binomial_img()', __name__)
        ntrials = 1
        p = 0.99 # probability of success (1)
        sp.arr2d = np.random.binomial(ntrials, p, sp.size)
        sp.arr2d.shape = (sp.rows,sp.cols)
        return sp.arr2d

#-----------------------------

    def get_list_of_tau_from_file(sp, fname_tau) :
        #fname_tau = fnm.path_cora_merge_tau()
        logger.info('get_list_of_tau_from_file: ' + fname_tau, __name__)
        return np.loadtxt(fname_tau, dtype=np.uint16)

#-----------------------------
#--------- MASKS -------------
#-----------------------------

    def get_mask_image_limits(sp) :
        cols_span = sp.col_end - sp.col_begin
        rows_span = sp.row_end - sp.row_begin
        arr1 = np.ones((rows_span, cols_span), dtype=np.uint8)
        sp.mask_image_limits = np.zeros((sp.rows,sp.cols), dtype=np.uint8)
        sp.mask_image_limits[sp.row_begin:sp.row_begin+rows_span, sp.col_begin:sp.col_begin+cols_span] += arr1[0:rows_span, 0:cols_span]
        return sp.mask_image_limits
        #return sp.get_random_img()


    def get_mask_blemish(sp) :
        sp.mask_blemish = gu.get_array_from_file(fnm.path_blem())
        return sp.mask_blemish


    def get_mask_hotpix(sp) :
        #sp.mask_hotpix = gu.get_array_from_file(fnm.path_hotpix())
        #return sp.mask_hotpix
        print 'get_mask_hotpix IS NOT IMPLEMENTED YET! get random binomial for now...'
        #return sp.get_random_img()
        return sp.get_random_binomial_img()


    def get_mask_total(sp) :
        if sp.mask_total != None : return sp.mask_total

        sp.mask_total = sp.get_mask_image_limits()
        if cp.ccdcorr_blemish.value() :
            sp.mask_total *= sp.get_mask_blemish()
 
        return sp.mask_total

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

