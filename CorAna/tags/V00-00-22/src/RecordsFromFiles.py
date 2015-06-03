#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module RecordsFromFiles...
#
#------------------------------------------------------------------------

"""Access to the intensity records from the files

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
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
import numpy as np

from   Logger  import logger
import GlobalUtils as gu
from FileNameManager  import fnm
from ConfigParametersCorAna   import confpars as cp
from EventTimeRecords import *

#-----------------------------

class RecordsFromFiles :
    """Access to the time records from the file
    """
    rows            = cp.bat_img_rows.value()
    cols            = cp.bat_img_cols.value()
    size            = cp.bat_img_size.value()
 
    fname_time             = fnm.path_cora_split_time_ind()   
    fname_int_stat_q       = fnm.path_cora_split_int_static_q()
    fname_q_ave_static     = fnm.path_cora_split_q_ave_static()
    fname_imon             = fnm.path_data_scan_monitors_data()
    fname_imon_cfg         = fnm.path_cora_split_imon_cfg()
    fname_data_ave         = fnm.path_data_ave()

#-----------------------------

    def __init__ (self) :
        """Constructor.
        @param fname the file name for ...
        """

        #self.intens_stat_q_bins_vs_t = None
        #self.time_arr = None

        pass

#-----------------------------

    def get_time_records_arr(self) :
        etr = EventTimeRecords(self.fname_time)
        logger.info('Get time records array from file: ' + self.fname_time, __name__)
        self.time_arr = etr.get_arr_for_plot()
        return self.time_arr

    def print_time_records_arr(self) :
        msg = 'time records: ind_ev, t, dt, ind_t:\n'
        for rec in self.time_arr : msg += str(rec) + '\n'
        logger.info(msg, __name__) 
        print msg

#-----------------------------

    def get_q_ave_for_stat_q_bins(self) :
        return gu.get_array_from_file(self.fname_q_ave_static)

#-----------------------------

    def get_intens_stat_q_bins_arr(self) :
        """Returns <I>(t, q-static) 2D array"""
        #if self.intens_stat_q_bins_vs_t is not None : return self.intens_stat_q_bins_vs_t

        if not os.path.exists(self.fname_int_stat_q) :
            msg = 'The file: %s is not available' % (self.fname_int_stat_q)
            gu.confirm_dialog_box(cp.guimain, msg)
            return None

        self.intens_stat_q_bins_vs_t = gu.get_array_from_file(self.fname_int_stat_q)[:,1:] # trim column with event number

        if self.intens_stat_q_bins_vs_t is not None :
            logger.info('I(t,q-stat) is taken from file ' + fnm.path_cora_split_int_static_q(), __name__)
            return self.intens_stat_q_bins_vs_t
        else :
            msg = 'I(t,q-stat) file: %s is not available' % (fnm.path_cora_split_int_static_q())
            msg+= '\nTo produce this file: \n1) in GUI "View Results", click on button "q stat"'\
                + '\n2) in GUI "Run/Split" click on button "Run" and repeat processing.'
            logger.info(msg, __name__)
            gu.confirm_dialog_box(cp.guimain, msg)
            return np.ones((self.rows,self.cols), dtype=np.uint8)


    def print_intens_stat_q_bins_arr(self) :
        msg = 'Intensity in static q bins vs event:\n'
        for rec in self.intens_stat_q_bins_vs_t :
            msg += '%10.3f  %10.3f  %10.3f  %10.3f ...\n' % \
                   (rec[0], rec[1], rec[2], rec[3])
        logger.info(msg, __name__) 
        print msg


    def get_intens_stat_q_bins(self) :
        """Returns <I>(q-static) averaged over all events"""
        arr = self.get_intens_stat_q_bins_arr()
        nevts = arr.shape[0]
        self.intens_stat_q_bins_aver = np.sum(arr,axis=0) / nevts
        return self.intens_stat_q_bins_aver


    def print_intens_stat_q_bins(self) :
        nevts = self.intens_stat_q_bins_vs_t.shape[0]
        msg = '<I>(q-static) averaged over %d events:\n' % nevts 
        msg += str(self.intens_stat_q_bins_aver ) + '\n' 
        logger.info(msg, __name__) 
        print msg

#-----------------------------

# BldInfo(FEEGasDetEnergy)         FEEGasDetEnergy  1 1 1 1   0 0     -1.0000   -1.0000    1.0000
# BldInfo(XCS-IPM-02)              XCS-IPM-02       1 1 1 1   0 0     -1.0000   -1.0000    1.0000
# BldInfo(XCS-IPM-mono)            XCS-IPM-mono     1 1 1 1   0 1      0.0200    0.5000    0.2600
# DetInfo(XcsBeamline.1:Ipimb.4)   Ipimb.4          1 1 1 1   0 0     -1.0000   -1.0000    1.0000
# DetInfo(XcsBeamline.1:Ipimb.5)   Ipimb.5          1 1 1 1   0 0     -1.0000   -1.0000    1.0000

    def values_from_rec(self, rec) :
        return str(rec[0]), str(rec[1]), int(rec[2]), int(rec[3]), int(rec[4]), int(rec[5]), int(rec[6]), int(rec[7]), \
               float(rec[8]), float(rec[9]), float(rec[10])  


    def get_imon_cfg_pars(self) :
        logger.info('get_imon_cfg_pars', __name__)
        list_recs = gu.get_text_list_from_file(self.fname_imon_cfg)
        if list_recs is None :
            self.list_of_imon_cfg_pars = None 
            return

        self.list_of_imon_cfg_pars = map(self.values_from_rec, list_recs)
        return self.list_of_imon_cfg_pars
        

    def print_imon_cfg_pars(self) :
        msg = 'Parameters from imon config file: ' + self.fname_imon_cfg + '\n'
        for rec in self.get_imon_cfg_pars() : msg += str(rec) + '\n' 
        logger.info(msg, __name__) 
        print msg


    def listMaskForIMonChannels(self,imon):
        list_of_recs = self.get_imon_cfg_pars()
        rec = list_of_recs[imon]
        return rec[2:6]


    def nparrMaskForIMonChannels(self,imon):        
        return np.array(self.listMaskForIMonChannels(imon),dtype=int)


    def printMaskForIMonChannels(self,imon):
        msg = 'Mask for imon %d channels: ' % (imon) + str(self.nparrMaskForIMonChannels(imon))
        logger.info(msg, __name__) 
        print msg

#-----------------------------

    def getIMonArray(self,imon):
        logger.info('getIMonArray for imon %d '%(imon), __name__)
        arr_all = gu.get_array_from_file(self.fname_imon)
        if arr_all is None : return None
        logger.info('Array shape: ' + str(arr_all.shape), __name__)

        ibase    = 1+imon*4
        arr_imon = arr_all[:,ibase:ibase+4]
        #print 'arr_imon:\n', arr_imon
        #print 'arr_imon.shape:', arr_imon.shape

        #mask = self.maskForIMonChannels(imon)
        #npmask = np.array(mask,dtype=float)
        npmask = self.nparrMaskForIMonChannels(imon)

        size   = arr_imon.shape[0]
        npcol1 = np.ones(size)

        X,Y = np.meshgrid(npmask,npcol1)
        arr_prod = (arr_imon * X)        
        arr_sum  = arr_prod.sum(1) 
        
        #print 'npmask=', npmask
        #print 'size=', size
        #print X
        #print X.shape
        #print arr_imon
        #print arr_imon.shape
        #print arr_prod
        #print arr_prod.shape
        return arr_sum


    def printIMonArray(self,imon):    
        msg = 'Sum of intensity monitor channels for imon %d '%(imon) +'\n'
        msg += str(self.getIMonArray(imon))
        print msg

#-----------------------------

    def get_data_ave_array(self) :    
        arr = gu.get_array_from_file(self.fname_data_ave)
        if arr is None : return np.ones((self.rows,self.cols), dtype=np.uint8)
        return arr

#-----------------------------
#-----------------------------
# Singleton?
rff = RecordsFromFiles()

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    #rff = RecordsFromFiles()

    #arr_t = rff.get_time_records_arr()
    #rff.print_time_records_arr()

    arr_i     = rff.get_intens_stat_q_bins_arr()
    arr_i_ave = rff.get_intens_stat_q_bins()
    rff.print_intens_stat_q_bins_arr()
    rff.print_intens_stat_q_bins()
    arr_imon_cfg = rff.get_imon_cfg_pars()
    rff.print_imon_cfg_pars()
    rff.printMaskForIMonChannels(2)
    rff.printIMonArray(2)

    sys.exit ( 'End of test for RecordsFromFiles' )

#-----------------------------
