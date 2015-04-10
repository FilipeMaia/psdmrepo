#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module EventTimeRecords...
#
#------------------------------------------------------------------------

"""Access to the time records from the file

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

#-----------------------------

class EventTimeRecords :
    """Access to the time records from the file
    """

    def __init__ (self, fname) :
        """Constructor.
        @param fname the file name for ...
        """
        self.fname = fname
        msg = 'Get time records from the file: ' + self.fname
        logger.info(msg, __name__) 

        self.get_time_record_list_from_file()
        #self.print_records_all()
        #self.print_records_for_plot()
        #self.print_arr_for_plot()


    def values_from_rec(self, rec) :
        return int(rec[0]), float(rec[1]), float(rec[2]), str(rec[3]), int(rec[4]), int(rec[5]), int(rec[6])


    def values_for_plot(self, rec) :
        return rec[0], rec[1], rec[2], rec[6]


    def get_time_record_list_from_file(self):
        logger.debug('on_but_tspl', __name__)
        list_recs = gu.get_text_list_from_file(self.fname)
        if list_recs is None :
            self.list_vals = None 
            return
        #list_recs[1][:] = ['1', '8.026429', '8.026429', '20120616-080244.698036743', '8255', '1', '1']
        self.list_vals_all      = map(self.values_from_rec, list_recs)
        self.list_vals_for_plot = map(self.values_for_plot, self.list_vals_all)
        self.arr_for_plot       = np.array(self.list_vals_for_plot)


    def print_records_all(self):
        #logger.debug('Array shape: ' + str(self.arr.shape), __name__)

        for rec in self.list_vals_all :
            print rec


    def print_records_for_plot(self):
        #logger.debug('Array shape: ' + str(self.arr.shape), __name__)

        for rec in self.list_vals_for_plot :
            print rec


    def print_arr_for_plot(self):
        #logger.debug('Array shape: ' + str(self.arr_for_plot.shape), __name__)
        print 'self.arr_for_plot.shape:', self.arr_for_plot.shape
        #print 'self.arr_for_plot:\n',  self.arr_for_plot
        print 'ind_ev:\n', self.arr_for_plot[:,0]
        print 't:\n',      self.arr_for_plot[:,1]
        print 'dt:\n',     self.arr_for_plot[:,2]
        print 'ind_t:\n',  self.arr_for_plot[:,3]
        #print 'Equivalent without numpy but with loop in the list of comperhensive...: t=\n', \
        #[v[1] for v in list_vals_for_plot]


    def get_arr_for_plot(self) :
        return self.arr_for_plot

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    #etr = EventTimeRecords ('work/cora-xcsi0112-r0015-data-scan-tstamp-list.txt')
    etr = EventTimeRecords ('/reg/neh/home1/dubrovin/LCLS/PSANA-V01/work-1/t1-xcsi0112-r0015-data-scan-tstamp-list.txt')
    arr = etr.get_arr_for_plot()
    print 'Array for plot of shape; ', arr.shape, '\n', arr

    sys.exit ( 'End of test for EventTimeRecords' )

#-----------------------------
