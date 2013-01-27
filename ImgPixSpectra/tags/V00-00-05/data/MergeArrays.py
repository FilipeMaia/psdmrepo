#!/usr/bin/env python
import sys

import h5py
import numpy as np
import time

#---------------------------------------------------
def get_list_of_input_file_names() :
    """Returns the list of files with 2-d arrays which need to be summed."""

    list_of_files = [
    #'spv2-cxi49812-r0200-ev-000000-042962.np',
    #'spv2-cxi49812-r0201-ev-000000-015990.np',
    'spv2-cxi49812-r0202-ev-000000-040336.np',
    'spv2-cxi49812-r0203-ev-000000-010197.np',
    'spv2-cxi49812-r0204-ev-000000-066368.np',
    'spv2-cxi49812-r0205-ev-000000-032111.np',
    'spv2-cxi49812-r0206-ev-000000-073412.np',
    'spv2-cxi49812-r0207-ev-000000-029735.np']
    #'spv2-cxi49812-r0208-ev-000000-035901.np',
    #'spv2-cxi49812-r0209-ev-000000-027595.np',
    #'spv2-cxi49812-r0210-ev-000000-034038.np',
    #'spv2-cxi49812-r0211-ev-000000-015550.np',
    #'spv2-cxi49812-r0212-ev-000000-047923.np',
    #'spv2-cxi49812-r0213-ev-000000-022799.np',
    #'spv2-cxi49812-r0214-ev-000000-025638.np']

    print 'list_of_files = ', list_of_files
    return list_of_files

#---------------------------------------------------

def print_time_stamp() :
    """Prints the current local time.""" 
    tloc   = time.localtime(time.time())
    tstamp = time.strftime('%Y-%m-%d %H:%M',tloc)
    print 'Local time:', tstamp
    #tstart0= time.time()

#---------------------------------------------------

def spectra_merging(out_fname='sum-of-spectra.np') :
    """Merging of spectral arrays.

    1. Get the list of files for merging from get_list_of_input_file_names()
    2. Add the spectral 2d arrays together
    3. Save them in the file out_fname
    """

    list_of_files = get_list_of_input_file_names()
    print_time_stamp()
    spectra_sum = None

    for fname in list_of_files :
        spectra = np.fromfile(fname, dtype=np.int16)
        print 'Read spectrum from the file :', fname,  
        print ' shape :', spectra.shape  

        if  spectra_sum == None :
            spectra_sum  = spectra
        else :
            spectra_sum += spectra

    spectra_sum.tofile(out_fname)
 
    print 'The merged spectral array is saved in the file =', out_fname

#---------------------------------------------------

if __name__ == "__main__" :

    spectra_merging(out_fname = 'sum-2d-arr.np')
    sys.exit ( 'End of job' )

#---------------------------------------------------
