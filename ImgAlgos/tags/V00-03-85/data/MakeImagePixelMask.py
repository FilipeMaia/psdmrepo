#!/usr/bin/env python
#--------------------

import numpy as np
import sys
import os
#import matplotlib.pyplot as plt
#import h5py

from optparse import OptionParser

#--------------------

def get_input_parameters() :

    def_infname = 'xxx/cspad-cxi49012-r0025-background-ave.dat'
    def_oufname = 'mask.dat'
    def_cols    = 1024;
    def_rows    = 1024;

    parser = OptionParser(description='Process optional input parameters.', usage = "usage: %prog [options]")
    parser.add_option('-f', '--infile', dest='infname', default=def_infname, action='store', type='string', help='input file name')
    parser.add_option('-o', '--oufile', dest='oufname', default=def_oufname, action='store', type='string', help='output file name')
    parser.add_option('-c', '--cols', dest='cols', default=def_cols, action='store', type='int', help='number of columns in the image array')
    parser.add_option('-r', '--rows', dest='rows', default=def_rows, action='store', type='int', help='number of rows in the image array')
    parser.add_option('-v', dest='verbose', action='store_true',  help='set flag to print more details',  default=True)
    parser.add_option('-q', dest='verbose', action='store_false', help='set flag to print less details')
    (opts, args) = parser.parse_args()

    print 'opts:',opts
    print 'args:',args

    return (opts, args)
    
#--------------------

def get_array_from_file(fname) :
    print 'get_array_from_file:', fname
    return np.loadtxt(fname, dtype=np.float32)

#--------------------

def save_array_in_file(arr, fname='mask.dat') :
    print 'save_array_in_file:', fname
    print 'arr:\n', arr
    print 'arr.shape:', arr.shape
    np.savetxt(fname, arr, fmt='%d', delimiter=' ') # New option in version 1.5.0. newline='\n') 
    #np.savetxt(fname, arr, fmt='%3.1f', delimiter=' ') # New option in version 1.5.0. newline='\n') 

#--------------------

def save_array_in_file_my(arr, fname='mask.dat') :
    print 'save_array_in_file:', fname
    print 'arr:\n', arr
    print 'arr.shape:', arr.shape
    rows,cols = arr.shape

    space = ' '
    f=open(fname,'w')
    for r in range(rows) :
        for c in range(cols) :
            f.write(space+str(arr[r,c]))
        f.write(space+'\n') # THIS SPACE IS IMPORTANT: the C++ stringstream skips last value without this space
    f.close()

#--------------------

def check_array_in_file(fname='mask.dat') :
    print 'check_array_in_file:', fname
    arr = get_array_from_file(fname)
    print 'arr:\n', arr
    print 'arr.shape:', arr.shape

#--------------------

def evaluate_mask_array(arr,threshold=30) :
    print 'evaluate_mask_array(...)'
    print 'Threshold: ', threshold
    print 'Input array:\n'
    print arr
    rows,cols = shape = arr.shape
    print 'shape, rows, cols=', shape, rows, cols
    #arr_mask = np.ones(shape, dtype=np.uint16) 
    #arr_mask = np.select([arr>threshold],[0], default=arr_mask)
    arr_mask = np.select([arr>threshold],[0], default=[1])
    #theta0 = np.select([theta<0, theta>=0],[theta+360,theta]) #[0,360]

    num_zeros, num_ones = np.bincount(arr_mask.flatten())

    print 'Total number of masked pixels is', num_zeros, 'of', arr.size, 'fraction of masked: %6.4f' % ( float(num_zeros)/arr.size )
    return arr_mask

#--------------------

def mask_rect(arr, row0=10, rowN=200, col0=0, colN=400) :
    print 'In array with shape:',arr.shape, 'the rect will be masked for rows:', row0, rowN, 'and cols:', col0, colN
    rect = np.zeros( (rowN-row0,colN-col0), dtype=np.int16 )
    arr[row0:rowN,col0:colN] = rect
    return arr

#--------------------

def do_main() :

    (opts, args) = get_input_parameters()

    #1. Generate transparent mask of units
    arr_mask = np.ones( (opts.rows, opts.cols), dtype=np.int16 )

    #2. Mask high background regions
    #arr      = get_array_from_file(opts.infname)
    #arr_mask = evaluate_mask_array(arr,threshold)

    #3. Mask rectangular region(s)  
    arr_mask = mask_rect(arr_mask, row0=100, rowN=200, col0=200, colN=400)    

    save_array_in_file(arr_mask,fname=opts.oufname)
    #save_array_in_file_my(arr_mask,fname=opts.oufname)

    check_array_in_file(opts.oufname)

#--------------------

if __name__ == '__main__' :

    do_main()

    sys.exit('The End')

#--------------------
