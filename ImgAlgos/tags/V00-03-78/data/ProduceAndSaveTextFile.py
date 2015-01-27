#!/usr/bin/env python
#--------------------

import os
import sys
import numpy as np

#--------------------

def get_array_from_file(fname, dtype=np.float32) :
    print 'get_array_from_text_file:', fname
    arr = np.loadtxt(fname, dtype=dtype)
    print 'arr.shape=', arr.shape
    return arr

def save_array_in_text_file(fname, arr, fmt='%f') :
    np.savetxt(fname, arr, fmt, delimiter=' ')


def get_array_from_bin_file(fname, dtype=np.float32) :
    print 'get_array_from_bin_file:', fname
    return np.fromfile(fname, dtype)


def get_numpy_array_from_file(fname) :
    print 'get_numpy_array_from_file:', fname
    return np.load(fname)


def save_image_array_in_file(fname,arr) :
    img = Image.fromarray(arr.astype(np.int16))
    img.save(fname)


def save_numpy_array_in_file(fname,arr) :
    np.save(fname,arr)


#--------------------

def get_input_parameters() :

    fname_def = 'array.txt'

    nargs = len(sys.argv)
    print 'sys.argv: ', sys.argv
    print 'nargs: ', nargs

    if nargs == 1 :
        print 'Will use all default parameters\n',\
              'Expected command: ' + sys.argv[0] + ' <infname>' 
        #sys.exit('CHECK INPUT PARAMETERS!')
    fname = fname_def

    if nargs  > 1 : fname = sys.argv[1]
    else          : fname = fname_def

    if nargs  > 2 :         
        print 'WARNING: Too many input arguments! Exit program.\n'
        sys.exit('CHECK INPUT PARAMETERS!')

    print 'Input file name  :', fname
 
    return fname

#--------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    #arr = mu + sigma*np.random.standard_normal(size=2400)
    arr = 100*np.random.standard_exponential(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)
    return arr


def get_array(fname) :

    #return 0.5 * np.ones((4*512,512), dtype=np.float32)

    arr = 0.001*np.arange(4*512*512)
    arr.shape = (4*512,512)
    return arr


#--------------------

def do_main() :

    fname = get_input_parameters()
    arr = get_array(fname)

    print 'arr:\n', arr
    print 'arr.shape=', arr.shape

    save_array_in_text_file(fname, arr)

#--------------------

if __name__ == '__main__' :
    do_main()
    sys.exit('The End')

#--------------------
