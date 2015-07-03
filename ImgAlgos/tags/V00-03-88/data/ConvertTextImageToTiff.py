#!/usr/bin/env python
#--------------------

import os
import sys
import numpy as np
from optparse import OptionParser
#import scipy.misc as scimisc
import Image

#--------------------

def get_input_parameters() :

    def_fnpre = 'cspad2x2'
    def_dname = '.'
    def_fnext = '.dat'

    parser = OptionParser(description='Process optional input parameters.', usage = "usage: %prog [options]")
    parser.add_option('-f', '--fnpre', dest='fnpre', default=def_fnpre, action='store', type='string', help='input file name prefix, default: %s' % def_fnpre)
    parser.add_option('-e', '--fnext', dest='fnext', default=def_fnext, action='store', type='string', help='input file name extension, default: %s' % def_fnext) 
    parser.add_option('-d', '--dname', dest='dname', default=def_dname, action='store', type='string', help='input/output directory name, default: %s' % def_dname)

    (opts, args) = parser.parse_args()
    print 'opts:',opts
    #print 'args:',args
    return (opts, args)

#--------------------

def save_array_in_file(ofname, arr, dtype=np.int16) : # np.float32
    #scimisc.imsave(ofname, arr) # saves as uint8
    img = Image.fromarray(arr.astype(dtype))  # or int16, int32, float32
    tfile = 'tmp_img.tiff'
    img.save(ofname)

    #img.save(tfile)
    #cmd = 'convert -depth 32 -define quantum:format=floating-point %s %s' % (tfile, ofname) \
    #      if dtype == np.float32 else 'convert %s -define quantum:format=signed %s' % (tfile, ofname)

    #os.system(cmd)
  
#--------------------

def get_array_from_file(fname, dtype=np.float32) :
    #print 'get_array_from_file:', fname
    return np.loadtxt(fname, dtype=dtype)

#--------------------

def get_list_of_files_to_convert(dirname, fnpre='cspad2x2', fnext='.dat') :
    list_of_files = []
    for fname in os.listdir(dirname) :
        if fname.split('-')[0] == fnpre and os.path.splitext(fname)[1] == fnext :
            list_of_files.append(fname)
    return  list_of_files

#--------------------

def do_main() :

    opts, args = get_input_parameters()
    fnpre = opts.fnpre
    fnext = opts.fnext
    dname = opts.dname

    for fname in get_list_of_files_to_convert(dname, fnpre, fnext) :
        ifpath = os.path.join(dname, fname)       
        ofpath = os.path.join(dname, os.path.splitext(fname)[0] + '.tiff')
        print 'Convert ' + ifpath + ' ===> ' + ofpath
        dtype = np.float32 # np.int16
        dtype = np.float32 # np.int16
        arr = get_array_from_file(ifpath, dtype)
        save_array_in_file(ofpath, arr, dtype)
        #print 'arr=\n', arr
  
#--------------------

if __name__ == '__main__' :
    do_main()
    sys.exit('The End')

#--------------------
