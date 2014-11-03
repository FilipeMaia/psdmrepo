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

    def_fname = 'cspad2x2.1'
    def_dname = '.'

    parser = OptionParser(description='Process optional input parameters.', usage = "usage: %prog [options]")
    parser.add_option('-f', '--fname', dest='fname', default=def_fname, action='store', type='string', help='input file name prefix, i.e. cspad2x2.1')
    parser.add_option('-d', '--dname', dest='dname', default=def_dname, action='store', type='string', help='input/output directory name')
    (opts, args) = parser.parse_args()
    print 'opts:',opts
    #print 'args:',args
    return (opts, args)

#--------------------

def save_array_in_file(ofname, arr) :
    #scimisc.imsave(ofname, arr) # saves as uint8
    img = Image.fromarray(arr.astype(np.int16))  # or int32
    img.save(ofname)
 
#--------------------

def get_array_from_file(fname) :
    #print 'get_array_from_file:', fname
    return np.loadtxt(fname, dtype=np.float32)

#--------------------

def get_list_of_files_to_convert(dirname, fname_prefix) :
    list_of_files = []
    for fname in os.listdir(dirname) :
        if fname.split('-')[0] == fname_prefix and os.path.splitext(fname)[1] == '.txt' :
            list_of_files.append(fname)
    return  list_of_files

#--------------------

def do_main() :

    opts, args = get_input_parameters()
    fname = opts.fname
    dname = opts.dname

    for fname in get_list_of_files_to_convert(dname, fname) :
        ifpath = dname + '/'+ fname        
        ofpath = dname + '/'+ os.path.splitext(fname)[0] + '.tiff'
        print 'Convert ' + ifpath + ' ===> ' + ofpath
        arr = get_array_from_file(ifpath)
        save_array_in_file(ofpath, arr)
        #print 'arr=\n', arr
  
#--------------------

if __name__ == '__main__' :
    do_main()
    sys.exit('The End')

#--------------------
