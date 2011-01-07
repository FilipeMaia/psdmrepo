#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PrintHDF5...
#
#------------------------------------------------------------------------

"""Print structure and content of HDF5 file

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

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
#import sys

#-----------------------------
# Imports for other modules --
#-----------------------------
#from PkgPackage.PkgModule import PkgClass

import h5py

#------------------------
# Exported definitions --
#------------------------

#----------------------------------

def print_hdf5_file_structure(fname):
    """Prints the HDF5 file structure"""

    offset = '    '
    f = h5py.File(fname, 'r') # open read-only
    print_group(f,offset)
    f.close()
    print '=== EOF ==='


#----------------------------------

def print_group(g,offset):
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    print "Structure of the",
    if   isinstance(g,h5py.File):    print "'File'",
    elif isinstance(g,h5py.Group):   print "'Group' from file",
    elif isinstance(g,h5py.Dataset): print "'Dataset' from file",
    print g.file,"\n",g.name
    if   isinstance(g,h5py.Dataset): print offset, "(Dateset)   len =", g.shape #, subg.dtype
    else:                            print_group_content(g,offset)

#----------------------------------

def print_group_content(g,offset):
    """Prints content of the file/group/dataset iteratively, starting from the sub-groups of g"""
    for key,val in dict(g).iteritems():
        subg = val 
        print offset, key, #,"   ", subg.name #, val, subg.len(), type(subg), 
        if   isinstance(subg, h5py.Dataset):
            print " (Dateset)   len =", subg.shape #, subg.dtype
        elif isinstance(subg, h5py.Group):
            print " (Group)   len =",len(subg)
            offset_subg = offset + '    '
            print_group_content(subg,offset_subg)

#----------------------------------

def print_info(ds):
    """Prints attributes and all other available info for group or data"""
    #print "ds.value          = ", ds.value
    print "ds.name           = ", ds.name
    print "ds.dtype          = ", ds.dtype
    print "ds.shape          = ", ds.shape
    print "ds.attrs          = ", ds.attrs 
    print "ds.attrs.keys()   = ", ds.attrs.keys() 
    print "ds.attrs.values() = ", ds.attrs.values() 
    print "ds.id             = ", ds.id 
    print "ds.ref            = ", ds.ref 
    print "ds.parent         = ", ds.parent
    print "ds.file           = ", ds.file
    #print "ds.items()        = ", ds.items()

#----------------------------------


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
