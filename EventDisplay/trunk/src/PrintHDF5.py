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
import h5py

#-----------------------------
# Imports for other modules --
#-----------------------------
#from PkgPackage.PkgModule import PkgClass

import ConfigParameters as cp

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

def print_dataset_info(ds):
    """Prints attributes and all other available info for group or data"""
    if isinstance(ds,h5py.Dataset):
        print "Dataset:",
        print "ds.name           = ", ds.name
        print "ds.dtype          = ", ds.dtype
        print "ds.shape          = ", ds.shape
        print "ds.value          = ", ds.value

    if isinstance(ds,h5py.Group):
        print "Group:",
        print "ds.name           = ", ds.name
        print_group_items(ds)

    if isinstance(ds,h5py.File):
        print "File:"
        print "file.name           = ", file.name
        print "Run number          = ", file.attrs['runNumber']

    print "ds.id             = ", ds.id 
    print "ds.ref            = ", ds.ref 
    print "ds.parent         = ", ds.parent
    print "ds.file           = ", ds.file

    #print_attributes(ds)

#----------------------------------

def print_file_info(file):
    """Prints attributes and all other available info for group or data"""

    print "file.name           = ", file.name
    print "file.attrs          = ", file.attrs 
    print "file.attrs.keys()   = ", file.attrs.keys() 
    print "file.attrs.values() = ", file.attrs.values() 
    print "file.id             = ", file.id 
    print "file.ref            = ", file.ref 
    print "file.parent         = ", file.parent
    print "file.file           = ", file.file

    print "Run number          = ", file.attrs['runNumber']
    print_attributes(file)

#----------------------------------

def print_group_items(g):
    """Prints items in this group"""

    list_of_items = g.items()
    Nitems = len(list_of_items)
    print "Number of items in the group = ", Nitems
    #print "g.items() = ", list_of_items
    if Nitems != 0 :
        for item in list_of_items :
            print '     ', item 
                        
#----------------------------------

def print_attributes(ds):
    """Prints all attributes for data set or file"""

    Nattrs = len(ds.attrs)
    print "Number of attrs.  = ", Nattrs
    if Nattrs != 0 :
        print "ds.attrs          = ", ds.attrs 
        print "ds.attrs.keys()   = ", ds.attrs.keys() 
        print "ds.attrs.values() = ", ds.attrs.values() 
        print 'Attributes :'
        for key,val in dict(ds.attrs).iteritems() :
            print '%24s : %s' % (key, val)


#----------------------------------

def print_dataset_metadata_from_file(dsname):
    """Open file and print attributes for input dataset"""

    # Check for unreadable datasets:
    if(dsname == '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV1/XppGon.0:Cspad.0/data'):
        print 'This is CSpad data'
        return
    if(dsname == '/Configure:0000/Run:0000/CalibCycle:0000/EvrData::DataV3/NoDetector.0:Evr.0/evrData'):
        print 'TypeError: No NumPy equivalent for TypeVlenID exists'         
        return

    fname = cp.confpars.dirName+'/'+cp.confpars.fileName
    print 'Open file : %s' % (fname)
    f  = h5py.File(fname, 'r') # open read-only
    ds = f[dsname]
    print_dataset_info(ds)
    print_attributes(ds)
    f.close()
    print '=== End of attributes ==='
                            
#----------------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
