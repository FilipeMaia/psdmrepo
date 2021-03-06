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
import sys
import os
import time
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

    f = h5py.File(fname, 'r') # open read-only
    print_hdf5_item_structure(f)
    f.close()
    print '=== EOF ==='

#----------------------------------

def print_hdf5_item_structure(g, offset='    ') :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    if   isinstance(g,h5py.File) :
        print g.file, '(File)', g.name

    elif isinstance(g,h5py.Dataset) :
        print '(Dataset)', g.name, '    len =', g.shape #, g.dtype

    elif isinstance(g,h5py.Group) :
        print '(Group)', g.name

    else :
        print 'WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name
        sys.exit ( "EXECUTION IS TERMINATED" )

    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        for key,val in dict(g).iteritems() :
            subg = val
            print offset, key, #,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ')

#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------

def get_item_last_name(dsname):
    """Returns the last part of the full item name (after last slash)"""

    path,name = os.path.split(str(dsname))
    return name

def get_item_path_to_last_name(dsname):
    """Returns the path to the last part of the item name"""

    path,name = os.path.split(str(dsname))
    return path

def get_item_path_and_last_name(dsname):
    """Returns the path and last part of the full item name"""

    path,name = os.path.split(str(dsname))
    return path, name

#----------------------------------

def get_item_second_to_last_name(dsname):
    """Returns the 2nd to last part of the full item name"""

    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))

    return name2 

#----------------------------------

def get_item_third_to_last_name(dsname):
    """Returns the 3nd to last part of the full item name"""

    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    path3,name3 = os.path.split(str(path2))

    str(name3)

    return name3 

#----------------------------------

def get_item_name_for_title(dsname):
    """Returns the last 3 parts of the full item name (after last slashes)"""

    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    path3,name3 = os.path.split(str(path2))

    return name3 + '/' + name2 + '/' + name1


#----------------------------------

def get_item_name_for_title_4(dsname):
    """Returns the last 4 parts of the full item name (after last slashes)"""

    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    path3,name3 = os.path.split(str(path2))
    path4,name4 = os.path.split(str(path3))

    return name4 + '/' + name3 + '/' + name2 + '/' + name1

#----------------------------------

def CSpadIsInTheName(dsname):
    
    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    path3,name3 = os.path.split(str(path2))

    #print '       last name:', name1
    #print '2nd to last name:', name2
    #print '3rd to last name:', name3
    #print 'name3[0:5]', name3[0:5]

    cspadIsInTheName = False
    if name3[0:5] == 'CsPad' and name1 == 'data' : cspadIsInTheName = True
    #print 'cspadIsInTheName :', cspadIsInTheName

    return cspadIsInTheName

#----------------------------------

def print_time(ds,ind):
    """Prints formatted time if the dataset is 'time'"""
    
    item_last_name = get_item_last_name(str(ds.name))
    if item_last_name == 'time' :
        tarr = ds[ind]
        tloc = time.localtime(tarr[0]) # converts sec to tuple struct_time in local
       #tgmt = time.gmtime(tarr[0])    # converts sec to tuple struct_time in UTC
        print 'Special stuff for "time" :',tarr[0],'sec,',  tarr[1],'nsec, ', #, time.ctime(int(tarr[0]))
        print 'time local :', time.strftime('%Y-%m-%d %H:%M:%S',tloc)
       #print 'time (GMT) :', time.strftime('%Y-%m-%d %H:%M:%S',tgmt)
    
#----------------------------------

def isDataset(ds):
    """Check if the input dataset is a h5py.Dataset (exists as expected in HDF5)"""
    return isinstance(ds,h5py.Dataset)

#----------------------------------

def print_dataset_info(ds):
    """Prints attributes and all other available info for group or data"""
    if isinstance(ds,h5py.Dataset):
        print "Dataset:",
        print "ds.name         = ", ds.name
        print "ds.dtype        = ", ds.dtype
        print "ds.shape        = ", ds.shape
        print "len(ds.shape)   = ", len(ds.shape)
        if len(ds.shape) != 0 :
            print "ds.shape[0]     = ", ds.shape[0]

        # Print data array
        if   len(ds.shape)==1 and ds.shape[0] == 0 : #check if the ds.shape scalar and in not an array 
            print get_item_last_name(ds.name) + ' - item has no associated data.'

        elif len(ds.shape)==0 or ds.shape[0] == 0  or ds.shape[0] == 1 : #check if the ds.shape scalar or array with dimension 0 or 1
            print "ds.value    = ", ds.value

        elif ds.shape[0] < cp.confpars.eventCurrent: #check if the ds.shape array size less than current event number
            print " data for ds[0]:"
            print ds[0]

        else :
            print " Assume that the array 1st index is an event number ", cp.confpars.eventCurrent
            print ds[cp.confpars.eventCurrent]

            print_time(ds,cp.confpars.eventCurrent)

        print_data_structure(ds)   

    if isinstance(ds,h5py.Group):
        print "Group:",
        print "ds.name = ", ds.name
        print_group_items(ds)

    if isinstance(ds,h5py.File):
        print "File:"
        print "file.name        = ", file.name
        print "Run number       = ", file.attrs['runNumber']

    print "ds.id             = ", ds.id 
    print "ds.ref            = ", ds.ref 
    print "ds.parent         = ", ds.parent
    print "ds.file           = ", ds.file

    #print_attributes(ds)

#----------------------------------

def print_data_structure(ds):
    """Prints data structure of the dataset"""
    print 50*'I'
    print 'UNROLL AND PRINT DATASET SUBSTRUCTURE'
    iterate_over_data_structure(ds)
    print 50*'I'

#----------------------------------

def iterate_over_data_structure(ds,offset0=''):
    """Prints data structure of the dataset"""

    offset=offset0+'    '

    print offset, 'ds.shape =', ds.shape, '  len(ds.shape) =', len(ds.shape), '  shape dimension(s) =',
    if len(ds.shape) == 0 :
        print offset, 'ZERO-CONTENT DATA! : ds.dtype=',  ds.dtype
        return

    for shapeDim in ds.shape:
        print shapeDim,
    print ' '

    if len(ds.shape) > 0 :
        print offset,'Sample of data ds[0]=', ds[0]

    if len(ds.dtype) == 0 or ds.dtype.names == None :
        print offset, 'NO MORE DAUGHTERS AVAILABLE because',\
              ' len(ds.dtype) =', len(ds.dtype),\
              ' ds.dtype.names =', ds.dtype.names
        return

    #print offset, 'ds.dtype.fields =', ds.dtype.fields
    print offset, 'ds.dtype        =', ds.dtype
    print offset, 'ds.dtype.names  =', ds.dtype.names

    if ds.dtype.names==None :
        print offset, 'ZERO-DTYPE.NAMES!'
        return

    for indname in ds.dtype.names :
        print offset,'Index Name =', indname         
        iterate_over_data_structure(ds[indname],offset)


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
    #if(dsname == '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV1/XppGon.0:Cspad.0/data'):
    #    print 'This is CSpad data'
    #    return

    if(dsname == '/Configure:0000/Run:0000/CalibCycle:0000/EvrData::DataV3/NoDetector.0:Evr.0'):
        print 'TypeError: No NumPy equivalent for TypeVlenID exists...\n',70*'='
        return

    if(dsname == '/Configure:0000/Run:0000/CalibCycle:0000/EvrData::DataV3/NoDetector.0:Evr.0/evrData'):
        print 'TypeError: No NumPy equivalent for TypeVlenID exists...\n',70*'='        
        return

    fname = cp.confpars.dirName+'/'+cp.confpars.fileName
    print 'Open file : %s' % (fname)
    f  = h5py.File(fname, 'r') # open read-only
    ds = f[dsname]
    print_dataset_info(ds)
    print_attributes(ds)
    print 'Path: %s\nItem: %s' % (os.path.split(str(dsname)))
    f.close()
    print 70*'='


#----------------------------------

def getListOfDatasetParNames(dsname=None):
    """Makes a list of the dataset parameter names"""

    listOfDatasetParNames = []
    if dsname=='None'  or \
       dsname=='Index' or \
       dsname=='Time'  or \
       dsname=='Is-not-used' or \
       dsname=='Select-X-parameter' :

        listOfDatasetParNames.append('None')
        return listOfDatasetParNames

    fname = cp.confpars.dirName+'/'+cp.confpars.fileName
    f = h5py.File(fname, 'r') # open read-only
    ds = f[dsname]

    for parName in ds.dtype.names :
        print parName
        listOfDatasetParNames.append(parName)

    f.close()

    listOfDatasetParNames.append('None')
    return listOfDatasetParNames

#----------------------------------

def getListOfDatasetParIndexes(dsname=None, parname=None):
    """Makes a list of the dataset parameter indexes"""

    listOfDatasetParIndexes = []
    if dsname=='None'  or \
       dsname=='Index' or \
       dsname=='Time'  or \
       dsname=='Is-not-used' or \
       dsname=='Select-X-parameter' :

        listOfDatasetParIndexes.append('None')
        return listOfDatasetParIndexes


    if not (parname=='ipimbData'   or \
            parname=='ipimbConfig' or \
            parname=='ipmFexData') :

        listOfDatasetParIndexes.append('None')
        return listOfDatasetParIndexes
 
    fname = cp.confpars.dirName+'/'+cp.confpars.fileName
    f = h5py.File(fname, 'r') # open read-only
    ds = f[dsname]

    dspar = ds[parname]

    for parIndex in dspar.dtype.names :
        print parIndex
        listOfDatasetParIndexes.append(parIndex)

    f.close()

    listOfDatasetParIndexes.append('None')
    return listOfDatasetParIndexes

                            
#----------------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    print_hdf5_file_structure('/reg/d/psdm/CXI/cxi35711/hdf5/cxi35711-r0009.h5')
    sys.exit ( 'End of test' )

#----------------------------------
