#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module HDF5TreeViewModel...
#
#------------------------------------------------------------------------

"""Makes QtGui.QStandardItemModel for QtGui.QTreeView

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
from PyQt4 import QtGui, QtCore
import h5py
import numpy as np

import ConfigParameters     as cp
import AppUtils.AppDataPath as apputils
import PrintHDF5            as printh5

#---------------------
#  Class definition --
#---------------------
class DataSetTreeViewModel (QtGui.QStandardItemModel) :
    """Makes QtGui.QStandardItemModel for QtGui.QTreeView.
    """
    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, window=0):

        QtGui.QStandardItemModel.__init__(self, parent)

        self.window = window
        self.dsname = cp.confpars.dsWindowParameters[self.window][0]

        self.str_file  = 'File'
        self.str_data  = 'Data'
        self.str_group = 'Group'

        self.load_icons()

        print 'DataSetTreeViewModel: self.dsname=', self.dsname

        #self.fill_example_tree_model()
        if self.dsname == 'None' : 
            self.fill_worning_model()
            cp.confpars.isSetWarningModel = True 
        else :    
            self.fill_dataset_tree_model()
            self.retreve_checked_items()
            cp.confpars.isSetWarningModel = False

    #-------------------
    #  Public methods --
    #-------------------

    def load_icons(self):
        """Load icons from files in directory HDF5Analysis/data/icons/"""

        self.apppath_icon_folder_open   = apputils.AppDataPath('HDF5Analysis/icons/folder_open.gif')
        self.apppath_icon_folder_closed = apputils.AppDataPath('HDF5Analysis/icons/folder_closed.gif')
        self.apppath_icon_data          = apputils.AppDataPath('HDF5Analysis/icons/table.gif')

        self.icon_folder_open   = QtGui.QIcon(self.apppath_icon_folder_open  .path())
        self.icon_folder_closed = QtGui.QIcon(self.apppath_icon_folder_closed.path())
        self.icon_data          = QtGui.QIcon(self.apppath_icon_data         .path())


    def open_hdf5_file(self, fname_input=None) :

        if fname_input==None : self.fname = cp.confpars.dirName+'/'+cp.confpars.fileName
        else                 : self.fname = fname_input
        print '=== Open HDF5 file: ' + self.fname

        try :
            self.h5file = h5py.File(self.fname,'r') # open read-only
        except IOError:
            print 'IOError: CAN NOT OPEN FILE:', self.fname
            return


    def close_hdf5_file(self) :
        self.h5file.close()
        print '=== Close HDF5 file ==='


    def get_dataset_from_hdf5_file(self,dsname) :
        print 'From hdf5 file get dataset :', dsname

        try :
            return self.h5file[str(dsname)]
        except KeyError:
            print 80*'!'
            print 'WARNING:', dsname, ' DATASET DOES NOT EXIST IN HDF5\n'
            print 80*'!'
            return None


    def fill_dataset_tree_model(self) :
        """Fills the dataset structure in the tree model"""

        self.open_hdf5_file()

        ds = self.get_dataset_from_hdf5_file(self.dsname) 

        #self.iterate_print_over_data_structure(ds)

        self.parentItem = self.invisibleRootItem()
        self.parentItem.setAccessibleDescription('root')
        self.parentItem.setAccessibleText(self.fname) # self.dsname
        header_item = QtGui.QStandardItem('File: ' + self.fname)
        header_item.setTextAlignment(QtCore.Qt.AlignRight)
        self.setHorizontalHeaderItem(0,header_item) # Set title for root-item

        #self.add_item_iterate(ds.dtype, self.parentItem)  # based on dtype
        self.add_item_iterate_ds(ds, self.parentItem, self.dsname) # based on dataset

        self.close_hdf5_file()


#----------------------------------




#----------------------------------

    def add_item_iterate_ds(self, ds, parentItem, indtitle='', offset0=''):
        """Add item to the data structure tree"""

        offset=offset0+'    '

        print offset + '==================== New item ==================='

        #isAddedToMenu = False

        try:
            self.ds_dtype = ds.dtype
            DTYPE_IS_AVAILABLE = True
        except AttributeError:
            print offset + 'AttributeError: current object has no attribute dtype'
            self.ds_dtype = None
            DTYPE_IS_AVAILABLE = False
            
        try:
            self.ds_shape = ds.shape
            if ds.shape : SHAPE_IS_AVAILABLE = True
            else        : SHAPE_IS_AVAILABLE = False
        except AttributeError:
            print offset + 'AttributeError: current object has no attribute shape'
            self.ds_shape = None
            SHAPE_IS_AVAILABLE = False

        if SHAPE_IS_AVAILABLE :
            try :
                str_dims = self.get_str_ds_shape_dims(ds)
                DIMS_ARE_AVAILABLE = True
            except AttributeError:
                str_dims = None               
                DIMS_ARE_AVAILABLE = False


        print offset + 'dsindex            :', indtitle
        print offset + 'DTYPE_IS_AVAILABLE :', DTYPE_IS_AVAILABLE
        print offset + 'SHAPE_IS_AVAILABLE :', SHAPE_IS_AVAILABLE
        self.print_ds_attributes(ds, offset)

        if DTYPE_IS_AVAILABLE : # ds.dtype IS AVAILABLE

            if SHAPE_IS_AVAILABLE and DIMS_ARE_AVAILABLE :

                print offset + 'THIS IS A COMPOSIT OBJECT WITH SHAPE  <--------------------'

                #title = indtitle + '  DATA ARRAY  ' + str_dims + ' ' + str(self.ds_shape)
                title = indtitle # + '  DATA ARRAY  ' + str_dims 
                item = self.add_item_to_tree(parentItem,                             # parent
                                             title,                                  # title
                                             self.icon_folder_closed,                # icon
                                             'group',                                # description
                                             str_dims,                               # text
                                             True)                                   # isCheckable
                #isAddedToMenu = True

                print offset + 'SHAPE IS NOT EMPTY !!! =', ds.shape, ' CONTINUE ITERATIONS'
                ds0 = self.get_ds0_shaped(ds)
                self.add_item_iterate_ds(ds0, item, str(self.ds_shape), offset) # str(self.ds_shape), str_dims


            elif ds.dtype.num == 20 :

                print offset + 'THIS IS A COMPOSIT OBJECT WITH NUM=20 <--------------------'

                title = indtitle # + ' INDEXED ARR ' + str(self.ds_shape)
                item = self.add_item_to_tree(parentItem,                             # parent
                                             title,                                  # title
                                             self.icon_folder_closed,                # icon
                                             'group',                                # description
                                             indtitle,                               # text
                                             True)                                   # isCheckable
                #isAddedToMenu = True

                for dtype_descr in ds.dtype.descr :        

                    indname = dtype_descr[0]
                    print offset + 'Index Name =', indname         

                    if indname != '' :
                        try:
                            chield_ds = ds[indname]
                        except (TypeError, KeyError):
                            chield_ds = None           
                            continue            
                        print offset + 'chield_ds [FROM INDEX]=' #,  chield_ds_dtype
                        self.add_item_iterate_ds(chield_ds, item, indname, offset)

                    else :                
                        print offset + 'INDEX IS NOT AVAILABLE - STOP ITERATIONS'

                        if SHAPE_IS_AVAILABLE :
                             print offset + 'THIS IS A COMPOSIT OBJECT WITH SHAPE  <--------------------'
                             print offset + 'SHAPE IS NOT EMPTY !!! =', ds.shape, ' CONTINUE ITERATIONS'
                             ds0 = self.get_ds0_shaped(ds)
                             str_dims = self.get_str_ds_shape_dims(ds)
                             self.add_item_iterate_ds(ds0, item, str(self.ds_shape), offset) # str(self.ds_shape), str_dims



            else : # NON COMPOSIT DATA OBJECT

                print offset + 'THIS IS A SINGLE DATA OBJECT      <--------------------'
                title = indtitle # + ' SINGLE DATA OBJECT ' + ds.dtype.name # + ' ' + str (self.ds_shape) + '  DATA: ' + str(ds)
                item = self.add_item_to_tree(parentItem,                             # parent
                                             title,                                  # title
                                             self.icon_data,                         # icon
                                             'data',                                 # description
                                             '',                                     # text
                                             True)                                   # isCheckable
                #isAddedToMenu = True



        else : # DTYPE IS NOT AVAILABLE

            if SHAPE_IS_AVAILABLE :
                print offset + 'SHAPE IS NOT EMPTY !!! =', ds.shape, ' ADD AS AN ARRAY, CONTINUE ITERATIONS'

                str_dims = self.get_str_ds_shape_dims(ds)
                title = indtitle # + '  NO DTYPE COMP ' + str_dims + str(self.ds_shape) + ' ' + str(ds)

                item = self.add_item_to_tree(parentItem,                             # parent
                                             title,                                  # title
                                             self.icon_data,                         # icon
                                             'data',                                 # description
                                             str_dims,                               # text
                                             True)                                   # isCheckable
                #isAddedToMenu = True

                ds0 = self.get_ds0_shaped(ds)
                self.add_item_iterate_ds(ds0,item,'array',offset)

            else : # SHAPE IS NOT AVAILABLE
                print offset + 'SHAPE IS EMPTY, ADD AS A DATASET, STOP ITERATIONS' 
                print offset + 'ds =', str(ds)
                title = indtitle # + '   DATA: ' + str(ds)
                item = self.add_item_to_tree(parentItem,                             # parent
                                             title,                                  # title
                                             self.icon_data,                         # icon
                                             'data',                                 # description
                                             'Data w/o shape',                       # text
                                             True)                                   # isCheckable
                #isAddedToMenu = True


#----------------------------------

    def print_ds_attributes(self, ds, offset=''):

        if  self.ds_dtype != None :
            print offset + 'ds.dtype =',           ds.dtype
            print offset + 'ds.dtype.descr     =', ds.dtype.descr    
            print offset + 'ds.dtype.type      =', ds.dtype.type
            print offset + 'ds.dtype.kind      =', ds.dtype.kind
            print offset + 'ds.dtype.char      =', ds.dtype.char
            print offset + 'ds.dtype.num       =', ds.dtype.num 
            print offset + 'ds.dtype.str       =', ds.dtype.str 
            print offset + 'ds.dtype.name      =', ds.dtype.name 
            print offset + 'ds.dtype.itemsize  =', ds.dtype.itemsize
            print offset + 'ds.dtype.fields    =', ds.dtype.fields
            print offset + 'ds.dtype.names     =', ds.dtype.names
            print offset + 'ds.dtype.hasobject =', ds.dtype.hasobject
            print offset + 'ds.dtype.flags     =', ds.dtype.flags    
            print offset + 'ds.dtype.isbuiltin =', ds.dtype.isbuiltin
            print offset + 'ds.dtype.isnative  =', ds.dtype.isnative 
            print offset + 'ds.dtype.alignment =', ds.dtype.alignment
            print offset + 'ds.dtype.subdtype  =', ds.dtype.subdtype  
            print offset + 'ds.dtype.shape     =', ds.dtype.shape

#----------------------------------

    def add_item_iterate(self, ds_dtype, parentItem, indtitle='', offset0=''):
        """Add group to the data structure tree. This iterator is based on dtype roll-out."""

        offset=offset0+'    '

        print offset + '==================== Next item ==================='
        print offset + 'ds_dtype            =', ds_dtype
        print offset + 'ds_dtype.descr      =', ds_dtype.descr
        print offset + 'ds_dtype.type       =', ds_dtype.type      
        print offset + 'ds_dtype.kind       =', ds_dtype.kind      
        print offset + 'ds_dtype.char       =', ds_dtype.char      
        print offset + 'ds_dtype.num        =', ds_dtype.num       
        print offset + 'ds_dtype.str        =', ds_dtype.str       
        print offset + 'ds_dtype.name       =', ds_dtype.name      
        print offset + 'ds_dtype.itemsize   =', ds_dtype.itemsize  
        print offset + 'ds_dtype.fields     =', ds_dtype.fields    
        print offset + 'ds_dtype.names      =', ds_dtype.names     
        print offset + 'ds_dtype.hasobject  =', ds_dtype.hasobject 
        print offset + 'ds_dtype.flags      =', ds_dtype.flags     
        print offset + 'ds_dtype.isbuiltin  =', ds_dtype.isbuiltin 
        print offset + 'ds_dtype.isnative   =', ds_dtype.isnative  
        print offset + 'ds_dtype.alignment  =', ds_dtype.alignment 
        print offset + 'ds_dtype.subdtype   =', ds_dtype.subdtype  
        print offset + 'ds_dtype.shape      =', ds_dtype.shape     

        if ds_dtype.num != 20 :
            print offset + 'THIS IS A SINGLE DATA OBJECT      <--------------------'
            title = indtitle + '    ' + ds_dtype.name + str (ds_dtype.shape)
            item = self.add_item_to_tree(parentItem,                             # parent
                                         title,                                  # title
                                         self.icon_data,                         # icon
                                         'data',                                 # description
                                         ds_dtype.shape,                         # text
                                         True)                                   # isCheckable
        else :
            print offset + 'THIS IS A COMPOSIT OBJECT <--------------------'

            if ds_dtype.shape :
                print offset + 'SHAPE IS NOT EMPTY !!! =', ds_dtype.shape, ' ADD AS DATA ARRAY'

            title = indtitle + str(ds_dtype.shape)
            item = self.add_item_to_tree(parentItem,                             # parent
                                         title,                                  # title
                                         self.icon_folder_closed,                # icon
                                         'group',                                # description
                                         ds_dtype.shape,                         # text
                                         True)                                   # isCheckable

            for dtype_descr in ds_dtype.descr :        

                indname = dtype_descr[0]
                print offset + 'Index Name =', indname         

                if indname != '' :
                    try:
                        chield_ds_dtype = ds_dtype[indname]
                    except (TypeError, KeyError):
                        chield_ds_dtype = None           
                        continue            
                    print offset + 'chield_ds_dtype [FROM INDEX]=',  chield_ds_dtype
                    self.add_item_iterate(chield_ds_dtype, item, indname, offset)

                else :
                
                    chield_ds_dtype = ds_dtype.subdtype[0]
                    print offset + 'chield_ds_dtype [FROM SUBDTYPE]=', chield_ds_dtype                    
                    self.add_item_iterate(chield_ds_dtype, item, 'unnamed', offset)

#----------------------------------

    def add_item_to_tree(self, parentItem, title, icon, description, text, isCheckable=False) :
        item = QtGui.QStandardItem(QtCore.QString(title))
        item.setIcon(icon)
        item.setAccessibleDescription(str(description))
        item.setAccessibleText(str(text))
        item.setCheckable(isCheckable)
        parentItem.appendRow(item)
        return item

#----------------------------------

    def print_ds_type_attributes(self,ds_dtype):

        if  ds_dtype != None :
            print 'TEST:ds_dtype =',           ds_dtype
            print 'TEST:len(ds_dtype)      =', len(ds_dtype)
            print 'TEST:ds_dtype.type      =', ds_dtype.type
            print 'TEST:ds_dtype.kind      =', ds_dtype.kind
            print 'TEST:ds_dtype.char      =', ds_dtype.char
            print 'TEST:ds_dtype.num       =', ds_dtype.num 
            print 'TEST:ds_dtype.str       =', ds_dtype.str 
            print 'TEST:ds_dtype.name      =', ds_dtype.name 
            print 'TEST:ds_dtype.itemsize  =', ds_dtype.itemsize
            print 'TEST:ds_dtype.fields    =', ds_dtype.fields
            print 'TEST:ds_dtype.names     =', ds_dtype.names
            print 'TEST:ds_dtype.hasobject =', ds_dtype.hasobject
            print 'TEST:ds_dtype.flags     =', ds_dtype.flags    
            print 'TEST:ds_dtype.isbuiltin =', ds_dtype.isbuiltin
            print 'TEST:ds_dtype.isnative  =', ds_dtype.isnative 
            print 'TEST:ds_dtype.descr     =', ds_dtype.descr    
            print 'TEST:ds_dtype.alignment =', ds_dtype.alignment
            print 'TEST:ds_dtype.subdtype  =', ds_dtype.subdtype
            print 'TEST:ds_dtype.shape     =', ds_dtype.shape

#----------------------------------

#----------------------------------

    def get_str_ds_shape_dims(self,ds) :
        str_ds_shape_dims = ''
        for shapeDim in ds.shape :
            if str_ds_shape_dims != '' : str_ds_shape_dims += ' '
            str_ds_shape_dims += str(shapeDim)

        ds0_shape = self.get_ds0_shape(ds)

        for shapeDim in ds0_shape :
            str_ds_shape_dims += ' '
            str_ds_shape_dims += str(shapeDim)

        return str_ds_shape_dims

#----------------------------------

    def get_ds0_shaped(self,ds) :
        if   len(ds.shape) == 0 : ds0 = None
        elif len(ds.shape) == 1 : ds0 = ds[0]
        elif len(ds.shape) == 2 : ds0 = ds[0][0]
        elif len(ds.shape) == 3 : ds0 = ds[0][0][0]
        elif len(ds.shape) == 4 : ds0 = ds[0][0][0][0]
        elif len(ds.shape) == 5 : ds0 = ds[0][0][0][0][0]
        elif len(ds.shape) == 6 : ds0 = ds[0][0][0][0][0][0]
        elif len(ds.shape) == 7 : ds0 = ds[0][0][0][0][0][0][0]
        elif len(ds.shape) == 8 : ds0 = ds[0][0][0][0][0][0][0][0]
        elif len(ds.shape) == 9 : ds0 = ds[0][0][0][0][0][0][0][0][0]
        return ds0

#----------------------------------

    def get_ds0_shape(self,ds) :
        if   len(ds.shape) == 0 : ds0_shape = None
        elif len(ds.shape) == 1 : ds0_shape = ds[0].shape
        elif len(ds.shape) == 2 : ds0_shape = ds[0][0].shape
        elif len(ds.shape) == 3 : ds0_shape = ds[0][0][0].shape
        elif len(ds.shape) == 4 : ds0_shape = ds[0][0][0][0].shape
        elif len(ds.shape) == 5 : ds0_shape = ds[0][0][0][0][0].shape
        elif len(ds.shape) == 6 : ds0_shape = ds[0][0][0][0][0][0].shape
        elif len(ds.shape) == 7 : ds0_shape = ds[0][0][0][0][0][0][0].shape
        elif len(ds.shape) == 8 : ds0_shape = ds[0][0][0][0][0][0][0][0].shape
        elif len(ds.shape) == 9 : ds0_shape = ds[0][0][0][0][0][0][0][0][0].shape
        return ds0_shape    

#----------------------------------

    def iterate_print_over_data_structure(self,ds,offset0=''):
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

        for indname in ds.dtype.names :
            print offset,'Index Name =', indname         
            self.iterate_print_over_data_structure(ds[indname],offset)

#---------------------

    def fill_worning_model(self) :
        parentItem = self.invisibleRootItem()
        item = QtGui.QStandardItem(QtCore.QString("Select the dataset in HDF5 tree, then in option 2 of this GUI."))
        #item.setIcon(self.icon_folder_closed)
        parentItem.appendRow(item)
        header_item = QtGui.QStandardItem(self.dsname)
        header_item.setTextAlignment(QtCore.Qt.AlignLeft)
        self.setHorizontalHeaderItem(0,header_item)

#---------------------

    def fill_example_tree_model(self) :
        for k in range(0, 5):
            parentItem = self.invisibleRootItem()
            #parentItem.setIcon(QIcon("folder_open.gif"))
            for i in range(0, k):
                item = QtGui.QStandardItem(QtCore.QString("itemA %0 %1").arg(k).arg(i))
                item.setIcon(self.icon_data)
                item.setCheckable(True) 
                parentItem.appendRow(item)
                item = QtGui.QStandardItem(QtCore.QString("itemB %0 %1").arg(k).arg(i))
                item.setIcon(self.icon_folder_closed)
                parentItem.appendRow(item)
                parentItem = item
                print 'append item %s' % (item.text())

        header_item = QtGui.QStandardItem(self.dsname)
        header_item.setTextAlignment(QtCore.Qt.AlignLeft)
        self.setHorizontalHeaderItem(0,header_item)

#---------------------
#---------------------
#---------------------
#---------------------

    def expand_checked_items(self,tree):
        """Iterates over items in the QTreeModel and expand all checked items"""
        self.tree = tree
        self._iterate_over_items_and_expand_checked(self.parentItem)

    #---------------------

    def _iterate_over_items_and_expand_checked(self,item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        if state == 'CHECKED' or state == 'TRISTATE' :
            print ' Expand item.text():', item.text()
            self._expand_all_parents(item)
            
        if item.hasChildren():
            for row in range(item.rowCount()) :
                chield_item = item.child(row,0)
                self._iterate_over_items_and_expand_checked(chield_item)                

#---------------------

    def _expand_all_parents(self,item):
        """Expand all parent groups for this checked item"""
        item_parent = item.parent()
        ind_parent  = self.indexFromItem(item_parent)
        if(ind_parent.column() != -1) :
            if item_parent.accessibleDescription() == 'group' :
                self.tree.expand(ind_parent)
                item_parent.setIcon(self.icon_folder_open)
                self._expand_all_parents(item_parent)

#---------------------
#---------------------
#---------------------
#---------------------

    def get_list_of_checked_items(self):
        """Iterates over items in the QTreeModel, find all checked, and make a list"""
        self.list_of_checked_dataset_indexes = []
        self._iterate_over_items_and_find_checked(self.parentItem)
        return self.list_of_checked_dataset_indexes

#---------------------

    def _iterate_over_items_and_find_checked(self,item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        if state == 'CHECKED' or state == 'TRISTATE' :
            list_of_indexes_for_one_item = self.get_index_titles_to_item(item)
            print '  Found checked item.text():', item.text()
            #print '  list of indexes:', list_of_indexes_for_one_item
            self.list_of_checked_dataset_indexes.append(list_of_indexes_for_one_item)
                        
        if item.hasChildren():
            for row in range(item.rowCount()) :
                chield_item = item.child(row,0)
                self._iterate_over_items_and_find_checked(chield_item)                

#---------------------
#---------------------
#---------------------
#---------------------

    def reset_checked_items(self):
        """Iterates over the list of item in the QTreeModel and uncheck all checked items"""
        self._iterate_over_items_and_uncheck(self.parentItem)

    #---------------------

    def _iterate_over_items_and_uncheck(self,item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        if state == 'CHECKED' or state == 'TRISTATE' :
            print ' Uncheck item.text():', item.text()
            item.setCheckState(0) # 0 means UNCHECKED
            
        if item.hasChildren():
            for row in range(item.rowCount()) :
                chield_item = item.child(row,0)
                self._iterate_over_items_and_uncheck(chield_item)    

#---------------------
#---------------------
#---------------------
#---------------------

    def retreve_checked_items(self):
        """Use the input list of items and check them in the tree model"""
        self.list_of_indexes_2d = cp.confpars.dsWindowParameters[self.window][2]
        print 'dsname in hdf5 :', self.dsname
        print 'self.parentItem.accessibleText() = ', self.parentItem.accessibleText()

        if self.list_of_indexes_2d == None : return
        self._iterate_over_items_and_check_items_for_list(self.parentItem)

    #---------------------

    def _iterate_over_items_and_check_items_for_list(self,item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""

        if item.isCheckable(): 
            list_of_indexes_for_item = self.get_index_titles_to_item(item)
            if list_of_indexes_for_item in self.list_of_indexes_2d :
                print ' Check the item with indexes:', list_of_indexes_for_item
                item.setCheckState(2) # 2 means CHECKED; 1-TRISTATE
            
        if item.hasChildren():
            for row in range(item.rowCount()) :
                chield_item = item.child(row,0)
                self._iterate_over_items_and_check_items_for_list(chield_item)                

#---------------------
#---------------------
#---------------------
#---------------------

    def get_dataset_dims(self,item):
        """Returns the list of dataset numerical shape-dimensions"""
        if item.accessibleDescription() == 'data' :
            str_dims = str(item.accessibleText())
            #print '\nget_dataset_dims::str_dims  =', str_dims
            dims     = np.fromstring(str_dims, dtype = int, sep=' ')
            return dims
        else :
            return None

#---------------------
        
    def get_dataset_prod_of_dims(self,item) :
        """Returns the product of all numerical dimensions (shapes) of the dataset"""
        if item.accessibleDescription() != 'data' : return 0
        list_of_dims = self.get_dataset_dims(item)
        prod = 1
        for dim in list_of_dims :
            prod *= dim
        #print "dims = ", list_of_dims, "    product = ", prod 
        return prod

#---------------------
#---------------------
#---------------------
#---------------------

    def get_dims_from_item_title(self,item_title):

        if item_title[0] == '(' : # THIS SHOULD BE THE SHAPE
            str_dims = item_title[1:len(item_title)-1]
            #print 'get_dims_from_item_title : str_dims =', str_dims
            dims = np.fromstring(str_dims, dtype = np.int, sep=',')            
            #print 'get_dims_from_item_title : dims =', dims,
            #print ' len(dims) =', len(dims)
            return dims

        else : return None

#---------------------

    def get_indexes_to_item(self,item):
        """Returns the list of symbolic AND NUMERICAL indexes to the indicated item. The hdf5 dsname is not included."""

        list_of_index_titles_to_item = self.get_index_titles_to_item(item)

        self.list_of_indexes = []
        for title in list_of_index_titles_to_item :

            if title[0] == '(' : # THIS SHOULD BE THE SHAPE
                list_of_dims = self.get_dims_from_item_title(title)
                for dim in list_of_dims:
                    numerical_index = 0 # dim-1
                    self.list_of_indexes.append(numerical_index)
            else :
                self.list_of_indexes.append(title)

        return self.list_of_indexes




#---------------------

    def get_index_titles_to_item(self,item):
        """Returns the ORDERED !!! list of SYMBOLIC indexes to the indicated item. The hdf5 dsname is not included."""

        self.list_of_named_indexes = []
        self._iterate_over_parent_items(item)
        self.list_of_named_indexes.reverse()
        return self.list_of_named_indexes

#---------------------

    def _iterate_over_parent_items(self,item):
        """Add the item symbolic index to the list and iterate over parents"""

        #self.print_item_attributes(item)

        #if item.accessibleDescription() == 'group' :
        self.list_of_named_indexes.append(str(item.text()))

        item_parent = item.parent()
        ind_parent  = self.indexFromItem(item_parent)

        if ind_parent.column() != -1 :
        #if item_parent != 0 :
            self._iterate_over_parent_items(item_parent)

#---------------------

    def print_item_attributes(self,item):

        print 'print_item_attributes ---------------------------------------' 
        print 'print_item_attributes : item.text()                  =', item.text()
        print 'print_item_attributes : item.accessibleText()        =', item.accessibleText()
        print 'print_item_attributes : item.accessibleDescription() =', item.accessibleDescription()

#---------------------
#---------------------
#---------------------
#---------------------

    def get_dataset(self, item, num_index=None) :

        #if item.accessibleDescription() != 'data' :
        #    print 'THIS IS NOT A DATA ITEM...'
        #    return None

        list_of_named_indexes = self.get_indexes_to_item(item)
        #print 'list_of_named_indexes = ', list_of_named_indexes

        self.open_hdf5_file()

        for index in list_of_named_indexes : 

            if index == self.dsname :
                ds = self.get_dataset_from_hdf5_file(self.dsname) 
            else :
                print 'Apply index :', index
                ds = ds[index]

        if num_index != None : ds = ds[num_index]

        self.close_hdf5_file()

        return ds

#---------------------

    def get_dataset_0(self, item) :
        ds = self.get_dataset(item, 0)
        #print ds
        return ds

#---------------------
#---------------------
#---------------------
#---------------------

    def set_all_group_icons(self,icon):
        """Iterates over all items in the QTreeModel and set icon for all groups"""
        self.new_icon = icon
        self._iterate_over_items_and_set_icon(self.parentItem)

#---------------------

    def _iterate_over_items_and_set_icon(self,item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        if item.accessibleDescription() == 'group' :
            item.setIcon(self.new_icon)
            
        if item.hasChildren():
            for row in range(item.rowCount()) :
                chield_item = item.child(row,0)
                self._iterate_over_items_and_set_icon(chield_item)                

#---------------------
#---------------------
#---------------------
#---------------------
