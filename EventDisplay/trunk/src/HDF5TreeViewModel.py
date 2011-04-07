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

#-----------------------------
# Imports for other modules --
#-----------------------------

import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------
class HDF5TreeViewModel (QtGui.QStandardItemModel) :
    """Makes QtGui.QStandardItemModel for QtGui.QTreeView.
    """
    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None):

        self.icon_folder_open   = QtGui.QIcon("EventDisplay/src/icons/folder_open.gif")
        self.icon_folder_closed = QtGui.QIcon("EventDisplay/src/icons/folder_closed.gif")
        self.icon_data          = QtGui.QIcon("EventDisplay/src/icons/table.gif")

        QtGui.QStandardItemModel.__init__(self, parent)

        self.str_file  = 'File'
        self.str_data  = 'Data'
        self.str_group = 'Group'

       #self._model_example()
        self._model_hdf5_tree()

    #-------------------
    #  Public methods --
    #-------------------

    #--------------------
    #  Private methods --
    #--------------------

    def _model_hdf5_tree(self) :
        """Puts the HDF5 file structure in the model tree"""

        fname = cp.confpars.dirName+'/'+cp.confpars.fileName
        print 'Makes the tree view model for HDF5 file : ' + fname
        f = h5py.File(fname, 'r') # open read-only
        self._begin_construct_tree(f)
        f.close()
        print '=== EOF ==='

    #---------------------

    def _begin_construct_tree(self, g):
        """Adds the input file/group/dataset (g) name and begin iterations on its content"""

        print "Add structure of the",
        if   isinstance(g,h5py.File):    print "'File'",
        elif isinstance(g,h5py.Group):   print "'Group' from file",
        elif isinstance(g,h5py.Dataset): print "'Dataset' from file",
        print g.file,"\n",g.name
        self.parentItem = self.invisibleRootItem()
        self.parentItem.setAccessibleDescription(self.str_file)
        self.parentItem.setAccessibleText(g.name) # Root item does not show this text...
        #self.parentItem.setIcon(self.icon_folder_open) # Root item does not show icon...
        
        if isinstance(g,h5py.Dataset):
            print offset, "(Dateset)   len =", g.shape #, subg.dtype
            item = QtGui.QStandardItem(QtCore.QString(g.key()))
            item.setAccessibleDescription(self.str_data)
            self.parentItem.appendRow(item)            
        else:
            self._add_group_to_tree(g,self.parentItem) # start recursions from here

    #---------------------

    def _add_group_to_tree(self, g, parentItem):
        """Adds content of the file/group/dataset iteratively, starting from the sub-groups of g"""

        d = dict(g)
        list_keys = sorted(d.keys())
        list_vals = d.values()
        #print 'list_keys =', list_keys 

        for key in list_keys:
        #for key,val in dict(g).iteritems():

            #subg = val
            subg = d[key]

            item = QtGui.QStandardItem(QtCore.QString(key))
            #print '    k=', key, #,"   ", subg.name #, val, subg.len(), type(subg), 
            if isinstance(subg, h5py.Dataset):
                #print " (Dateset)   len =", subg.shape #, subg.dtype
                item.setIcon(self.icon_data)
                item.setCheckable(True)
                item.setAccessibleDescription(self.str_data)
                item.setAccessibleText(str(key))
                parentItem.appendRow(item)
                
            elif isinstance(subg, h5py.Group):
                #print " (Group)   len =",len(subg) 
                #offset_subg = offset + '    '
                item.setIcon(self.icon_folder_closed)
                item.setAccessibleDescription(self.str_group)
                item.setAccessibleText(str(key))
                parentItem.appendRow(item)

                self._add_group_to_tree(subg,item )

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def getFullNameFromItem(self, item): 
        """Returns the full name in the HDF5 tree model for the given item"""
        ind = self.indexFromItem(item)        
        return self.getFullNameFromIndex(ind)

    #---------------------
    
    def getFullNameFromIndex(self, ind): 
        """Begin recursion from item with given ind(ex) and forms the full name in the self._full_name"""
        item = self.itemFromIndex(ind)
        self._full_name = item.text() 
        self._getFullName(ind) 
        return str(self._full_name) ### QString object is converted to str

    #---------------------

    def _getFullName(self, ind): 
        """Recursion from child to parent"""
        ind_par  = self.parent(ind)
        if(ind_par.column() == -1) :
            item = self.itemFromIndex(ind)
            self._full_name = '/' + self._full_name
            #print 'Item full name :' + self._full_name
            return self._full_name
        else:
            item_par = self.itemFromIndex(ind_par)
            self._full_name = item_par.text() + '/' + self._full_name
            self._getFullName(ind_par)

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def set_all_group_icons(self,icon):
        """Iterates over the list of item in the QTreeModel and set icon for all groups"""
        self.new_icon = icon
        self._iteration_over_items_and_set_icon(self.parentItem)

    #---------------------

    def _iteration_over_items_and_set_icon(self,parentItem):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        if parentItem.accessibleDescription() == 'Group' :
            parentItem.setIcon(self.new_icon)
            
        if parentItem.hasChildren():
            for row in range(parentItem.rowCount()) :
                item = parentItem.child(row,0)
                self._iteration_over_items_and_set_icon(item)                

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def reset_checked_items(self):
        """Iterates over the list of item in the QTreeModel and uncheck all checked items"""
        self._iteration_over_items_and_uncheck(self.parentItem)

    #---------------------

    def _iteration_over_items_and_uncheck(self,parentItem):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][parentItem.checkState()]
        if state == 'CHECKED' or state == 'TRISTATE' :
            print ' Uncheck item.text():', parentItem.text()
            parentItem.setCheckState(0) # 0 means UNCHECKED
            
        if parentItem.hasChildren():
            for row in range(parentItem.rowCount()) :
                item = parentItem.child(row,0)
                self._iteration_over_items_and_uncheck(item)                

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def expand_checked_items(self,view):
        """Iterates over the list of item in the QTreeModel and expand all checked items"""
        self.view = view
        self._iteration_over_items_and_expand_checked(self.parentItem)

    #---------------------

    def _iteration_over_items_and_expand_checked(self,parentItem):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][parentItem.checkState()]
        if state == 'CHECKED' or state == 'TRISTATE' :
            print ' Expand item.text():', parentItem.text()
            #Now we have to expand all parent groups for this checked item...
            self._expand_parents(parentItem)
            
        if parentItem.hasChildren():
            for row in range(parentItem.rowCount()) :
                item = parentItem.child(row,0)
                self._iteration_over_items_and_expand_checked(item)                

    #---------------------

    def _expand_parents(self,item):
        item_parent = item.parent()
        ind_parent  = self.indexFromItem(item_parent)
        if(ind_parent.column() != -1) :
            if item_parent.accessibleDescription() == 'Group' :
                self.view.expand(ind_parent)
                item_parent.setIcon(self.icon_folder_open)
                self._expand_parents(item_parent)

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def retreve_checked_items(self,list_of_checked_item_names):
        """Use the input list of items and check them in the tree model"""
        self.list_of_checked_item_names = list_of_checked_item_names
        for name in self.list_of_checked_item_names :
            print 'Retreve the CHECKMARK for item', name 
        self._iteration_over_items_and_check_from_list(self.parentItem)    

    #---------------------

    def _iteration_over_items_and_check_from_list(self,parentItem):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""

        if parentItem.isCheckable():
            item_name = self.getFullNameFromItem(parentItem)
            if item_name in self.list_of_checked_item_names :
                print ' Check the item:', item_name # parentItem.text()
                parentItem.setCheckState(2) # 2 means CHECKED; 1-TRISTATE
            
        if parentItem.hasChildren():
            for row in range(parentItem.rowCount()) :
                item = parentItem.child(row,0)
                self._iteration_over_items_and_check_from_list(item)                

    #---------------------
    #---------------------
    #---------------------
    #---------------------
    
    def get_list_of_checked_item_names_for_model(self):
        """Get the list of checked item names for the self tree model"""
        list_of_checked_items = self.get_list_of_checked_items()
        return self.get_list_of_checked_item_names(list_of_checked_items)

    #---------------------
    
    def get_list_of_checked_item_names(self,list_of_checked_items):
        """Get the list of checked item names from the input list of items"""
        self.list_of_checked_item_names=[]
        print 'The number of CHECKED items in the tree model =', len(self.list_of_checked_items)
        
        for item in list_of_checked_items :
            item_full_name = self.getFullNameFromItem(item)
            self.list_of_checked_item_names.append(item_full_name)
            print 'Checked item :', item_full_name
            
        return self.list_of_checked_item_names   

    #---------------------

    def get_list_of_checked_items(self):
        """Returns the list of checked item names in the QTreeModel"""
        self.list_of_checked_items=[]
        #self._iteration_over_tree_model_item_children_v1(self.parentItem)
        #self._iteration_over_tree_model_item_children_v2(self.parentItem)

        self._iteration_over_items_find_checked(self.parentItem)
        return self.list_of_checked_items

    #---------------------

    def _iteration_over_items_find_checked(self,parentItem):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][parentItem.checkState()]
        if state == 'CHECKED' :
            #print ' checked item.text():', parentItem.text()
            self.list_of_checked_items.append(parentItem)
            
        if parentItem.hasChildren():
            for row in range(parentItem.rowCount()) :
                item = parentItem.child(row,0)
                self._iteration_over_items_find_checked(item)                

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def _iteration_over_tree_model_item_children_v1(self,parentItem):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        parentIndex = self.indexFromItem(parentItem)
        print ' item.text():', parentItem.text(),
        print ' row:',         parentIndex.row(),        
        print ' col:',         parentIndex.column()

        if parentItem.hasChildren():
            Nrows = parentItem.rowCount()
            print ' rowCount:', Nrows

            for row in range(Nrows) :
                item = parentItem.child(row,0)
                self._iteration_over_tree_model_item_children_v1(item)                

    #---------------------

    def _iteration_over_tree_model_item_children_v2(self,parentItem):
        """Recursive iteration over item children in the freame of the QtGui.QStandardItemModel"""
        print ' parentItem.text():', parentItem.text()
        if parentItem.hasChildren():
            list_of_items = parentItem.takeColumn(0) # THIS GUY REMOVES THE COLUMN !!!!!!!!
            parentItem.insertColumn(0, list_of_items) 
            for item in list_of_items : 
                self._iteration_over_tree_model_item_children_v2(item)                

    #---------------------

    def _model_example(self) :
        """Makes the model tree for example"""
        for k in range(0, 6):
            parentItem = self.invisibleRootItem()
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

#---------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    
    sys.exit ( "Module is not supposed to be run as main module" )
#---------------------
