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
        parentItem = self.invisibleRootItem()
        if isinstance(g,h5py.Dataset):
            print offset, "(Dateset)   len =", g.shape #, subg.dtype
            item = QtGui.QStandardItem(QtCore.QString(g.key()))
            parentItem.appendRow(item)            
        else:
            self._add_group_to_tree(g,parentItem) # start recursions from here

    #---------------------

    def _add_group_to_tree(self, g, parentItem):
        """Adds content of the file/group/dataset iteratively, starting from the sub-groups of g"""
        for key,val in dict(g).iteritems():
            subg = val
            item = QtGui.QStandardItem(QtCore.QString(key))
            print '    ', key, #,"   ", subg.name #, val, subg.len(), type(subg), 
            if   isinstance(subg, h5py.Dataset):
                #print " (Dateset)   len =", subg.shape #, subg.dtype
                item.setIcon(self.icon_data)
                item.setCheckable(True)
                parentItem.appendRow(item)
                
            elif isinstance(subg, h5py.Group):
                print " (Group)   len =",len(subg) 
                #offset_subg = offset + '    '
                item.setIcon(self.icon_folder_closed)
                parentItem.appendRow(item)

                self._add_group_to_tree(subg,item )

    #---------------------



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
