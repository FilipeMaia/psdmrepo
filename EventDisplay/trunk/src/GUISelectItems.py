#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISelectItems...
#
#------------------------------------------------------------------------

"""Shows the HDF5 file tree-structure and allows to select data items

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

#-----------------------------
# Imports for other modules --
#-----------------------------
import HDF5TreeViewModel as h5model
import ConfigParameters as cp
import PrintHDF5        as printh5

#---------------------
#  Class definition --
#---------------------
#class GUISelectItems ( QtGui.QWidget ) :
class GUISelectItems ( QtGui.QMainWindow ) :
    """Shows the HDF5 file tree-structure and allows to select data items.

    @see BaseClass
    @see OtherClass
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, parent=None):
        #super(GUISelectItems, self).__init__(parent)
        QtGui.QWidget.__init__(self, parent)

        """Constructor."""

        self.setGeometry(10, 10, 300, 600)
        self.setWindowTitle('Item selection tree')

        #layout = QHBoxLayout()

        self.icon_folder_open   = QtGui.QIcon("EventDisplay/src/icons/folder_open.gif")
        self.icon_folder_closed = QtGui.QIcon("EventDisplay/src/icons/folder_closed.gif")
        self.icon_data          = QtGui.QIcon("EventDisplay/src/icons/table.gif")
        self.icon_apply         = QtGui.QIcon("EventDisplay/src/icons/button_ok.png")
        self.icon_reset         = QtGui.QIcon("EventDisplay/src/icons/undo.png")
        self.icon_exit          = QtGui.QIcon("EventDisplay/src/icons/exit.png")
        self.icon_expand        = QtGui.QIcon("EventDisplay/src/icons/folder_open.gif")
        self.icon_collapse      = QtGui.QIcon("EventDisplay/src/icons/folder_closed.gif")

        actApply    = QtGui.QAction(self.icon_apply,    'Apply',   self)
        actReset    = QtGui.QAction(self.icon_reset,    'Reset',   self)
        actExit     = QtGui.QAction(self.icon_exit,     'Exit',    self)
        actExpand   = QtGui.QAction(self.icon_expand,   'Expand',  self)
        actCollapse = QtGui.QAction(self.icon_collapse, 'Collapse',self)

        self.connect(actApply,     QtCore.SIGNAL('triggered()'), self.processApply)
        self.connect(actReset,     QtCore.SIGNAL('triggered()'), self.processReset)
        self.connect(actExit,      QtCore.SIGNAL('triggered()'), self.processExit)
        self.connect(actExpand,    QtCore.SIGNAL('triggered()'), self.processExpand)
        self.connect(actCollapse,  QtCore.SIGNAL('triggered()'), self.processCollapse)
        #self.connect(actExit,  QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))

        menubar = self.menuBar()
        optAct  = menubar.addMenu('&Actions')
        optAct.addAction(actApply)
        optAct.addAction(actReset)
        optAct.addAction(actExit)

        optView = menubar.addMenu('&View')
        optView.addAction(actExpand)
        optView.addAction(actCollapse)

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(actExit)
        toolbar.addAction(actApply)
        toolbar.addAction(actReset)
        toolbar.addAction(actExpand)
        toolbar.addAction(actCollapse)

        self.model = h5model.HDF5TreeViewModel()

        #---------------------------------------
        #self.model = QtGui.QStandardItemModel()
        #for k in range(0, 5):
        #    parentItem = self.model.invisibleRootItem()
        #    for i in range(0, k):
        #        item = QtGui.QStandardItem(QtCore.QString("itemA %0 %1").arg(k).arg(i))
        #        item.setIcon(self.icon_data)
        #        item.setCheckable(True) 
        #        parentItem.appendRow(item)
        #        item = QtGui.QStandardItem(QtCore.QString("itemB %0 %1").arg(k).arg(i))
        #        item.setIcon(self.icon_folder_closed)
        #        parentItem.appendRow(item)
        #        parentItem = item
        #        print 'append item %s' % (item.text())
        #---------------------------------------

        #self.view = QtGui.QListView()
        #self.view = QtGui.QTableView()
        self.view = QtGui.QTreeView()
        self.view.setModel(self.model)
        #print 'Root is decorated ? ', self.view.rootIsDecorated() # Returns True
        #self.view.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        #self.view.expandAll()
        #self.view.setExpanded(cp.confpars.indExpandedItem,True)
        self.view.setAnimated(True)
        self.setCentralWidget(self.view)

        #self.show()
        
        #self.connect(self.view.selectionModel(), QtCore.SIGNAL('selectionChanged(QItemSelection, QItemSelection)'), self.itemSelected)
        self.connect(self.view.selectionModel(), QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.cellSelected)
        #self.view.clicked.connect(self.someMethod1)       # This works
        #self.view.doubleClicked.connect(self.someMethod2) # This works
        self.model.itemChanged.connect(self.itemChanged)
        self.view.expanded.connect(self.itemExpanded)
        self.view.collapsed.connect(self.itemCollapsed)

    #-------------------
    # Private methods --
    #-------------------

    def getFullNameFromItem(self, item): 
        """Returns the full name in the HDF5 tree model for the given item"""
        #item = self.model.itemFromIndex(ind)        
        ind   = self.model.indexFromItem(item)        
        return self.getFullNameFromIndex(ind)

    def getFullNameFromIndex(self, ind): 
        """Begin recursion from item with given ind(ex) and forms the full name in the self._full_name"""
        item = self.model.itemFromIndex(ind)
        self._full_name = item.text()
        self._getFullName(ind)
        return self._full_name

    def _getFullName(self, ind): 
        """Recursion from child to parent"""
        ind_par  = self.model.parent(ind)
        if(ind_par.column() == -1) :
            item = self.model.itemFromIndex(ind)
            self._full_name = '/' + self._full_name
            #print 'Item full name :' + self._full_name
            return str(self._full_name)
        else:
            item_par = self.model.itemFromIndex(ind_par)
            self._full_name = item_par.text() + '/' + self._full_name
            self._getFullName(ind_par)

    #-------------------
    #  Public methods --
    #-------------------

    def closeEvent(self, event): # if the x is clicked
        print 'closeEvent'
        self.processExit()

    def processExit(self):
        print 'Exit button is clicked'
        self.close()

    def processApply(self):
        print 'Apply button is clicked'
        cp.confpars.list_of_checked_item_names=[]

        list_of_checked_items = self.model.get_list_of_checked_items()
        for item in list_of_checked_items :
            item_full_name = self.getFullNameFromItem(item)
            cp.confpars.list_of_checked_item_names.append(item_full_name)
            print 'Checked item :', item_full_name
            
    def processExpand(self):
        print 'Expand button is clicked'
        self.view.expandAll()

    def processCollapse(self):
        print 'Collapse button is clicked'
        self.view.collapseAll()
 
    def processReset(self):
        print 'Reset button is clicked'

    def someMethod1(self):
        print '1-clicked!'

    def someMethod2(self):
        print '2-clicked!'

    def itemExpanded(self, ind): 
        item = self.model.itemFromIndex(ind)
        item.setIcon(self.icon_folder_open)
        print 'Item expanded : ', item.text()  

    def itemCollapsed(self, ind):
        item = self.model.itemFromIndex(ind)
        item.setIcon(self.icon_folder_closed)
        print 'Item collapsed : ', item.text()  

    def itemSelected(self, selected, deselected):
        print len(selected),   "items selected"
        print len(deselected), "items deselected"
        print 
        
        #print "item selected %s" % (QStandardItem(selected).text())

    def cellSelected(self, ind_sel, ind_desel):
        #print "ind   selected row, col = ", ind_sel.row(),  ind_sel.column()
        #print "ind deselected row, col = ", ind_desel.row(),ind_desel.column() 
        item       = self.model.itemFromIndex(ind_sel)
        dsfullname = str(self.getFullNameFromItem(item))
        print "Item with name '%s' is selected" % ( dsfullname )
        #print ' isEnabled=',item.isEnabled() 
        #print ' isCheckable=',item.isCheckable() 
        #print ' checkState=',item.checkState()
        #print ' isSelectable=',item.isSelectable() 
        #print ' isTristate=',item.isTristate() 
        #print ' isEditable=',item.isEditable() 
        #print ' isExpanded=',self.view.isExpanded(ind_sel)
        printh5.print_dataset_metadata_from_file(dsfullname)

    def itemChanged(self, item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        print "Item with full name %s, is at state %s\n" % ( self.getFullNameFromItem(item),  state)

#------------------
#  Main for test --
#------------------

def main():
    app = QtGui.QApplication(sys.argv)
    ex  = GUISelectItems()
    ex.show()
    app.exec_()

if __name__ == "__main__" :
    main()
    sys.exit ( "Module is not supposed to be run as main module" )

#------------------
