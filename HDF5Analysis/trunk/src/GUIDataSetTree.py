#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDataSetTree...
#
#------------------------------------------------------------------------

"""Shows the data set tree-structure and allows to select data items

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
import ConfigParameters     as cp
import AppUtils.AppDataPath as apputils
import DataSetTreeViewModel as dstvmodel

#---------------------
#  Class definition --
#---------------------

class GUIDataSetTree(QtGui.QWidget):
     
    def __init__(self, parent=None, window=0):
        super(GUIDataSetTree, self).__init__(parent)
        #QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 100, 200, 400)
        self.setWindowTitle('Item selection tree')
        self.setFrame()

        self.window = window
        self.dsname = cp.confpars.dsWindowParameters[self.window][0]

        self.butExpColl    = QtGui.QPushButton('Expand')
        self.butExpCheck   = QtGui.QPushButton('Exp.Checked')
        self.butApplyCheck = QtGui.QPushButton('Apply')
        self.butReset      = QtGui.QPushButton('Reset')
        self.butRetreve    = QtGui.QPushButton('Retreve')
        #self.butExit      = QtGui.QPushButton('Exit')

        self.model = dstvmodel.DataSetTreeViewModel(None,self.window)

        self.tree = QtGui.QTreeView()
        self.tree.setModel(self.model)
        #self.tree.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        #self.tree.expandAll()
        self.tree.setAnimated(True)
        
        self.hboxButs = QtGui.QHBoxLayout()
        self.hboxButs.addWidget(self.butExpColl)
        self.hboxButs.addWidget(self.butExpCheck)
        self.hboxButs.addWidget(self.butReset)
        self.hboxButs.addWidget(self.butRetreve)
        self.hboxButs.addWidget(self.butApplyCheck)
        #self.hboxButs.addWidget(self.butExit)

        self.vboxTree = QtGui.QVBoxLayout()
        self.vboxTree.addWidget(self.tree)
        self.vboxTree.addLayout(self.hboxButs)
        self.setLayout(self.vboxTree)

        self.connect(self.tree.selectionModel(), QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.on_itemClick)
        self.model.itemChanged.connect(self.on_itemChanged)
        self.tree.expanded.connect(self.itemExpanded)
        self.tree.collapsed.connect(self.itemCollapsed)
        self.connect(self.butExpColl,    QtCore.SIGNAL('clicked()'), self.processExpColl )
        self.connect(self.butExpCheck,   QtCore.SIGNAL('clicked()'), self.processExpChecked )
        self.connect(self.butApplyCheck, QtCore.SIGNAL('clicked()'), self.processApplyChecked )
        self.connect(self.butReset,      QtCore.SIGNAL('clicked()'), self.processReset )
        self.connect(self.butRetreve,    QtCore.SIGNAL('clicked()'), self.processRetreve )
        #self.connect(self.butExit,     QtCore.SIGNAL('clicked()'), self.processExit )

        self.showToolTips()
        if not cp.confpars.isSetWarningModel :
            self.processExpChecked() # SHOULD NOT DO THAT FOR WARNING MODEL...


    def showToolTips(self):
        # Tips for buttons and fields:
        #self.butExit.setToolTip('Click on mouse left button\n' +\
        #                        'in order to exit.')
        self.butExpColl.   setToolTip('Expand/Collapse all groups in the dataset tree.')
        self.butExpCheck.  setToolTip('Expand groups with checked items only.')
        self.butReset.     setToolTip('Reset all checked items to unchecked state in the dataset tree.')
        self.butRetreve.   setToolTip('Retreve checked items in the dataset tree from configuration parameters.')
        self.butApplyCheck.setToolTip('Apply checked items in the dataset tree to configuration parameters.')

    #def setDSName(self, dsname):
    #    if dsname=='None' : title = 'Dataset substructure'
    #    else              : title = dsname
    #    self.model.setHorizontalHeaderItem(0,QtGui.QStandardItem(title))

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def closeEvent(self, event): # if the x is clicked
        print 'closeEvent'
        self.processExit()

    def processExit(self):
        print 'Exit button is clicked'
        self.close()

    def processExpand(self):
        print 'Expand   the dataset tree'
        self.tree.expandAll()
        self.model.set_all_group_icons(self.model.icon_folder_open)
        cp.confpars.dsTreeIsExpanded = True

    def processCollapse(self):
        print 'Collapse the dataset tree'
        self.tree.collapseAll()
        self.model.set_all_group_icons(self.model.icon_folder_closed)
        cp.confpars.dsTreeIsExpanded = False

    def processExpColl(self): # Flip/flop between Expand and Collaple the HDF5 tree
        print 'Expand/Collapse button is clicked :',
        if cp.confpars.dsTreeIsExpanded :
            self.butExpColl.setText('Expand')
            self.processCollapse()
        else :
            self.butExpColl.setText('Collapse')
            self.processExpand()
 
    def processExpChecked(self):
        print 'ExpandChecked button is clicked, expand the tree for checked items only.'
        self.processCollapse() # first, collapse the tree
        self.model.expand_checked_items(self.tree)
        self.butExpColl.setText('Collapse')
        cp.confpars.dsTreeIsExpanded = True       # Change status for expand/collapse button

    def processReset(self):
        print 'Reset button is clicked, reset all checked items in unchecked state.'
        self.model.reset_checked_items()

    def processRetreve(self):
        print 'Retreve button is clicked, retreve all checked items from the config. par. list.'
        self.model.retreve_checked_items()
        self.processExpChecked()

    #def processApply(self):
    #    print 'Apply button is clicked'

    #def processReset(self):
    #    print 'Reset button is clicked'

    def itemExpanded(self, ind):
        item = self.model.itemFromIndex(ind)
        item.setIcon(self.model.icon_folder_open)
        print 'Item expanded : ', item.text()  

    def itemCollapsed(self, ind):
        item = self.model.itemFromIndex(ind)
        item.setIcon(self.model.icon_folder_closed)
        print 'Item collapsed : ', item.text()  

    def itemSelected(self, selected, deselected):
        print len(selected),   "items selected"
        print len(deselected), "items deselected"
        print 
        #print "item selected %s" % (QStandardItem(selected).text())

    def processApplyChecked(self):
        print 'ApplyChecked is clicked'
        cp.confpars.dsWindowParameters[self.window][2] = list = self.model.get_list_of_checked_items()
        cp.confpars.dsWindowParameters[self.window][1] = len(list)
        print 'List of checked dataset indexes:'
        for dsindexes in list : print dsindexes           

 
    def on_itemClick(self, ind_sel, ind_desel):
        item = self.model.itemFromIndex(ind_sel)
        print '\n\n====> Clicked on item with'
        print 'text : %s' % (item.text()),
        print ', checkState=',item.checkState(), 
        print ', isExpanded=',self.tree.isExpanded(ind_sel)
        print 'The dataset indexes:', self.model.get_full_path_to_item(item)
        print 'The dataset shape dims:', self.model.get_dataset_dims(item)
        prod_of_dims = self.model.get_dataset_prod_of_dims(item)
        if prod_of_dims < 10000 :
            ds = self.model.get_dataset(item)
            print 'ds:\n',    ds
        else :
            ds_0 = self.model.get_dataset_0(item) 
            print 'ds[0]:\n', ds_0

        #print "ind   selected : ", ind_sel.row(),  ind_sel.column()
        #print "ind deselected : ", ind_desel.row(),ind_desel.column() 
        #print ' isSelectable=',item.isSelectable(), 
        #print ' isTristate=',item.isTristate(), 
        #print ' isEditable=',item.isEditable(), 
        #print ' isEnabled=',item.isEnabled(), 
        #print ' isCheckable=',item.isCheckable(), 

        #ind_par  = self.model.parent(ind_sel)
        #if(ind_par.column() != -1):
        #    item_par = self.model.itemFromIndex(ind_par)
        #    print " has parent '%s' \n" % ( item_par.text() )

    def on_itemChanged(self,  item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        print '\n\n====> Checked item with'
        print "text : '%s', is at state %s" % ( item.text(),  state)
        print 'The dataset indexes:', self.model.get_full_path_to_item(item)
        print 'The dataset shape dims:', self.model.get_dataset_dims(item)

#---------------------------------- 

def main():
    app = QtGui.QApplication(sys.argv)
    widget = GUIDataSetTree()
    widget.show()
    app.exec_()

if __name__ == '__main__':
    main()

#---------------------------------- 

