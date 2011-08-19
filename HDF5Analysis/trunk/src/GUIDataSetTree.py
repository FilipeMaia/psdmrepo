#!/usr/bin/env python

import sys
from PyQt4 import QtGui, QtCore
import AppUtils.AppDataPath as apputils

class GUIDataSetTree(QtGui.QWidget):
     
    def __init__(self, parent=None):
        super(GUIDataSetTree, self).__init__(parent)
        #QtGui.QWidget.__init__(self, parent)

        # !!!!! THE /data/ SUBDIRECTORY SHOULD BE OMITTED IN PATH !!!!!
        self.apppath_icon_folder_open   = apputils.AppDataPath('HDF5Analysis/icons/folder_open.gif')
        self.apppath_icon_folder_closed = apputils.AppDataPath('HDF5Analysis/icons/folder_closed.gif')
        self.apppath_icon_data          = apputils.AppDataPath('HDF5Analysis/icons/table.gif')

        self.icon_folder_open   = QtGui.QIcon(self.apppath_icon_folder_open  .path())
        self.icon_folder_closed = QtGui.QIcon(self.apppath_icon_folder_closed.path())
        self.icon_data          = QtGui.QIcon(self.apppath_icon_data         .path())

        self.setGeometry(100, 100, 200, 400)
        self.setWindowTitle('Item selection tree')
        self.setFrame()

        self.butExit = QtGui.QPushButton('Exit')

        self.model = QtGui.QStandardItemModel()
        self.fillTreeModelExample()

        self.model.setHorizontalHeaderItem(0,QtGui.QStandardItem('Dataset substructure'))

        self.tree = QtGui.QTreeView()
        self.tree.setModel(self.model)
        #self.tree.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        #self.tree.expandAll()
        self.tree.setAnimated(True)
        
        self.vboxTree = QtGui.QVBoxLayout()
        self.vboxTree.addWidget(self.tree)
        self.vboxTree.addWidget(self.butExit)
        self.setLayout(self.vboxTree)

        self.connect(self.tree.selectionModel(), QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.cellSelected)
        self.model.itemChanged.connect(self.on_itemChanged)
        self.tree.expanded.connect(self.itemExpanded)
        self.tree.collapsed.connect(self.itemCollapsed)
        self.connect(self.butExit, QtCore.SIGNAL('clicked()'), self.processExit )

        self.showToolTips()


    def fillTreeModelExample(self) :

        for k in range(0, 5):
            parentItem = self.model.invisibleRootItem()
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

    def showToolTips(self):
        # Tips for buttons and fields:
        self.butExit.setToolTip('Click on mouse left button\n' +\
                                'in order to exit.')

    def setDSName(self, dsname):
        if dsname=='None' : title = 'Dataset substructure'
        else              : title = dsname
        self.model.setHorizontalHeaderItem(0,QtGui.QStandardItem(title))

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

    def processApply(self):
        print 'Apply button is clicked'

    def processReset(self):
        print 'Reset button is clicked'

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
        #print "ind   selected : ", ind_sel.row(),  ind_sel.column()
        #print "ind deselected : ", ind_desel.row(),ind_desel.column() 
        item     = self.model.itemFromIndex(ind_sel)
        print "Item with text '%s' is selected" % ( item.text() ),
        #print ' isEnabled=',item.isEnabled(), 
        #print ' isCheckable=',item.isCheckable(), 
        print ' checkState=',item.checkState(), 
        #print ' isSelectable=',item.isSelectable(), 
        #print ' isTristate=',item.isTristate(), 
        #print ' isEditable=',item.isEditable(), 
        print ' isExpanded=',self.tree.isExpanded(ind_sel)
        
        ind_par  = self.model.parent(ind_sel)
        if(ind_par.column() != -1):
            item_par = self.model.itemFromIndex(ind_par)
            print " has parent '%s' \n" % ( item_par.text() )

    def on_itemChanged(self,  item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        print "Item with text '%s', is at state %s\n" % ( item.text(),  state)
        #print "Item with text '%s' is changed\n" % ( item.text() )


#---------------------------------- 

def main():
    app = QtGui.QApplication(sys.argv)
    widget = GUIDataSetTree()
    widget.show()
    app.exec_()

if __name__ == '__main__':
    main()

#---------------------------------- 

