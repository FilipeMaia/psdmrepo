#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDirTree...
#
#------------------------------------------------------------------------

"""GUI works with dark run"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#-------------------
#  Import modules --
#-------------------
import sys
import os

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

from ConfigParametersForApp import cp
from Logger                 import logger

#----------------------------

class GUIDirTree (QtGui.QWidget):

    def __init__(self, parent=None, dir_top='.') :
        #super(GUIQTreeView, self).__init__(parent)
        QtGui.QWidget.__init__(self, parent)

        self.dir_top = dir_top

        self.setGeometry(100, 100, 200, 300)
        self.setWindowTitle('Item selection tree')

        self.setFrame()

        cp.setIcons()
 
        #self.icon_folder_open   = QtGui.QIcon("icons/folder_open.gif")
        #self.icon_folder_closed = QtGui.QIcon("icons/folder_closed.gif")
        #self.icon_table         = QtGui.QIcon("icons/table.gif")

        #self.view = QtGui.QListView()
        #self.view = QtGui.QTableView()
        self.view = QtGui.QTreeView()
 
        self.make_model() # defines self.model

        self.view.setModel(self.model)
        #self.view.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        #self.view.expandAll()
        self.view.setAnimated(True)
 
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.view)

        if parent == None :
            self.setLayout(vbox)

        self.connect(self.view.selectionModel(), QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.itemSelected)
        #self.view.clicked.connect(self.someMethod1)       # This works
        #self.view.doubleClicked.connect(self.someMethod2) # This works
        self.model.itemChanged.connect(self.itemChanged)
        self.view.expanded.connect(self.itemExpanded)
        self.view.collapsed.connect(self.itemCollapsed)
        #self.view.expandToDepth(2)

        self.view.expandAll()
        
        self.setStyle()

        cp.guidirtree = self


    def make_model(self) :

        if cp.dirtreemodel is None :
            cp.dirtreemodel = self.model = QtGui.QStandardItemModel()
            self.model.setHorizontalHeaderLabels('x')
            self.update_dir_tree(self.dir_top)
        else :
            self.model = cp.dirtreemodel 

        #self.model.setHorizontalHeaderItem(1,QtGui.QStandardItem('Project Title'))
        #self.model.setVerticalHeaderLabels('abc')


#----------------------------

    def update_dir_tree(self, dir_top=None) :
        if dir_top is not None :
            self.dir_top = dir_top
        self.model.clear()
        self.fill_dir_tree(self.dir_top)
        self.view.expandAll()

#----------------------------

    def fill_dir_tree(self, path, item=None, level=0) :

        if not os.path.exists(path) :
            return #'Path %s DOES NOT EXIST' % path

        item_parent = self.model.invisibleRootItem() if item is None else item

        item_add = QtGui.QStandardItem(QtCore.QString(os.path.basename(path)))
        item_parent.appendRow(item_add)

        if os.path.isfile(path) :
            item_add.setIcon(cp.icon_table)
            item_add.setCheckable(True) 
            return

        elif os.path.isdir(path) :
            item_add.setIcon(cp.icon_folder_open)
            #    ind = self.model.indexFromItem(item_add)
            #    self.view.setExpanded(ind,True)
            #    self.view.expand(ind)
            #else :
            #    item_add.setIcon(cp.icon_folder_closed)
                
            list_of_fnames = sorted(os.listdir(path))
            if list_of_fnames == [] :
                return # 'Directory %s IS EMPTY!' %  path     

            for fname in list_of_fnames :
                path_to_child = os.path.join(path, fname)
                self.fill_dir_tree(path_to_child, item=item_add, level=level+1)

        else : # for links etc
            pass

#----------------------------

    def get_list_of_checked_item_names(self) :
        basedir = os.path.dirname(self.dir_top)
        list = []
        for ind in self.model.persistentIndexList() :
            item = self.model.itemFromIndex(ind)
            if item.checkState() :
                fname = os.path.join(basedir, self.getFullNameFromIndex(ind))
                #print fname
                list.append(fname)
        return list

#----------------------------

    def getFullNameFromItem(self, item): 
        #item = self.model.itemFromIndex(ind)        
        ind   = self.model.indexFromItem(item)        
        return self.getFullNameFromIndex(ind)


    def getFullNameFromIndex(self, ind): 
        item = self.model.itemFromIndex(ind)
        if item is None : return 'None'
        self._full_name = item.text()
        self._getFullName(ind)
        return str(self._full_name)


    def _getFullName(self, ind): 
        ind_par  = self.model.parent(ind)
        if(ind_par.column() == -1) :
            item = self.model.itemFromIndex(ind)
            self.full_name = '/' + self._full_name
            #print 'Item full name :' + self._full_name
            return self._full_name
        else:
            item_par = self.model.itemFromIndex(ind_par)
            self._full_name = item_par.text() + '/' + self._full_name
            self._getFullName(ind_par)


    def getTextListOfChildren(self, index): 
        item = self.model.itemFromIndex(index)
        number_of_children = item.rowCount()
        txt_list_of_children = []
        for row in range(number_of_children) :
            child_item = item.child(row)
            txt_list_of_children.append(str(child_item.text()))
        return txt_list_of_children


    def itemChanged(self, item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        msg = 'Item %s, is at state %s' % ( self.getFullNameFromItem(item),  state)
        #print msg
        #logger.info(msg, __name__)       
        #print 'col:', self.model.columnCount(self.model.indexFromItem(item))
        #print 'row:', self.model.rowCount(self.model.indexFromItem(item))

        #list = self.get_list_of_checked_item_names()

        
    def itemExpanded(self, ind): 
        item = self.model.itemFromIndex(ind)
        item.setIcon(cp.icon_folder_open)
        msg = 'Item expanded: %s' % item.text()  
        logger.info(msg, __name__)       


    def itemCollapsed(self, ind):
        item = self.model.itemFromIndex(ind)
        item.setIcon(cp.icon_folder_closed)
        msg = 'Item collapsed: %s' % item.text()  
        logger.info(msg, __name__)       


    def itemSelected(self, selected, deselected):
        pass
        #msg1 = 'Item selected: %s' % self.getFullNameFromIndex(selected)
        #logger.info(msg1, __name__)       

        #txt_list_of_children = self.getTextListOfChildren(selected)
        
        #selected_txt = self.getFullNameFromIndex(selected)
        #self.onSelectedItem(selected_txt, txt_list_of_children)

        #deselected_txt = self.getFullNameFromIndex(deselected)
        #msg2 = 'Item deselected: %s' % self.getFullNameFromIndex(deselected)
        #logger.info(msg2, __name__)       
        #self.onDeselectedItem(deselected_txt)


    def onSelectedItem(self, path_from_calib, list_expected) :
        #cp.guitabs.setTabByName('Status')
        #cp.guistatus.statusOfDir(dir, list_expected)
        #self.get_list_of_checked_items()
        pass



    def setStyle(self):
        pass
        #self.setMinimumSize(100,200)
        #self.setMinimumSize(200,300)
        #self.setFixedWidth(300)
        self.setMinimumWidth(200)
        #self.setMinimumHeight(300)
        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
        
        #self.view.setMinimumWidth(200)

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        cp.guidirtree = None
 
#----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIDirTree (None, '/reg/d/psdm/SXR/sxrb5914/calib')
    #widget = GUIDirTree (None, '.')
    widget.show()
    app.exec_()

#----------------------------
