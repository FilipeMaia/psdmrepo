#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUICalibDirTree...
#
#------------------------------------------------------------------------

"""GUI works with dark run"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
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

#-----------------------------

class GUICalibDirTree (QtGui.QWidget):

    calib_types_cspad = [
        'center'
       ,'center_off'
       ,'offset'
       ,'offset_corr'
       ,'marg_gap_shift'
       ,'quad_rotation'
       ,'quad_tilt'
       ,'rotation'
       ,'tilt'
       ,'beam_vector'
       ,'beam_intersect'
       ,'pedestals'
       ,'pixel_status'
       ,'common_mode'
       ,'filter'
       ,'pixel_gain'
        ]

    calib_types_cspad2x2 = [
        'center'
       ,'tilt'     
       ,'pedestals'
       ,'pixel_status'
       ,'common_mode'
       ,'filter'
       ,'pixel_gain'
        ]
    
    calib_dets_cspad = [ 
        'XppGon.0:Cspad.0'
       ,'XcsEndstation.0:Cspad.0'
       ,'CxiDs1.0:Cspad.0'
       ,'CxiDsd.0:Cspad.0'
        ]

    calib_dets_cspad2x2 = [ 
        'XppGon.0:Cspad2x2.0'
       ,'XppGon.0:Cspad2x2.1'
       ,'MecTargetChamber.0:Cspad2x2.1'
       ,'MecTargetChamber.0:Cspad2x2.2'
       ,'MecTargetChamber.0:Cspad2x2.3'
       ,'MecTargetChamber.0:Cspad2x2.4'
       ,'MecTargetChamber.0:Cspad2x2.5'
       ,'CxiSc.0:Cspad2x2.0'
       ,'MecTargetChamber.0:Cspad2x2.1'
        ]

    calib_vers = [
        'CsPad::CalibV1'
       ,'CsPad2x2::CalibV1'
        ]


    def __init__(self, parent=None) :
        #super(GUIQTreeView, self).__init__(parent)
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 100, 250, 500)
        self.setWindowTitle('Item selection tree')


        #self.icon_folder_open   = QtGui.QIcon("icons/folder_open.gif")
        #self.icon_folder_closed = QtGui.QIcon("icons/folder_closed.gif")
        #self.icon_table         = QtGui.QIcon("icons/table.gif")

        self.fill_calib_dir_tree()

        #self.view = QtGui.QListView()
        #self.view = QtGui.QTableView()
        self.view = QtGui.QTreeView()
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



    def fill_calib_dir_tree(self) :

        self.model = QtGui.QStandardItemModel()
        self.model.setHorizontalHeaderLabels('x')
        #self.model.setVerticalHeaderLabels('abc')

        for v in self.calib_vers :
            det, vers = v.split('::',1)
            print 'det, vers =', det, vers

            parentItem = self.model.invisibleRootItem() 
            itemv = QtGui.QStandardItem(QtCore.QString(v))
            #itemv.setIcon(self.icon_table)
            itemv.setCheckable(True) 
            parentItem.appendRow(itemv)
  
            if det == 'CsPad' :
                self.calib_type_list = self.calib_types_cspad
                self.calib_det_list  = self.calib_dets_cspad
            elif det == 'CsPad2x2' :
                self.calib_type_list = self.calib_types_cspad2x2
                self.calib_det_list  = self.calib_dets_cspad2x2
            else :
                print 'UNKNOWN DETECTOR' 

            for d in self.calib_det_list :
                itemd = QtGui.QStandardItem(QtCore.QString(d))
                #itemd.setIcon(self.icon_table)
                itemd.setCheckable(True) 
                itemv.appendRow(itemd)
 
                for t in self.calib_type_list :
                    itemt = QtGui.QStandardItem(QtCore.QString(t))
                    itemt.setCheckable(True) 
                    itemd.appendRow(itemt)


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




    def itemExpanded(self, ind): 
        item = self.model.itemFromIndex(ind)
        #item.setIcon(self.icon_folder_open)
        print 'Item expanded : ', item.text()  

    def itemCollapsed(self, ind):
        item = self.model.itemFromIndex(ind)
        #item.setIcon(self.icon_folder_closed)
        print 'Item collapsed : ', item.text()  

    def itemSelected(self, selected, deselected):
        print 'items selected:'  , self.getFullNameFromIndex(selected)
        print 'items deselected:', self.getFullNameFromIndex(deselected)

    def itemChanged(self, item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        print "Item with full name %s, is at state %s\n" % ( self.getFullNameFromItem(item),  state)

 
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUICalibDirTree ()
    widget.show()
    app.exec_()

#-----------------------------
