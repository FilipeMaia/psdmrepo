#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplay...
#
#------------------------------------------------------------------------

"""Generates GUI to select information for rendaring in the event display.

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

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplay ( QtGui.QWidget ) :
    """Provides GUI to select information for rendering.

    Detailed description should be here...
    @see BaseClass
    @see OtherClass
    """

    #--------------------
    #  Class variables --
    #--------------------
    #publicStaticVariable = 0 
    #__privateStaticVariable = "A string"

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(80, 80, 250, 150)
        self.setWindowTitle('Checkbox')

        self.cb  = QtGui.QCheckBox('Show title', self)
        self.cb2 = QtGui.QCheckBox('CB2', self)
        self.cb.setFocusPolicy(QtCore.Qt.NoFocus) 
        self.cb2.move(10, 50)
        self.cb.move(10, 10)
        #self.cb.toggle() # set the check mark in the check box
        self.connect(self.cb, QtCore.SIGNAL('stateChanged(int)'), self.changeTitle)

    def changeTitle(self, value):
        if self.cb.isChecked():
            self.setWindowTitle('Checkbox')
        else:
            self.setWindowTitle('')

#    def paintEvent(self, e):
#        qp = QtGui.QPainter()
#        qp.begin(self)
#        self.drawWidget(qp)
#        qp.end()


def main():
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplay()
    ex.show()
    app.exec_()

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    main()

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
