#-----------------------------
import sys
import ImgExplorer as imgex
from PyQt4 import QtGui, QtCore
#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    w  = imgex.ImgExplorer(None)
    w.move(QtCore.QPoint(10,10))
    w.set_image_array( w.get_array2d_with_ring_for_test() )
    w.show()

    app.exec_()        

#-----------------------------
if __name__ == "__main__" :
    main()
    sys.exit ('End of test')
#-----------------------------
