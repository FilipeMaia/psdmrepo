#-----------------------------
import sys
import ImgExplorer as imgex
from PyQt4 import QtGui, QtCore
#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    w  = imgex.ImgExplorer(None)
    w.move(QtCore.QPoint(10,10))
    #w.get_image( w.icp.imageCurrent )
    w.set_image_array( imgex.getRandomWithRing2DArray() ) # if you need in a single image only...
    w.show()

    app.exec_()        

#-----------------------------
if __name__ == "__main__" :
    main()
    sys.exit ('End of test')
#-----------------------------
