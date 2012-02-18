#-----------------------------
import sys
import ImgExplorer as imgex
from PyQt4 import QtGui, QtCore
#-----------------------------


class ImgExplorerWithMyImages(imgex.ImgExplorer) :

    def __init__(self, parent=None, arr=None):
        imgex.ImgExplorer.__init__(self, None)

        self.myshape = (500,500)
        self.set_image_array( self.get_array2d_with_ring_for_test() )


    def get_image( self, eventFlag, increment=None) :
        print 'OWERWRITTEN get_image(', eventFlag, ', increment =', increment, ')'

        if eventFlag == self.icp.eventPrevious :
            self.set_image_array( imgex.getSmouth2DArray(self.myshape) ) 
        if eventFlag == self.icp.eventCurrent :
            self.set_image_array( self.get_array2d_with_ring_for_test() )
        if eventFlag == self.icp.eventNext :
            self.set_image_array( imgex.getRandom2DArray(self.myshape, mu=200, sigma=25) )

def main():

    app = QtGui.QApplication(sys.argv)

    w = ImgExplorerWithMyImages(None)
    w.move(QtCore.QPoint(10,10))
    w.show()

    app.exec_()        

#-----------------------------
if __name__ == "__main__" :
    main()
    sys.exit ('End of test')
#-----------------------------
