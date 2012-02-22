#------------------------------
import sys
import ImgExplorer as imgex
from PyQt4 import QtGui, QtCore
#------------------------------

class ImgExplorerWithMyImages(imgex.ImgExplorer) :

    def __init__(self, parent=None, arr=None):
        imgex.ImgExplorer.__init__(self, None)
        self.myshape = (500,500)
        self.get_image( self.icp.imageCurrent )

    def get_image( self, imageFlag, increment=None) :
        """This method overwrites the get_image(...) in class ImgControl.
        imageFlag may take 3 values: self.icp.imagePrevious / imageCurrent / imageNext.
        The increment value may be used in transition to the next or previous image. 
        """
        print 'MY IMAGES SUPPLIED BY THE get_image(', imageFlag, ', increment =', increment, ')'

        if imageFlag == self.icp.imagePrevious :
            self.set_image_array( imgex.getSmouth2DArray(self.myshape) ) 
        if imageFlag == self.icp.imageCurrent :
            self.set_image_array( imgex.getRandomWithRing2DArray(self.myshape) )
        if imageFlag == self.icp.imageNext :
            self.set_image_array( imgex.getRandom2DArray(self.myshape) )

#------------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    w = ImgExplorerWithMyImages()
    w.move(QtCore.QPoint(10,10))
    w.show()

    app.exec_()        

#------------------------------
if __name__ == "__main__" :
    main()
    sys.exit ('End of test')
#------------------------------
