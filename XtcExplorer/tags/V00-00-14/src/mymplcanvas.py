import sys, random
from PyQt4 import QtGui, QtCore

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

class PlotterWindow(QtGui.QMainWindow):
    def __init__(self,parent=None):
        app = QtGui.QApplication(sys.argv)
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle("PlotterWindow")
        self.statusBar()

        self.setCentralWidget(QtGui.QWidget(self))

        self.canvas = MyMplCanvas(self.centralWidget())


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)
        
        self.compute_initial_figure()
        
        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')
        
        
        
class MyScalarCanvas(MyMplCanvas):
    def __init__(self,*args,**kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
                
        timer = QtCore.QTimer(self)
        QtCore.QObject.connect(timer, QtCore.SIGNAL("timeout()"), self.update_figure)
        timer.start(5000)
        
    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')
            
    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [ random.randint(0, 10) for i in xrange(4) ]
                
        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()
