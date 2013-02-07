import numpy as np

array = np.loadtxt("testfile.txt")
array.shape

import XtcExplorer.cspad as cspad
mycspad = cspad.CsPad()
mycspad.make_image(array)
mycspad.image.shape

import  XtcExplorer.utilities as util
plotter = util.Plotter()
plotter.plot_image(mycspad.image)

