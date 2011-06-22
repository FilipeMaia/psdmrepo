#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module MyNavigationToolbar...
#
#------------------------------------------------------------------------

"""Module MyNavigationToolbar for CSPadAlignment package

CSPadAlignment package is intended to check quality of the CSPad alignment
using image of wires illuminated by flat field.
Shadow of wires are compared with a set of straight lines, which can be
interactively adjusted.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$
#------------------------------
#!/usr/bin/env python
#----------------------------------
#import sys
#import os
#import random

#import numpy as np

#matplotlib.use('GTKAgg') # forse Agg rendering to a GTK canvas (backend)
#matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

#from PyQt4 import QtGui, QtCore

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
#import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------
class MyNavigationToolbar ( NavigationToolbar ) :
    """ Need to re-emplement a few methods in order to get control on toolbar button click"""

    #def __init__(self, canvas):
        #print 'MyNavigationToolbar.__init__'
        #self.canvas = canvas
        #self.coordinates = True
        ##QtGui.QToolBar.__init__( self, parent )
        #NavigationToolbar2.__init__( self, canvas, None )
        #self = canvas.toolbar


    def home(self, *args) :
        print 'Home is clicked'
        fig = self.canvas.figure
        fig.myXmin = None
        fig.myXmax = None
        fig.myYmin = None
        fig.myYmax = None
        NavigationToolbar.home(self)


    def zoom(self, *args) :
        print 'Zoom is clicked'
        NavigationToolbar.zoom(self)
        #self.canvas.draw()
        #fig  = self.canvas.figure
        #axes = fig.gca() 
        #bounds = axes.viewLim.bounds
        #fig.myXmin = bounds[0]
        #fig.myXmax = bounds[0] + bounds[2] 
        #fig.myYmin = bounds[1] + bounds[3]
        #fig.myYmax = bounds[1]
        #fig.myZoomIsOn = True
        #print 'zoom: Xmin, Xmax, Ymin, Ymax =', fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax


    #def back(self, *args) :
    #    print 'Back is clicked'
    #    NavigationToolbar.back(self)


    #def forward(self, *args):
    #    print 'Forward is clicked'
    #    NavigationToolbar.forward(self)


    #def pan(self,*args):
    #    print 'Pan is clicked'
    #    NavigationToolbar.pan(self)


    #def save_figure(self, *args):
    #    print 'Save is clicked'
    #    NavigationToolbar.save_figure(self)

#-----------------------------

if __name__ == "__main__" :
    print 'This module is not supposed to run as a standalone job'

#-----------------------------
