#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Draw...
#
#------------------------------------------------------------------------

"""Module Draw for CSPadAlignment package

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
import os
import time
import matplotlib.pyplot as plt

import DeviationFromLine as dfl
import GlobalMethods     as gm
import ImageParameters   as imp
import numpy             as np

#----------------------------------

class Draw :
    print """Draw all plots except the image"""
    def __init__ (self) :

        self.arr = imp.impars.arr
        self.arr = None
        self.figNum = 0

        self.list_of_open_figs = []

        self.dflmethods = dfl.DeviationFromLine()

        #imp.impars.printImageArray()
        #imp.impars.printLineCoordinates()
#----------------------------------

    def draw2DArray(self, arr=None):
        if arr == None : arr = imp.impars.arr
        gm.openFigure(1,12,12,'CSpad standalone')
        print 'dimX, dimY =', arr.shape
        gm.drawImage(arr, 'CSpad image')
        plt.draw()
        self.showPlot ( mode=1 )

#----------------------------------
    
    def drawSpectrum(self, arr=None, nbins=120, ARange=(1000,4000)):
        if arr == None : arr = imp.impars.arr
        self.figNum = 1        
        self.set_fig('1x1')
        plt.hist(arr.flatten(), bins=nbins, range=ARange)    
        plt.draw()
        plt.savefig('plot-spectrum-' + imp.impars.plot_fname_suffix + '.png')
        self.showPlot ( mode=1 )

#----------------------------------

    def drawImageProfileAlongLine(self, arr=None, lind=None):
        if arr  == None : arr = imp.impars.arr
        if lind == None : lind=len(imp.impars.line_coord) - 1

        print 'Profile for line =', lind 

        self.figNum = imp.impars.figNumBase + lind
        fig = self.set_fig('2x1')

        self.dflmethods.drawProfileAlongLine(imp.impars.arr,imp.impars.line_coord[lind], lind)

        plt.draw()
        plt.show()
        #self.showPlot ( mode=1 )

#----------------------------------

    def plotForMovedLine(self, lind):
        """Plot for moved line at relese mouse button"""

        linetype = imp.impars.line_coord[lind][4]
        if linetype == 'h' or linetype == 'v' :
            self.plotDeviationFromSingleLine(lind)
        if linetype == 'p' :
            self.drawImageProfileAlongLine(imp.impars.arr,lind) # None stands for default image array
            
#----------------------------------

    def plotDeviationFromSingleLine(self, lind):
        """Plot deviation from all lines"""

        self.figNum = imp.impars.figNumBase + lind
        fig = self.set_fig('2x1')
        self.dflmethods.evaluateDeviationFromLine(imp.impars.arr,imp.impars.line_coord[lind], lind,
                                                  imp.impars.APeak, imp.impars.AWireMin, imp.impars.AWireMax)

        plt.draw()
        plt.show()
        #self.showPlot ( mode=1 )

#----------------------------------

    def plotDeviationFromLines(self):
        """Plot deviation from all lines"""

        for lind in range(len(imp.impars.line_coord)) :
            if imp.impars.line_coord[lind][4] == 'p' : continue
            self.figNum = imp.impars.figNumBase + lind
            fig = self.set_fig('2x1')
            self.dflmethods.evaluateDeviationFromLine(imp.impars.arr,imp.impars.line_coord[lind], lind,
                                                      imp.impars.APeak, imp.impars.AWireMin, imp.impars.AWireMax)
            plt.draw()
        self.showPlot ( mode=1 )

#----------------------------------

    def set_fig( self, type=None ):
        """Set current fig."""
        self.set_fig_v2(type)

#----------------------------------

    def set_fig_v1( self, type=None ):
        """Set current fig."""

        ##self.figNum += 1 
        if self.figNum in self.list_of_open_figs :
            self.fig = plt.figure(num=self.figNum)        
            self.set_window_position()
        else :
            self.fig = self.open_fig(type)
            self.set_window_position()
            self.list_of_open_figs.append(self.figNum)
        self.fig.clf()
        return self.fig

#----------------------------------

    def set_fig_v2( self, type=None ):
        """Set current fig."""
            
        ##self.figNum += 1 
        if self.figNum in self.list_of_open_figs :
            self.close_fig(self.figNum)

        self.fig = self.open_fig(type)
        self.set_window_position()
        self.list_of_open_figs.append(self.figNum)
        #self.fig.clf()
        return self.fig

#----------------------------------

    def set_window_position( self ):
        """Move window in desired position."""

        self.figOffsetNum =    0
        self.figOffsetX   = 1650
        self.figOffsetY   =    0

        posx = self.figOffsetX + 30*(self.figNum-self.figOffsetNum)
        posy = self.figOffsetY + 15*(self.figNum-self.figOffsetNum-1)

        plt.get_current_fig_manager().window.move(posx,posy) #### This works!
        #fig_QMainWindow = plt.get_current_fig_manager().window
        #fig_QMainWindow.move(posx,posy)

#----------------------------------

    def open_fig( self, type=None ):

        #plt.ion() # enables interactive mode

        if   type == '1x1' :
            self.fig = plt.figure(num=self.figNum, figsize=(6,6), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        elif type == '1x2' :
            self.fig = plt.figure(num=self.figNum, figsize=(5,10), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        elif type == '2x3' :
            self.fig = plt.figure(num=self.figNum, figsize=(6,9), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        elif type == '2x1' :
            self.fig = plt.figure(num=self.figNum, figsize=(10,5), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        elif type == '3x4' :
            self.fig = plt.figure(num=self.figNum, figsize=(6,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        elif type == 4 :
            self.fig = plt.figure(num=self.figNum, figsize=(10,10), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.08, bottom=0.02, right=0.98, top=0.98, wspace=0.2, hspace=0.1)
        else :
            self.fig = plt.figure(num=self.figNum)

        self.cidclose = self.fig.canvas.mpl_connect('close_event', self.processCloseEvent)

        print 'Open figure number=', self.figNum
        return self.fig

#----------------------------------

    def processCloseEvent( self, event ):
        """Figure will be closed automatically, but it is necesary to remove its number from the list..."""
        fig    = event.canvas.figure # plt.gcf() does not work, because closed canva may be non active
        figNum = fig.number 
        print 'CloseEvent for figure number = ', figNum
        if figNum in self.list_of_open_figs :
            self.list_of_open_figs.remove(figNum)

        event.canvas.mpl_disconnect(self.cidclose)

#----------------------------------

    def close_fig( self, figNum=None ):
        print """Close fig and its window."""

        #plt.ioff()
        if figNum==None :
            plt.close('all') # closes all the figure windows
        else :
            plt.close(figNum) # this generates close_event and calls processCloseEvent()

#----------------------------------

    def showPlot ( self, mode=1 ) :
        """showEvent: plt.show() or draw() depending on mode"""

        t_start = time.clock()
        if mode == 1 :   # Single event mode
            plt.show()  
        else :           # Slide show 
            plt.draw()   # Draws, but does not block
        print 'Time to show or draw (sec) = %f' % (time.clock() - t_start)


    
#--------------------------------------
# Make a single object of this class --
#--------------------------------------

draw = Draw()  

#----------------------------------
def main():
    print 'Draw initialization and test'

if __name__ == '__main__':
    main()
#----------------------------------

