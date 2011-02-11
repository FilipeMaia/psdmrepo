#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotsForImage...
#
#------------------------------------------------------------------------

"""Plots for any 'image' record in the EventeDisplay project.

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
import matplotlib.pyplot as plt
from numpy import *

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------
class PlotsForImage ( object ) :
    """Plots for any 'image' record in the EventeDisplay project.

    @see BaseClass
    @see OtherClass
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor, initialization"""
        pass

    #-------------------
    #  Public methods --
    #-------------------
  
    def plotImage( self, arr2d1ev, fig ):
        """Plot 2d image from input array."""

        fig.canvas.set_window_title("Image") 
        plt.clf() # clear plot
        fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)        
        
        pantit='Image, event ' + str(cp.confpars.eventCurrent)
        self.axes = plt.imshow(arr2d1ev, origin='upper', interpolation='nearest') # Just a histogram
        self.colb = plt.colorbar(self.axes, pad=0.01, fraction=0.10, shrink = 1) #, ticks=coltickslocs)
        plt.clim(cp.confpars.imageImageAmin,cp.confpars.imageImageAmax)
        
        plt.title(pantit,color='r',fontsize=20) # pars like in class Text
        plt.xlabel('X pixels')
        plt.ylabel('Y pixels')
        
        #plt.margins(x=0.05,y=0.05,tight=True)
        #plt.rc('lines', linewidth=2, color='r') # Set the current default parameters
        
        #str_event = 'Event ' + str(cp.confpars.eventCurrent)
        #plt.text(-50, -10, str_event, fontsize=24)

        #plt.savefig("my-image-hdf5.png")
        #plt.show()        


    def plotImageSpectrum( self, arr2d1ev, fig ):
        """Spectrum of amplitudes in the 2d input array."""

        plt.clf() # clear plot
        fig.canvas.set_window_title('Specrum') 
        pantit='Specrum, event ' + str(cp.confpars.eventCurrent)
        plt.title(pantit,color='r',fontsize=20) # pars like in class Text
        arrdimX,arrdimY = arr2d1ev.shape
        #print 'arr2d1ev.shape=', arr2d1ev.shape, arrdimX, arrdimY 
        print 'arr2d1ev=\n', arr2d1ev
        arr1d1ev = copy(arr2d1ev)
        arr1d1ev.resize(arrdimX*arrdimY)
        print 'arr1d1ev=\n', arr1d1ev
        #plt.hist(arr1d1ev,100)

        cp.confpars.imageSpectrumRange=(15,45)
        #cp.confpars.imageSpectrumNbins=30       
        #cp.confpars.imageSpectrumRange=None        
        #cp.confpars.imageSpectrumNbins=None        
        plt.hist(arr1d1ev, bins=cp.confpars.imageSpectrumNbins, range=(cp.confpars.imageSpectrumAmin,cp.confpars.imageSpectrumAmax))
        #plt.hist(arr1d1ev)


    def plotImageAndSpectrum( self, arr2d1ev, fig ):
        """Image and spectrum of amplitudes in the 2d input array."""
        #print 'Image and spectrum'

        self.fig = fig
        fig.canvas.set_window_title('Image and Spectrum')
        plt.clf() # clear plot
        fig.subplots_adjust(left=0.15, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)        
        
        #For Image 
        self.arr2d = arr2d1ev

        #print 'arr2d1ev.shape=', arr2d1ev.shape
        #print 'self.arr2d.shape=', self.arr2d.shape

        #For spectrum
        arrdimX,arrdimY = self.arr2d.shape
        self.arr1d = copy(arr2d1ev)
        self.arr1d.resize(arrdimX*arrdimY)            

        self.pantit =    'Event '   + str(cp.confpars.eventCurrent) 
        self.drawImageAndSpectrum(cp.confpars.imageImageAmin,cp.confpars.imageImageAmax)



    def drawImageAndSpectrum(self, Amin=None, Amax=None):
        """Plot 2d image from input array for a single pair"""

        ax2 = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
        #plt.subplot(212)
        self.axes1d = plt.hist(self.arr1d, bins=cp.confpars.imageSpectrumNbins, range=(Amin, Amax))
        #plt.xticks( arange(int(Amin), int(Amax), int((Amax-Amin)/3)) )
        colmin, colmax = plt.xlim()
        coltickslocs, coltickslabels = plt.xticks()
        #print 'colticks =', coltickslocs, coltickslabels
        
        ax1 = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=4)
        #plt.subplot(211)
        #print 'self.arr2d.shape=', self.arr2d.shape
        self.axes = plt.imshow(self.arr2d, origin='down', interpolation='nearest') # Just a histogram, origin='upper'
        plt.title(self.pantit,color='r',fontsize=20) # pars like in class Text

        #plt.text(50, -20, pantit, fontsize=24)
        self.colb = plt.colorbar(self.axes, pad=0.10, orientation=2, fraction=0.10, shrink = 1, aspect = 8, ticks=coltickslocs)

        plt.clim(colmin,colmax)
        #self.orglims = self.axes.get_clim()
           
        self.fig.canvas.mpl_connect('button_press_event', self.processMousButtonClick)

    def processMousButtonClick(self, event) :
       #print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y,
       #print ' xdata=',event.xdata,' ydata=', event.ydata
       if event.inaxes :
           lims = self.axes.get_clim()

           colmin = lims[0]
           colmax = lims[1]
           range = colmax - colmin
           value = colmin + event.xdata * range
           #print colmin, colmax, range, value

           # left button
           if event.button is 1 :
               if value > colmin and value < colmax :
                   colmin = value
                   print "new mininum: ", colmin
               else :
                   print "min has not been changed (click inside the color bar to change the range)"

           # middle button
           elif event.button is 2 :
               colmin, colmax = cp.confpars.imageImageAmin, cp.confpars.imageImageAmax
               print "reset"

           # right button
           elif event.button is 3 :
               if value > colmin and value < colmax :
                   colmax = value
                   print "new maximum: ", colmax
               else :
                   print "max has not been changed (click inside the color bar to change the range)"

           plt.clim(colmin,colmax)
           plt.clf()
           self.drawImageAndSpectrum(colmin,colmax)
           plt.draw() # redraw the current figure
    
        
#--------------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#--------------------------------
