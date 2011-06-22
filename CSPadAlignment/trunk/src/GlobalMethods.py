#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GlobalMethods...
#
#------------------------------------------------------------------------

"""Module GlobalMethods for CSPadAlignment package

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
#import math # cos(x), sin(x), radians(x), degrees() 
import numpy as np
#import scipy.ndimage as spi             
import matplotlib.pyplot as plt
import matplotlib.lines  as lines
#----------------------------------

def get_item_last_name(dsname):
    """Returns the last part of the full item name (after last slash)"""
    path,name = os.path.split(str(dsname))
    return name

#----------------------------------

def get_item_path_to_last_name(dsname):
    """Returns the path to the last part of the item name"""
    path,name = os.path.split(str(dsname))
    return path

#----------------------------------

def get_item_path_and_last_name(dsname):
    """Returns the path and last part of the full item name"""
    path,name = os.path.split(str(dsname))
    return path, name

#----------------------------------

def get_item_second_to_last_name(dsname):
    """Returns the 2nd to last part of the full item name"""
    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    return name2 

#----------------------------------

def get_item_third_to_last_name(dsname):
    """Returns the 3nd to last part of the full item name"""
    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    path3,name3 = os.path.split(str(path2))
    str(name3)
    return name3 

#----------------------------------

def get_item_name_for_title(dsname):
    """Returns the last 3 parts of the full item name (after last slashes)"""
    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    path3,name3 = os.path.split(str(path2))

    return name3 + '/' + name2 + '/' + name1

#----------------------------------

def saveNumpyArrayInFile(arr, fname='nparray.txt', format='%f') : # format='%f'
    print """Save numpy array in file """, fname
    np.savetxt(fname, arr, fmt=format)

#----------------------------------

def getNumpyArrayFromFile(fname='nparray.txt', datatype=np.float32) : # np.int16, np.float16, np.float32
    print """Load numpy array from file """, fname
    return np.loadtxt(fname, dtype=datatype)

#----------------------------------

def openFigure(num=1,sx=10,sy=10,title='Title') :
    plt.ion() 
    fig = plt.figure(num, figsize=(sx,sy), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    fig.subplots_adjust(left=0.08, bottom=0.05, right=0.97, top=0.93, wspace=0, hspace=0)
    #fig.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97, wspace=0, hspace=0)
    fig.canvas.set_window_title(title) 
    plt.clf() 
    plt.get_current_fig_manager().window.move(20,20)
    return fig

#----------------------------------

def drawImage(arr2d, title) :
    """Draws the image and color bar for a single 2D array."""
    axes = plt.imshow(arr2d, interpolation='nearest', origin='upper') # origin='upper','lower'
    colb = plt.colorbar(axes, pad=0.06, fraction=0.1, shrink=0.97, aspect = 20, orientation=1) #, ticks=coltickslocs)
    plt.title(title, color='r', fontsize=20)
    #plt.clim(imageAmin,imageAmax)

#----------------------------------

def drawHistogram(arr, nbins=50, xrange=(0,100)) :
    """Draws histogram with reduced number of ticks for vertical axes for input array (dimension does not matter)."""
    plt.hist(arr.flatten(), bins=nbins, range=xrange)
    Ymin, Ymax = plt.ylim()
    #plt.yticks( np.arange(int(Ymin), int(Ymax), int((Ymax-Ymin)/3)) )

#----------------------------------

def addRectangle(xy=(100,200), w=100, h=200):
    rec = plt.Rectangle(xy, width=w, height=h, edgecolor='w', linewidth=2, fill=False)
    plt.gca().add_patch(rec)

#----------------------------------

def addLine(x=(0,1000), y=(0,1000)): # http://matplotlib.sourceforge.net/api/artist_api.html
    line = lines.Line2D(x, y, linewidth=1, color='w')
    plt.gca().add_line(line) 

#----------------------------------

def drawOrShow(showTimeSec=None) :

    if str(showTimeSec).lower() == 'show' :
        plt.ioff()
        plt.show()
    
    elif showTimeSec != None :
        plt.draw()
        plt.draw()
        print 'Sleep', showTimeSec, 'sec'
        time.sleep(showTimeSec)
        #plt.close(1)

    else :
        plt.draw()
        plt.draw()
        plt.draw()

#----------------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    sys.exit ( "Module is not supposed to be run as main module" )

#----------------------------------
