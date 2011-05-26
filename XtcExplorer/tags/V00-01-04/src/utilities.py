#-----------------------------------------
# PyanaOptions
#-----------------------------------------

class PyanaOptions( object ):
    def __init__( self ):
        pass

    def getOptString(self, options_string) :
        """
        parse the option string,
        return the string, strip of any whitespaces
        """
        if options_string is None:
            return None

        # make sure there are no newline characters here
        options_string = options_string.strip()

        if ( options_string == "" or
             options_string == "None" or
             options_string == "No" ) :
            return None

        # all other cases:
        return options_string


    def getOptStrings(self, options_string) :
        """
        parse the option string,
        return a list of strings
        """
        if options_string is None:
            return None

        # make sure there are no newline characters here
        options_string = options_string.split("\n")
        options_string = " ".join(options_string)

        # make a list
        options = options_string.split(" ")

        if len(options)==0 :
            return []

        elif len(options)==1 :
            if ( options_string == "" or
                 options_string == "None" or
                 options_string == "No" ) :
                return []

        # all other cases:
        return options


    def getOptIntegers(self, options_string):
        """Return a list of integers
        """
        if options_string is None: return None

        opt = self.getOptStrings(options_string)
        N = len(opt)
        if N is 1:
            return int(opt)
        if N > 1 :
            items = []
            for item in opt :
                items.append( int(item) )
            return items
            

    def getOptInteger(self, options_string):
        """Return a single integer
        """
        if options_string is None: return None

        if options_string == "" : return None
        return int(options_string)


    def getOptBoolean(self, options_string):
        """Return a boolean
        """
        if options_string is None: return None

        opt = options_string
        if   opt == "False" or opt == "0" or opt == "No" or opt == "" : return False
        elif opt == "True" or opt == "1" or opt == "Yes" : return True
        else :
            print "utilities.getOptBoolean: cannot parse option ", opt
            return None

    def getOptBooleans(self, options_string):
        """Return a list of booleans
        """
        if options_string is None: return None

        opt_list = self.getOptStrings(options_string)
        N = len(opt_list)
        if N == 0 : return None
        
        opts = []
        for opt in optlist :
            opts.append( self.getOptBoolean(opt) )

        return opts


    def getOptFloats(self, options_string):
        """Return a list of integers
        """
        if options_string is None: return None

        opt = self.getOptStrings(options_string)
        N = len(opt)
        if N is 1:
            return float(opt)
        if N > 1 :
            items = []
            for item in opt :
                items.append( float(item) )
            return items
            

    def getOptFloat(self, options_string):
        """Return a single integer
        """
        if options_string is None: return None
        if options_string == "" : return None
        if options_string == "None" : return None

        return float(options_string)


#-----------------------------------------
# Data Storage Classes
#-----------------------------------------

class BaseData( object ):
    """ Base class for container objects
    storing event data in memory (as numpy arrays mainly).
    Useful for passing the data to e.g. ipython for further investigation
    """
    def __init__(self, name,type="BaseData"):
        self.name = name
        self.type = type
        
    def __str__( self ):
        itsme = "<%s object with name %s>" % (self.type, self.name)
        return itsme

    def __repr__( self ):
        itsme = "<%s object with name %s>" % (self.type, self.name)
        return itsme


class BldData( BaseData ):
    def __init__(self, name, type="BldData"):
        BaseData.__init__(self,name,type)
        self.time = None
        self.damage = None
        self.energy = None
        self.position = None
        self.angle = None
        self.charge = None
        self.fex_sum = None
        self.fex_channels = None
        self.raw_channels = None
        self.fex_position = None

    def show( self ):
        itsme = "\n%s: \n\t name = %s" % (self.type, self.name)
        if self.time is not None :
            itsme+="\n\t time = array of shape %s"%str(np.shape(self.time))
        if self.damage is not None :
            itsme+="\n\t damage = array of shape %s"%str(np.shape(self.damage))
        if self.energy is not None :
            itsme+="\n\t energy = array of shape %s"%str(np.shape(self.energy))
        if self.position is not None :
            itsme+="\n\t position = array of shape %s"%str(np.shape(self.position))
        if self.angle is not None :
            itsme+="\n\t angle = array of shape %s"%str(np.shape(self.angle))
        if self.charge is not None :
            itsme+="\n\t charge = array of shape %s"%str(np.shape(self.charge))
        if self.fex_sum is not None :
            itsme+="\n\t fex_sum = array of shape %s"%str(np.shape(self.fex_sum))
        if self.fex_channels is not None :
            itsme+="\n\t fex_channels = array of shape %s"%str(np.shape(self.fex_channels))
        if self.raw_channels is not None :
            itsme+="\n\t raw_channels = array of shape %s"%str(np.shape(self.raw_channels))
        if self.fex_position is not None :
            itsme+="\n\t fex_position = array of shape %s"%str(np.shape(self.fex_position))
        print itsme


class IpimbData( BaseData ):
    def __init__( self, name, type="IpimbData" ):
        BaseData.__init__(self,name,type)
        self.fex_sum = None
        self.fex_channels = None
        self.fex_position = None
        self.raw_channels = None

    def show( self ):
        """Printable description 
        """
        itsme = "\n%s: \n\t name = %s" % (self.type, self.name)
        if self.fex_sum is not None :
            itsme+="\n\t fex_sum = array of shape %s"%str(np.shape(self.fex_sum))
        if self.fex_channels is not None :
            itsme+="\n\t fex_channels = array of shape %s"%str(np.shape(self.fex_channels))
        if self.fex_position is not None :
            itsme+="\n\t fex_position = array of shape %s"%str(np.shape(self.fex_position))
        if self.raw_channels is not None :
            itsme+="\n\t raw_channels = array of shape %s"%str(np.shape(self.raw_channels))
        print itsme


class EpicsData( BaseData ):
    def __init__( self, name, type="EpicsData" ):
        BaseData.__init__(self,name,type)
        self.values = None
        self.shotnr = None
        self.status = None
        self.severity = None

    def show( self ):
        itsme = "\n%s: \n\t name = %s" % (self.type, self.name)
        if self.values is not None :
            itsme+="\n\t values = array of shape %s"%str(np.shape(self.values))
        if self.shotnr is not None :
            itsme+="\n\t shotnr = array of shape %s"%str(np.shape(self.shotnr))
        if self.status is not None :
            itsme+="\n\t status = array of shape %s"%str(np.shape(self.status))
        if self.severity is not None :
            itsme+="\n\t severity = array of shape %s"%str(np.shape(self.severity))
        print itsme


class ScanData( BaseData ) :
    def __init__(self, name, type="ScanData"):
        BaseData.__init__(self,name,type)

        self.scanvec = None
        self.arheader = None
        self.scandata = None

    def show(self):
        itsme = "\n%s: \n\t name = %s" % (self.type, self.name)
        if self.scanvec is not None :
            itsme+="\n\t scanvec = array of shape %s"%str(np.shape(self.scanvec))
        if self.arheader is not None :
            itsme+="\n\t arheader = list of scan data %s"% self.arheader 
        if self.scandata is not None :
            itsme+="\n\t scandata = array of shape %s"%str(np.shape(self.scandata))
        print itsme



class ImageData( BaseData ):
    def __init__(self, name, type="ImageData"):
        BaseData.__init__(self,name,type)
        self.image = None
        self.average = None
        self.dark = None

    def show( self ):
        itsme = "\n%s \n\t name = %s" % (self.type, self.name)
        if self.image is not None :
            itsme+="\n\t image = array of shape %s"%str(np.shape(self.image))
        if self.average is not None :
            itsme+="\n\t average = array of shape %s"%str(np.shape(self.average))
        if self.dark is not None :
            itsme+="\n\t dark = array of shape %s"%str(np.shape(self.dark))
        print itsme

class CsPadData( BaseData ):
    def __init__(self, name, type="CsPadData"):
        BaseData.__init__(self,name,type)
        self.image = None
        self.average = None
        self.dark = None

    def show( self ):
        itsme = "\n%s: \n\t name = %s" % (self.type, self.name)
        if self.image is not None :
            itsme+="\n\t image = array of shape %s"%str(np.shape(self.image))
        if self.average is not None :
            itsme+="\n\t average = array of shape %s"%str(np.shape(self.average))
        if self.dark is not None :
            itsme+="\n\t dark = array of shape %s"%str(np.shape(self.dark))
        print itsme



#-------------------------------------------------------
# Threshold
#-------------------------------------------------------

class Threshold( object ) :
    def __init__( self,
                  area = None,
                  minvalue = None,
                  maxvalue = None,
                  ) :

        self.area = area        
        self.minvalue = minvalue
        self.maxvalue = maxvalue



#-------------------------------------------------------
# Plotter
#-------------------------------------------------------

import time
import numpy as np
import matplotlib.pyplot as plt
from PyQt4 import QtCore

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid

        
class Plotter(object):
    def __init__(self):
        self.plot_vmin = None
        self.plot_vmax = None
        self.cid1 = None
        self.cid2 = None
        self.colb = None
        self.grid = None
        self.threshold = None
        self.shot_number = None
        self.display_mode = None

        # matplotlib backend is set to QtAgg, and this is needed to avoid
        # a bug in raw_input ("QCoreApplication::exec: The event loop is already running")
        QtCore.pyqtRemoveInputHook()

        
    def connect(self):

        if self.cid1 is None:
            self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        if self.cid2 is None:
            self.cid2 = self.fig.canvas.mpl_connect('pick_event', self.onpick)


    def draw_figurelist(self, fignum, event_display_images ) :
        """
        @fignum                  figure number, i.e. fig = plt.figure(num=fignum)
        @event_display_images    a list of tuples (title,image)
        """
        axspos = 0

        nplots = len(event_display_images)
        ncol = 3
        if nplots<3 : ncol = nplots
        nrow = int( nplots/ncol)
        fig = plt.figure(fignum,(5.0*ncol,4*nrow))
        fig.clf()
        fig.suptitle("Event#%d"%self.shot_number)


        pos = 0
        self.caxes = [] # list of references to colorbar Axes
        self.axims = [] # list of references to image Axes
        for ad, im in sorted(event_display_images) :
            pos += 1
            
            # Axes
            ax = fig.add_subplot(nrow,ncol,pos)
            ax.set_title( "%s" % ad )

            # AxesImage
            axim = plt.imshow( im, origin='lower' )
            self.axims.append( axim )
        
            cbar = plt.colorbar(axim,pad=0.02,shrink=0.78) 
            self.caxes.append( cbar.ax )
            
            self.orglims = axim.get_clim()
            # min and max values in the axes are


        plt.draw()

    def drawframe( self, frameimage, title="", fignum=1, showProj = False):

        if self.display_mode == 2 :
            plt.ion()

        self.fig = plt.figure(figsize=(10,8),num=fignum)
        self.fig.clf()

        axes = self.fig.add_subplot(111)

        self.fig.subplots_adjust(left=0.10,
                                 bottom=0.05,
                                 right=0.90,
                                 top=0.90,
                                 wspace=0.1,
                                 hspace=0.1)

        self.axesim = plt.imshow( frameimage,
                                  vmin=self.plot_vmin, vmax=self.plot_vmax )

        #self.colb = plt.colorbar(self.axesim,ax=axes,pad=0.01)#,fraction=0.10,shrink=0.90)
        divider = make_axes_locatable(axes)

        if showProj :
            #axes.set_aspect(1.0)
            axHistx = divider.append_axes("top", size="20%", pad=0.2,sharex=axes)
            axHisty = divider.append_axes("left", size="20%", pad=0.6,sharey=axes)

            # vertical and horizontal dimensions, axes, projections
            vdim,hdim = np.shape(frameimage)
            vbins = np.arange(0,vdim,1)
            hbins = np.arange(0,hdim,1)

            proj_vert = np.sum(frameimage,1)/hdim # sum along horizontal axis
            proj_horiz = np.sum(frameimage,0)/vdim # sum along vertical axis

            # these are the limits I want my histogram to use
            vmin = np.min(proj_vert) - 0.1 * np.min(proj_vert)
            vmax = np.max(proj_vert) + 0.1 * np.max(proj_vert)
            hmin = np.min(proj_horiz) - 0.1 * np.min(proj_horiz)
            hmax = np.max(proj_horiz) + 0.1 * np.max(proj_horiz) 

            print vmin,vmax,hmin,hmax
            #plt.clim( np.min(vmin,hmin), np.max(vmax,hmax))

            axHistx.plot(hbins,proj_horiz)
            axHisty.plot(proj_vert[::-1], vbins[::-1])
            #axHistx.hist(hbins, bins=hdim, histtype='step', weights=proj_horiz)
            #axHisty.hist(vbins, bins=vdim, histtype='step', weights=proj_vert,orientation='horizontal')

            axHistx.set_xlim(0,hdim)
            axHistx.set_ylim(hmin, hmax )
            #ticks = [ 10000 * np.around(hmin/10000) ,
            #          10000 * np.around((hmax-hmin)/20000),
            #          10000 * np.around(hmax/10000) ]
            ticks = [ np.around(hmin),
                      np.around((hmax-hmin)/2),
                      np.around(hmax) ]
            axHistx.set_yticks( ticks )

            #axHisty.set_ylim(0,vdim)
            #axHisty.set_xlim(vmin, vmax )
            axHisty.set_ylim(0,vdim)
            axHisty.set_xlim(vmax, vmin )
            #ticks = [ 10000 * np.around( vmin/10000) ,
            #          10000 * np.around((vmax-vmin)/20000),
            #          10000 * np.around(vmax/10000) ]
            ticks = [ np.around( vmax ) ,
                     np.around((vmax-vmin)/2),
                      np.around(vmin) ]
            axHisty.set_xticks( ticks )

        cax = divider.append_axes("right",size="5%", pad=0.05)
        self.colb = plt.colorbar(self.axesim,cax=cax)
        # colb is the colorbar object

        if self.plot_vmin is None: 
            self.orglims = self.axesim.get_clim()
            # min and max values in the axes are
            print "Original value limits: ", self.orglims
            self.plot_vmin, self.plot_vmax = self.orglims
                    
        # show the active region for thresholding
        if self.threshold and self.threshold.area is not None:
            xy = [self.threshold.area[0],self.threshold.area[2]]
            w = self.threshold.area[1] - self.threshold.area[0]
            h = self.threshold.area[3] - self.threshold.area[2]
            self.thr_rect = plt.Rectangle(xy,w,h, facecolor='none', edgecolor='red', picker=5)
            axes.add_patch(self.thr_rect)


        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('pick_event', self.onpick)

        if self.display_mode == 1 :
            print """
            To change the color scale, click on the color bar:
            - left-click sets the lower limit
            - right-click sets higher limit
            - middle-click resets to original
            """

        plt.suptitle(title)
        #axes.set_title(title)
        plt.draw()
        
    def onpick(self, event):
        print "Current   threshold = ", self.threshold.minvalue
        print "          active area [xmin xmax ymin ymax] = ", self.threshold.area
        print "To change threshold value, middle-click..." 
        print "To change active area, right-click..." 

        if event.mouseevent.button == 3 :
            print "Enter new coordinates to change this area:"
            xxyy_string = raw_input("xmin xmax ymin ymax = ")
            xxyy_list = xxyy_string.split(" ")
            
            if len( xxyy_list ) != 4 :
                print "Invalid entry, ignoring"
                return
        
            for i in range (4):
                self.threshold.area[i] = float( xxyy_list[i] )
            
            x = self.threshold.area[0]
            y = self.threshold.area[2]
            w = self.threshold.area[1] - self.threshold.area[0]
            h = self.threshold.area[3] - self.threshold.area[2]
            
            self.thr_rect.set_bounds(x,y,w,h)
            plt.draw()
            
        if event.mouseevent.button == 2 :
            text = raw_input("Enter new threshold value (current = %.2f) " % self.threshold.minvalue)
            if text == "" :
                print "Invalid entry, ignoring"
            else :
                self.threshold.minvalue = float(text)
                print "Threshold value has been changed to ", self.threshold.minvalue
            plt.draw()

            
    # define what to do if we click on the plot
    def onclick(self, event) :
    
        # can we open a dialogue box here?
        if not event.inaxes and event.button == 3 :
            print "can we open a menu here?"
            

        # change display mode
        if not event.inaxes and event.button == 2 :
            new_mode = None
            new_mode_str = raw_input("Switch display mode? Enter new mode: ")
            if new_mode_str != "":
                if new_mode_str == "NoDisplay"   or new_mode_str == "0" : new_mode = 0
                if new_mode_str == "Interactive" or new_mode_str == "1" : new_mode = 1
                if new_mode_str == "SlideShow"   or new_mode_str == "2" : new_mode = 2
            if self.display_mode != new_mode:
                print "Display mode has been changed from (%d) to (%d)" , (self.display_mode,new_mode)
                self.display_mode = new_mode
                

        # change color scale
        if self.colb is not None and event.inaxes == self.colb.ax :

            print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y
            print ' xdata=',event.xdata,' ydata=', event.ydata
        
            lims = self.axesim.get_clim()
        
            self.plot_vmin = lims[0]
            self.plot_vmax = lims[1]
            range = self.plot_vmax - self.plot_vmin
            value = self.plot_vmin + event.ydata * range
            #print self.plot_vmin, self.plot_vmax, range, value
        
            # left button
            if event.button is 1 :
                self.plot_vmin = value
                print "mininum changed:   ( %.2f , %.2f ) " % (self.plot_vmin, self.plot_vmax )
                
                # middle button
            elif event.button is 2 :
                self.plot_vmin, self.plot_vmax = self.orglims
                print "reset"
                    
                # right button
            elif event.button is 3 :
                self.plot_vmax = value
                print "maximum changed:   ( %.2f , %.2f ) " % (self.plot_vmin, self.plot_vmax )
                
            
        
            plt.clim(self.plot_vmin,self.plot_vmax)
            plt.draw() # redraw the current figure



