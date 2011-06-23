#-----------------------------------------
# PyanaOptions
#-----------------------------------------

class PyanaOptions( object ):
    """Class PyanaOptions

    Collection of functions to convert the string options of pyana
    into values, or lists of values, of the expected type
    """
    def __init__( self ):
        pass

    def getOptString(self, options_string) :
        """Return the string, strip of any whitespaces
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
        """Return a list of strings 
        """
        if options_string is None:
            return None

        # strip off any leading or trailing whitespaces
        options_string = options_string.strip()

        # make sure there are no newline characters here
        options_string = options_string.split("\n")
        options_string = " ".join(options_string)

        # make a list
        options = options_string.split()

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
    """Base class for container objects storing event data
    in memory (as numpy arrays mainly). Useful for passing
    the data to e.g. ipython for further investigation
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
    """Beam-Line Data 
    """
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
        self.raw_channels_volt = None
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
        if self.raw_channels_volt is not None :
            itsme+="\n\t raw_channels_volt = array of shape %s"%str(np.shape(self.raw_channels_volt))
        if self.fex_position is not None :
            itsme+="\n\t fex_position = array of shape %s"%str(np.shape(self.fex_position))
        print itsme

class IpimbData( BaseData ):
    """Ipimb Data (from Intensity and Position monitoring boards)
    """
    def __init__( self, name, type="IpimbData" ):
        BaseData.__init__(self,name,type)
        self.fex_sum = None
        self.fex_channels = None
        self.fex_position = None
        self.raw_channels = None
        self.raw_channels_volt = None

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
        if self.raw_channels_volt is not None :
            itsme+="\n\t raw_channels_volt = array of shape %s"%str(np.shape(self.raw_channels_volt))
        print itsme


class EncoderData( BaseData ):
    """Encoder data
    """
    def __init__( self, name, type="EncoderData" ):
        BaseData.__init__(self,name,type)
        self.values = None

    def show( self ):
        """Printable description 
        """
        itsme = "\n%s: \n\t name = %s" % (self.type, self.name)
        if self.values is not None :
            itsme+="\n\t values = array of shape %s"%str(np.shape(self.values))
        print itsme


class WaveformData( BaseData ):
    """Waveform data from Acqiris digitizers
    """
    def __init__( self, name, type="WaveformData" ):
        BaseData.__init__(self,name,type)
        self.wf_voltage = None
        self.wf_time = None

    def show( self ):
        """Printable description 
        """
        itsme = "\n%s: \n\t name = %s" % (self.type, self.name)
        if self.wf_voltage is not None :
            itsme+="\n\t wf_voltage = array of shape %s"%str(np.shape(self.wf_voltage))
        if self.wf_time is not None :
            itsme+="\n\t wf_time = array of shape %s"%str(np.shape(self.wf_time))
        print itsme


class EpicsData( BaseData ):
    """Control and Monitoring PVs from EPICS
    """
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
    """Scan data
    """
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
    """Image data
    """
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
    """CsPad data
    """
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
    """Class Threshold

    To keep track of threshold settings (value and area of interest)
    """
    def __init__( self,
                  area = None,
                  value = None,
                  ) :

        self.area = area        
        self.value = value



#-------------------------------------------------------
# Plotter
#-------------------------------------------------------
import time
import numpy as np
import matplotlib.pyplot as plt
from PyQt4 import QtCore

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid

class Frame(object):
    """Frame (axes/subplot) manager

    In principle one should think that the 'figure' and 'axes' 
    containers would be sufficient to navigate the plot, right?
    Yet, I find no way to have access to everything, like colorbar.
    # So I make my own container... let me know if you know a better way
    """
    def __init__(self, name=""):
        self.name = name

        self.axes = None       # the patch containing the image
        self.axesim = None     # the image AxesImage
        self.image = None      # the image (numpy array)
        self.colb = None       # the colorbar
        self.projx = None      # projection onto the horizontal axis
        self.projy = None      # projection onto the vertical axis

        # threshold associated with this plot (image)
        self.threshold = None

        self.first = True

        # display-limits for this plot (image)
        self.vmin = None
        self.vmax = None
        self.orglims = None

    def show(self):
        itsme = "%s"%self
        itsme +="\n name = %s " % self.name        
        itsme +="\n axes = %s " % self.axes
        itsme +="\n axesim = %s " % self.axesim
        itsme +="\n threshold = %s " % self.threshold
        itsme +="\n vmin = %s " % self.vmin
        itsme +="\n vmax = %s " % self.vmax
        itsme +="\n orglims = %s " % str(self.orglims)
        itsme +="\n projx = %s " % self.projx
        itsme +="\n projy = %s " % self.projy
        print itsme

    def update_axes(self):
        # max along each axis
        proj_vert = self.image.max(axis=1) # for each row, maximum bin value
        proj_horiz = self.image.max(axis=0) # for each column, maximum bin value
        
        # ---------------------------------------------------------
        # these are the limits I want my histogram to use
        vmin = np.min(proj_vert) - 0.1 * np.min(proj_vert)
        #vmin = 0
        vmax = np.max(proj_vert) + 0.1 * np.max(proj_vert)
        hmin = np.min(proj_horiz) - 0.2 * np.min(proj_horiz)
        #hmin = 0.0
        hmax = np.max(proj_horiz) + 0.2 * np.max(proj_horiz) 
        
        # unless vmin and vmax has been set for the image
        if self.vmax is not None: 
            vmax = self.vmax + 0.1 * self.vmax
            hmax = self.vmax + 0.1 * self.vmax
        if self.vmin is not None: 
            vmin = self.vmin - 0.1 * self.vmin
            hmin = self.vmin - 0.1 * self.vmin
            
        # -------------horizontal-------------------
        roundto = 1
        if hmax > 100 : roundto = 10
        if hmax > 1000 : roundto = 100
        if hmax > 10000 : roundto = 1000
        ticks = [ roundto * np.around(hmin/roundto) ,
                  roundto * np.around((hmax-hmin)/(2*roundto)),
                  roundto * np.around(hmax/roundto) ]
        self.projx.set_yticks( ticks )
        
        self.projx.set_ylim( np.min(ticks[0],hmin), np.max(ticks[-1],hmax) )
        
        
        # -------------vertical---------------
        roundto = 1
        if vmax > 100 : roundto = 10
        if vmax > 1000 : roundto = 100
        if vmax > 10000 : roundto = 1000
        ticks = [ roundto * np.around( vmin/roundto) ,
                  roundto * np.around((vmax-vmin)/(2*roundto)),
                  roundto * np.around(vmax/roundto) ]
        self.projy.set_xticks( ticks )
        
        self.projy.set_xlim( np.max(ticks[-1],vmax), np.min(ticks[0],vmin) )

    
class Plotter(object):
    """Figure (canvas) manager
    """
    def __init__(self):
        self.fig = None
        self.fignum = None
        # a figure has one or more plots/frames
        self.frames = None # list of class Frame in this figure
        self.frame = {} # dictionary / hash table to access the Frames (only if named)

        self.display_mode = None
        # flag if interactively changed

        self.first = True

        self.settings() # defaults

        # matplotlib backend is set to QtAgg, and this is needed to avoid
        # a bug in raw_input ("QCoreApplication::exec: The event loop is already running")
        QtCore.pyqtRemoveInputHook()

    def add_frame(self, frame_name=""):
        if self.frames is None:
            self.frames = []
        aframe = Frame(frame_name)
        self.frames.append(aframe)
        
        if frame_name != "" :
            self.frame[frame_name] = aframe
        
    def settings(self
                 , width = 5 # width of a single plot
                 , height = 4 # height of a single plot
                 , nplots=1  # total number of plots in the figure
                 , maxcol=3  # maximum number of columns
                 ):
        self.w = width
        self.h = height
        self.nplots = nplots
        self.maxcol = maxcol
        

    def create_figure(self, fignum, nplots=1):
        """ Make the matplotlib figure.
        This clears and rebuilds the canvas, although
        if the figure was made earlier, some of it is recycled
        """
        ncol = 1
        nrow = 1
        if nplots == 4:
            ncol = 2
            nrow = 2
        elif nplots > 1 :
            ncol = self.maxcol
            if nplots<self.maxcol : ncol = nplots
            nrow = int( nplots/ncol )
            if (nplots%ncol) > 0 : nrow+=1
        
        #print "Figuresize: ", self.w*ncol,self.h*nrow
        #print "Figure conf: %d rows x %d cols" % ( nrow, ncol)
        
        # --- sanity check ---
        max =  ncol * nrow
        if nplots > max :
            print "utitilities.py: Something wrong with the subplot configuration"
            print "                Not enough space for %d plots in %d x %d"%(nplots,ncol,nrow)


        self.fig = plt.figure(fignum,(self.w*ncol,self.h*nrow))
        self.fignum = fignum
        self.fig.clf()
        self.fig.set_size_inches(self.w*ncol,self.h*nrow)

        self.fig.subplots_adjust(left=0.05,   right=0.95,
                                 bottom=0.05, top=0.90,
                                 wspace=0.2,  hspace=0.2 )
        
        # add subplots and frames
        if self.frames is None:
            self.frames = []

        for i in range (1,nplots+1):
            ax = self.fig.add_subplot(nrow,ncol,i)

            if len(self.frames) >=  i:
                self.frames[i-1].axes = ax
            else :
                aframe = Frame()
                aframe.axes = ax
                self.frames.append(aframe)

            #print "Subplot ", i, " created ", nrow, ncol, i, self.frames[i-1].name
                    
        self.connect()

    def close_figure(self):
        #print plt.get_fignums()
        plt.close(self.fignum)
        
    def connect(self,plot=None):
        if plot is None: 
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('pick_event', self.onpick)

        else :
            self.fig.canvas.mpl_connect('button_press_event', plot.onclick)
            self.fig.canvas.mpl_connect('pick_event', plot.onpick)
            

    def onpick(self, event):

        #print "The following artist object was picked: ", event.artist

        # in which Frame?
        for aplot in self.frames :
            if aplot.axes == event.artist.axes : 

                print "Current   threshold = ", aplot.threshold.value
                print "          active area [xmin xmax ymin ymax] = ", aplot.threshold.area
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
                        aplot.threshold.area[i] = float( xxyy_list[i] )
            
                    x = aplot.threshold.area[0]
                    y = aplot.threshold.area[2]
                    w = aplot.threshold.area[1] - aplot.threshold.area[0]
                    h = aplot.threshold.area[3] - aplot.threshold.area[2]
                    
                    aplot.thr_rect.set_bounds(x,y,w,h)
                    plt.draw()
            
                if event.mouseevent.button == 2 :
                    text = raw_input("Enter new threshold value (current = %.2f) " % aplot.threshold.value)
                    if text == "" :
                        print "Invalid entry, ignoring"
                    else :
                        aplot.threshold.value = float(text)
                        print "Threshold value has been changed to ", aplot.threshold.value
                        plt.draw()

            
    # define what to do if we click on the plot
    def onclick(self, event) :

        if self.first : 
            print """
            To change the color scale, click on the color bar:
            - left-click sets the lower limit
            - right-click sets higher limit
            - middle-click resets to original
            """
            self.first = False
        

        # -------------- clicks outside axes ----------------------
        # can we open a dialogue box here?
        if not event.inaxes and event.button == 3 :
            print "can we open a menu here?"
            
        # change display mode
        if not event.inaxes and event.button == 2 :
            new_mode = None
            new_mode_str = raw_input("Plotter: switch display mode? Enter new mode: ")
            if new_mode_str != "":
                
                if new_mode_str == "NoDisplay"   :
                    new_mode = 0
                if new_mode_str == "0"           :
                    new_mode = 0
                    new_mode_str = "NoDisplay"

                if new_mode_str == "Interactive" :
                    new_mode = 1
                if new_mode_str == "1"           :
                    new_mode = 1
                    new_mode_str = "Interactive" 

                if new_mode_str == "SlideShow"   :
                    new_mode = 2
                if new_mode_str == "2"           :
                    new_mode = 2
                    new_mode_str = "SlideShow"   

                print "Plotter display mode has been changed from %s to %d (%s)" % \
                      (self.display_mode,new_mode,new_mode_str)
                self.display_mode = new_mode 

                if new_mode == 2 :
                    # if we switch from Interactive to SlideShow mode
                    # the figure needs to be properly closed 
                    # and recreated after setting ion (mpl interactive mode)
                    # if not, the figure remains hidden after you close the GUI
                    #self.close_figure()
                    plt.close('all')
                    plt.ion()


        # -------------- clicks inside axes ----------------------
        if event.inaxes :

            # find out which axes was clicked...

            # ... colorbar?
            for aplot in self.frames :
                if aplot.colb and aplot.colb.ax == event.inaxes: 
                    
                    print "You clicked on colorbar of plot ", aplot.name

                    print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y
                    print ' xdata=',event.xdata,' ydata=', event.ydata
        
                    lims = aplot.axesim.get_clim()
        
                    aplot.vmin = lims[0]
                    aplot.vmax = lims[1]
                    range = aplot.vmax - aplot.vmin
                    value = aplot.vmin + event.ydata * range
                    print "min,max,range,value = ",aplot.vmin,aplot.vmax,range,value
            
                    # left button
                    if event.button == 1 :
                        aplot.vmin = value
                        print "mininum changed:   ( %.2f , %.2f ) " % (aplot.vmin, aplot.vmax )
                
                    # middle button
                    elif event.button == 2 :
                        aplot.vmin, aplot.vmax = aplot.orglims
                        print "reset"
                        
                    # right button
                    elif event.button == 3 :
                        aplot.vmax = value
                        print "maximum changed:   ( %.2f , %.2f ) " % (aplot.vmin, aplot.vmax )
                
                    
                    aplot.axesim.set_clim(aplot.vmin,aplot.vmax)
                    aplot.update_axes()
                    plt.draw()




    def draw_figurelist(self, fignum, event_display_images, title="",showProj=False,extent=None ) :
        """ Draw several frames in one canvas
        
        @fignum                  figure number, i.e. fig = plt.figure(num=fignum)
        @event_display_images    a list of tuples (title,image)
        @return                  new display_mode if any (else return None)
        """

        self.create_figure(fignum, nplots=len(event_display_images))
        self.fig.suptitle(title)

        pos = 0
        for tuple in event_display_images :
            pos += 1
            ad = tuple[0]
            im = tuple[1]
            xt = None
            if len(tuple)==3 : xt = tuple[2]
            
            self.drawframe(im,title=ad,fignum=fignum,position=pos,showProj=showProj,extent=xt)
            
        plt.draw()
        return self.display_mode

    def draw_figure( self, frameimage, title="", fignum=1,position=1, showProj = False,extent=None):
        """ Draw a single frame in one canvas
        """
        self.create_figure(fignum)
        self.fig.suptitle(title)
        self.drawframe(frameimage,title,fignum,position,showProj,extent)

        plt.draw()
        return self.display_mode


    def drawframe( self, frameimage, title="", fignum=1,position=1, showProj = False,extent=None):
        """ Draw a single frame
        """
        index= position-1
        aplot = self.frames[index]
        aplot.image = frameimage
        
        # get axes
        aplot.axes = self.fig.axes[index]
        aplot.axes.set_title( title )

        if aplot.name == "" and title != "" :
            aplot.name = title

        # AxesImage
        aplot.axesim = aplot.axes.imshow( frameimage,
                                          origin='lower',
                                          extent=extent,
                                          vmin=aplot.vmin,
                                          vmax=aplot.vmax )
        

        divider = make_axes_locatable(aplot.axes)

        if showProj :
            aplot.projx = divider.append_axes("top", size="20%", pad=0.03,sharex=aplot.axes)
            aplot.projy = divider.append_axes("left", size="20%", pad=0.03,sharey=aplot.axes)
            aplot.projx.set_title( aplot.axes.get_title() )

            start_x = 0
            start_y = 0
            if extent is not None:
                start_x = extent[0]
                start_y = extent[2]

            # vertical and horizontal dimensions, axes, projections
            vdim,hdim = np.shape(frameimage)
            hbins = np.arange(start_x, start_x+hdim, 1)
            vbins = np.arange(start_y, start_y+vdim, 1)

            # sum or average along each axis, 
            #proj_vert = np.sum(frameimage,1)/hdim # for each row, sum of elements
            #proj_horiz = np.sum(frameimage,0)/vdim # for each column, sum of elements

            # max along each axis
            proj_vert = frameimage.max(axis=1) # for each row, maximum bin value
            proj_horiz = frameimage.max(axis=0) # for each column, maximum bin value

            aplot.projx.plot(hbins,proj_horiz)
            aplot.projy.plot(proj_vert[::-1], vbins[::-1])
            #aplot.projx.hist(hbins, bins=hdim, histtype='step', weights=proj_horiz)
            #aplot.projy.hist(vbins, bins=vdim, histtype='step', weights=proj_vert,orientation='horizontal')
            aplot.projx.get_xaxis().set_visible(False)

            aplot.projx.set_xlim( start_x, start_x+hdim)
            aplot.projy.set_ylim( start_y, start_y+vdim)

            aplot.update_axes()
            

        cax = divider.append_axes("right",size="5%", pad=0.05)
        aplot.colb = plt.colorbar(aplot.axesim,cax=cax)
        # colb is the colorbar object

        if aplot.vmin is None: 
            aplot.orglims = aplot.axesim.get_clim()
            # min and max values in the axes are
            print "%s original value limits: %s" % (aplot.name,aplot.orglims)
            aplot.vmin, aplot.vmax = aplot.orglims
                    
        # show the active region for thresholding
        print "threshold = ", aplot.threshold
        print "area     =  ", aplot.threshold.area
        if aplot.threshold and aplot.threshold.area is not None:
            xy = [aplot.threshold.area[0],aplot.threshold.area[2]]
            w = aplot.threshold.area[1] - aplot.threshold.area[0]
            h = aplot.threshold.area[3] - aplot.threshold.area[2]
            aplot.thr_rect = plt.Rectangle(xy,w,h, facecolor='none', edgecolor='red', picker=10)
            aplot.axes.add_patch(aplot.thr_rect)
            print "Plotting the red rectangle in area ", aplot.threshold.area

        aplot.axes.set_title(title)
        
        
