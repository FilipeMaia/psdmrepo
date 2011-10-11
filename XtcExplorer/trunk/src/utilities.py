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

    def getOptStringsDict(self, options_string) :
        """Return a dictionary of strings
        """
        if options_string is None:
            return {}
        
        mylist = self.getOptStrings(options_string)
        mydict = {}
        for entry in mylist:
            items = entry.split(":")
            if len(items) > 1 :
                mydict[items[0]] = items[1].strip('([])').split(',')
            else:
                mydict[items[0]] = None
                
        return mydict
           
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
        
    def show( self ):
        itsme = "\n%s \n\t name = %s" % (self.type, self.name)
        for item in dir(self):
            if item.find('__')>=0 : continue
            attr = getattr(self,item)
            if attr is not None:
                if type(attr)==str:
                    print item, "(str) = ", attr
                elif type(attr)==np.ndarray:
                    print item, ": ndarray of dimension(s) ", attr.shape
                else:
                    print item, " = ", type(attr)
                    
    def get_plottables_base(self):
        plottables = {}
        for item in dir(self):
            if item.find('__')>=0 : continue
            attr = getattr(self,item)
            if attr is not None:
                if type(attr)==np.ndarray:
                    plottables[item] = attr
        return plottables
                                
    def get_plottables(self):
        return self.get_plottables_base()
                                
                                
                                
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


class IpimbData( BaseData ):
    """Ipimb Data (from Intensity and Position monitoring boards)
    """
    def __init__( self, name, type="IpimbData" ):
        BaseData.__init__(self,name,type)
        self.fex_sum = None
        self.fex_channels = None
        self.fex_position = None
        self.raw_channels = None
        self.raw_voltages = None



class EncoderData( BaseData ):
    """Encoder data
    """
    def __init__( self, name, type="EncoderData" ):
        BaseData.__init__(self,name,type)
        self.values = None



class WaveformData( BaseData ):
    """Waveform data from Acqiris digitizers
    """
    def __init__( self, name, type="WaveformData" ):
        BaseData.__init__(self,name,type)
        self.wf_voltages = None
        self.wf2_voltages = None
        self.wf_time = None
        self.channels = None

    def get_plottables(self):
        plottables = self.get_plottables_base()
        for ch in self.channels: 
            plottables["volt_vs_time_ch%d"%ch] = (self.wf_time[ch],self.wf_voltages[ch])
        return plottables

class EpicsData( BaseData ):
    """Control and Monitoring PVs from EPICS
    """
    def __init__( self, name, type="EpicsData" ):
        BaseData.__init__(self,name,type)
        self.values = None
        self.shotnr = None
        self.status = None
        self.severity = None



class ScanData( BaseData ) :
    """Scan data
    """
    def __init__(self, name, type="ScanData"):
        BaseData.__init__(self,name,type)

        self.scanvec = None
        self.arheader = None
        self.scandata = None



class ImageData( BaseData ):
    """Image data
    """
    def __init__(self, name, type="ImageData"):
        BaseData.__init__(self,name,type)
        self.image = None      # the image
        self.average = None    # the average collected so far
        self.dark = None       # the dark that was subtracted
        self.roi = None        # list of coordinates defining ROI
        
        # The following are 1D array if unbinned, 2D if binned (bin array being the first dim)
        self.spectrum = None   # Array of image intensities (1D, or 2D if binned)
        self.projX = None      # Average image intensity projected onto horizontal axis
        self.projY = None      # Average image intensity projected onto vertical axis
        
        # The following are always 2D arrays, binned. bins vs. values
        self.projR = None      # Average image intensity projected onto radial axis (2D)
        self.projTheta = None  # Average image intensity projected onto polar angle axis (2D_
        
    def get_plottables(self):
        plottables = self.get_plottables_base()
        if self.roi is not None:
            try:
                c = self.roi
                print "image? ", self.image
                print "roi? ", self.image[c[0]:c[1],c[2]:c[3]]
                plottables["roi"] = self.image[c[0]:c[1],c[2]:c[3]]
            except:
                print "setting ROI failed, did you define the image? "
        return plottables
                
class CsPadData( BaseData ):
    """CsPad data
    """
    def __init__(self, name, type="CsPadData"):
        BaseData.__init__(self,name,type)
        self.image = None
        self.average = None
        self.dark = None



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
import numpy as np

from PyQt4 import QtCore

# uncomment these two if you want to run the pyana job in batch mode
# import matplotlib
# matplotlib.use('PDF')

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid

class Frame(object):
    """Frame (axes/subplot) manager

    In principle one should think that the 'figure' and 'axes' 
    containers would be sufficient to navigate the plot, right?
    Yet, I find no way to have access to everything, like colorbar.
    # So I make my own container... let me know if you know a better way
    """
    def __init__(self, name="", title=""):
        self.name = name
        self.title = title

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
        itsme +="\n title = %s " % self.title
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
        if self.projx is None: return
        if self.projy is None: return
        # does nothing right now...
        # but if needed, this is where to update axes of projection plots
        # if these need to change when image colorscale changes

    def set_ticks(self, limits = None ):
        
        if limits is None: 
            vmin, vmax = self.projy.get_xlim()
            hmin, hmax = self.projx.get_ylim()
            limits = (vmin,vmax,hmin,hmax)
        vmin,vmax,hmin,hmax = limits

        # -------------horizontal-------------------
        roundto = 1
        if hmax > 100 : roundto = 10
        if hmax > 1000 : roundto = 100
        if hmax > 10000 : roundto = 1000

        nticks = 3
        firsttick = roundto * np.around(hmin/roundto)
        lasttick = roundto * np.around(hmax/roundto)
        interval = roundto * np.around((hmax-hmin)/((nticks-1)*roundto))
        ticks = []
        for tck in range (nticks):
            ticks.append( firsttick + tck * interval )
                  
        self.projx.set_yticks( ticks )
        self.projx.set_ylim( np.min(ticks[0],hmin), np.max(ticks[-1],hmax) )
        
        # -------------vertical---------------
        roundto = 1
        if vmax > 100 : roundto = 10
        if vmax > 1000 : roundto = 100
        if vmax > 10000 : roundto = 1000

        nticks = 3
        firsttick = roundto * np.around(vmin/roundto)
        lasttick = roundto * np.around(vmax/roundto)
        interval = roundto * np.around((vmax-vmin)/((nticks-1)*roundto))
        ticks = []
        for tck in range (nticks):
            ticks.append( firsttick + tck * interval )

        self.projy.set_xticks( ticks )
        self.projy.set_xlim( np.max(ticks[-1],vmax), np.min(ticks[0],vmin) )


    

class Plotter(object):
    
    """Figure (canvas) manager
    """
    def __init__(self):
        self.fig = None
        self.fignum = None
        # a figure has one or more plots/frames
        
        self.frames = {} # dictionary / hash table to access the Frames

        self.display_mode = None
        # flag if interactively changed

        self.threshold = None
        self.vmin = None
        self.vmax = None

        self.first = True

        self.settings() # defaults

        # matplotlib backend is set to QtAgg, and this is needed to avoid
        # a bug in raw_input ("QCoreApplication::exec: The event loop is already running")
        QtCore.pyqtRemoveInputHook()

    def add_frame(self, name="", title=""):
        aframe = None

        if name == "":
            name = "frame%d",len(self.frames)+1
        
        if name in self.frames:
            aframe = self.frames[name]
        else :
            self.frames[name] = Frame(name)
            aframe = self.frames[name]

        aframe.title = title
        return aframe

        
    def settings(self
                 , width = 8 # width of a single plot
                 , height = 7 # height of a single plot
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
        for i in range (1,nplots+1):
            ax = self.fig.add_subplot(nrow,ncol,i)

            key = "fig%d_frame%d"%(fignum,i)
            if key in self.frames :
                self.frames[key].axes = ax
            else :
                aframe = self.add_frame(key)
                aframe.axes = ax
                #self.frames[key] = aframe 

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
        for aplot in self.frames.itemsvalues() :
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
            for key,aplot in self.frames.iteritems() :
                if aplot.colb and aplot.colb.ax == event.inaxes: 

                    #print "You clicked on colorbar of plot ", aplot.name
                    #print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y
                    #print ' xdata=',event.xdata,' ydata=', event.ydata

                    # color/value limits
                    clims = aplot.axesim.get_clim()        
                    aplot.vmin = clims[0]
                    aplot.vmax = clims[1]

                    range = aplot.vmax - aplot.vmin
                    value = aplot.vmin + event.ydata * range
                    #print "min,max,range,value = ",aplot.vmin,aplot.vmax,range,value
            
                    # left button
                    if event.button == 1 :
                        aplot.vmin = value
                        print "mininum of %s changed:   ( %.2f , %.2f ) " % (key, aplot.vmin, aplot.vmax )
                
                    # middle button
                    elif event.button == 2 :
                        aplot.vmin, aplot.vmax = aplot.orglims
                        print "reset %s to original: ( %.2f , %.2f ) " % (key, aplot.vmin, aplot.vmax )
                        
                    # right button
                    elif event.button == 3 :
                        aplot.vmax = value
                        print "maximum of %s changed:   ( %.2f , %.2f ) " % (key, aplot.vmin, aplot.vmax )
                
                    aplot.axesim.set_clim(aplot.vmin,aplot.vmax)
                    #aplot.update_axes()
                    plt.draw()




    def plot_image(self, image, fignum=1, title="", showProj = False, extent=None):
        """ plot_image
        utility function for when plotting a single image outside of pyana
        """
        self.create_figure(fignum,1)
        self.fig.suptitle(title)
        self.drawframe(image, showProj=showProj, extent=extent)

        plt.draw()
        plt.show()

    def plot_several(self, fignum, list_of_arrays, title="" ):
        """ Draw several frames in one canvas
        
        @fignum                  figure number, i.e. fig = plt.figure(num=fignum)
        @list_of_arrays          a list of tuples (title, array)
        @return                  new display_mode if any (else return None)
        """
        #if self.fig is None: 
        self.create_figure(fignum, nplots=len(list_of_arrays))
        self.fig.suptitle(title)
            
        pos = 0
        for tuple in list_of_arrays :
            pos += 1
            ad = tuple[0]
            im = tuple[1]
            xt = None
            if len(tuple)==3 : xt = tuple[2]

            if type(im)==np.ndarray:
                if len( im.shape ) > 1:
                    self.drawframe(im,title=ad,fignum=fignum,position=pos)
                else :
                    plt.plot(im)
            elif type(im)==tuple:
                print "tuple"
                pass
                
        plt.draw()
        return self.display_mode

    def draw_figurelist(self, fignum, event_display_images, title="",showProj=False,extent=None ) :
        """ Draw several frames in one canvas
        
        @fignum                  figure number, i.e. fig = plt.figure(num=fignum)
        @event_display_images    a list of tuples (title,image)
        @return                  new display_mode if any (else return None)
        """
        #if self.fig is None: 
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
        key = "fig%d_frame%d"%(fignum,position)
        aplot = self.frames[key]
        aplot.image = frameimage

        if ( aplot.vmin is None) and (self.vmin is not None ):
            aplot.vmin = self.vmin
        if ( aplot.vmax is None) and (self.vmax is not None ):
            aplot.vmax = self.vmax
        
        # get axes
        aplot.axes = self.fig.axes[position-1]
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

            # --- sum or average along each axis, 
            maskedimage = np.ma.masked_array(frameimage, mask=(frameimage==0) )
            proj_vert = np.ma.average(maskedimage,1) # for each row, average of elements
            proj_horiz = np.ma.average(maskedimage,0) # for each column, average of elements
            
            x1,x2,y1,y2 = aplot.axesim.get_extent()
            start_x = x1
            start_y = y1

            # vertical and horizontal dimensions, axes, projections
            vdim,hdim = aplot.axesim.get_size()        
            hbins = np.arange(start_x, start_x+hdim, 1)
            vbins = np.arange(start_y, start_y+vdim, 1)

            aplot.projx.plot(hbins,proj_horiz)
            aplot.projy.plot(proj_vert, vbins)
            aplot.projx.get_xaxis().set_visible(False)
        
            aplot.projx.set_xlim( start_x, start_x+hdim)
            aplot.projy.set_ylim( start_y+vdim, start_y)

            aplot.set_ticks()
            #aplot.update_axes()
            

        cax = divider.append_axes("right",size="5%", pad=0.05)
        aplot.colb = plt.colorbar(aplot.axesim,cax=cax)
        # colb is the colorbar object

        aplot.orglims = aplot.axesim.get_clim()
        if aplot.vmin is not None:
            aplot.orglims = ( aplot.vmin, aplot.orglims[1] )
        if aplot.vmax is not None:
            aplot.orglims = ( aplot.orglims[0], aplot.vmax )

        aplot.vmin, aplot.vmax = aplot.orglims


        
        # show the active region for thresholding
        if aplot.threshold and aplot.threshold.area is not None:
            xy = [aplot.threshold.area[0],aplot.threshold.area[2]]
            w = aplot.threshold.area[1] - aplot.threshold.area[0]
            h = aplot.threshold.area[3] - aplot.threshold.area[2]
            aplot.thr_rect = plt.Rectangle(xy,w,h, facecolor='none', edgecolor='red', picker=10)
            aplot.axes.add_patch(aplot.thr_rect)
            print "Plotting the red rectangle in area ", aplot.threshold.area

        aplot.axes.set_title(title)
        
        
