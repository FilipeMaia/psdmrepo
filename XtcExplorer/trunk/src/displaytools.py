import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, SpanSelector
from matplotlib.patches import Circle, Rectangle, Polygon
class DataDisplay(object):

    def __init__(self, mode=0):
        self.event_number = 0
        self.image_disp = None
        self.wf_disp = None
        self.display_mode = mode
        self.figures = [] # keep a list of reference to all figures

    def show_wf(self, datalist):
        if self.wf_disp is None:
            self.wf_disp = Plotter()
            self.wf_disp.display_mode = self.display_mode
            self.wf_disp.settings(7,7) # set default frame size
            self.wf_disp.threshold = None
            
        i = 0
        for wf_data in datalist:
            """Single Waveform"""
            if wf_data.wf is not None:
                name = "wf(%s)"%wf_data.name
                title = "Single event (#%d), %s"%(self.event_number,wf_data.name)
                nbins = wf_data.wf.size
                contents = (wf_data.ts[0:nbins],wf_data.wf)
                self.wf_disp.add_frame(name,title,contents, aspect='auto')
            
            """Average Waveform"""
            if wf_data.average is not None:
                name = "wfavg(%s)"%wf_data.name
                title = "Average of %d events, %s"%(wf_data.counter,wf_data.name)
                nbins = wf_data.average.size
                contents = (wf_data.ts[0:nbins],wf_data.average)
                self.wf_disp.add_frame(name,title,contents, aspect='auto')
                
            """Stack"""
            if wf_data.stack is not None:
                name = "wfstack(%s)"%wf_data.name
                title = "Stack, %s"%wf_data.name
                wf_image = wf_data.stack
                if type(wf_image).__name__=='list' :
                    wf_image = np.float_(wf_image)
                contents = (wf_image,) # a tuple
                self.wf_disp.add_frame(name,title,contents,aspect='equal')
                self.wf_disp.frames[name].axis_values = wf_data.ts

                
        newmode = self.wf_disp.plot_all_frames(ordered=True)

        # This title is common to all the plots
        plt.suptitle("Event %d" % (self.event_number))
        return newmode
            



    def show_bld(self, datalist):
        for bld in datalist:
            if bld.name == "EBeam":
                self.plot_ebeam(bld)
            if bld.name == "GasDetector":
                self.plot_gasdet(bld)
            if bld.name == "PhaseCavity":
                self.plot_phasecavity(bld)
            plt.draw()
        return


    def plot_ebeam(self,ebeam):

        print "Making plot of array of length %d"%ebeam.shots.size

        # make figure                
        fig = plt.figure(100, figsize=(8,8) )
        fig.clf()
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.suptitle("BldInfo:EBeam data shot#%d"%self.event_number)
        
        ax1 = fig.add_subplot(221)
        if (np.size(ebeam.shots) != np.size(ebeam.energies) ):
            print "event    ", self.n_shots
            print "axis     ", np.size(ebeam.shots), np.shape(ebeam.shots)
            print "energies ", np.size(ebeam.energies), np.shape(ebeam.energies)

        plt.plot(ebeam.shots,ebeam.energies)
        plt.title("Beam Energy")
        plt.xlabel('Datagram record',horizontalalignment='left') # the other right
        plt.ylabel('Beam Energy',horizontalalignment='right')
        
        ax2 = fig.add_subplot(222)
        plt.scatter(ebeam.positions[:,0],ebeam.angles[:,0])
        plt.title('Beam X')
        plt.xlabel('position X',horizontalalignment='left')
        plt.ylabel('angle X',horizontalalignment='left')
            
        ax3 = fig.add_subplot(223)
        plt.scatter(ebeam.positions[:,1],ebeam.angles[:,1])
        plt.title("Beam Y")
        plt.xlabel('position Y',horizontalalignment='left')
        plt.ylabel('angle Y',horizontalalignment='left')
            
        ax4 = fig.add_subplot(224)
        n, bins, patches = plt.hist(ebeam.charge, 100, normed=1, histtype='stepfilled')
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
        plt.title('Beam Charge')
        plt.xlabel('Beam Charge',horizontalalignment='left') # the other right

        return

    def gasdet_energies(self, gasdet, i):
        try:
            energies = gasdet.energies[:,i]
        except:
            if i == 0:
                energies = [ x.f_11_ENRC() for x in gasdet.energies ]
            elif i == 1:
                energies = [ x.f_12_ENRC() for x in gasdet.energies ]
            elif i == 2:
                energies = [ x.f_21_ENRC() for x in gasdet.energies ]
            elif i == 3:
                energies = [ x.f_22_ENRC() for x in gasdet.energies ]
        return energies

    def plot_gasdet(self,gasdet):
        print "Making plot of gasdet of length %d"%gasdet.shots.size
                
        # make figure
        fig = plt.figure(101, figsize=(8,8) )
        fig.clf()
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.suptitle("BldInfo:GasDetector data shot#%d"%self.event_number)
        
        # numpy array (4d)
        ax1 = fig.add_subplot(221)
        n, bins, patches = plt.hist(self.gasdet_energies(gasdet, 0), 60, histtype='stepfilled')
        plt.setp(patches,'facecolor', 'r', 'alpha', 0.75)
        plt.title('Energy 11')
        plt.xlabel('Energy E[0]',horizontalalignment='left')
        
        ax2 = fig.add_subplot(222)
        n, bins, patches = plt.hist(self.gasdet_energies(gasdet, 1), 60, histtype='stepfilled')
        #n, bins, patches = plt.hist(gasdet.energies[:,1], 60,histtype='stepfilled')
        plt.setp(patches,'facecolor', 'g', 'alpha', 0.75)
        plt.title('Energy 12')
        plt.xlabel('Energy E[1]',horizontalalignment='left')
        
        ax3 = fig.add_subplot(223)
        n, bins, patches = plt.hist(self.gasdet_energies(gasdet, 2), 60, histtype='stepfilled')
        #n, bins, patches = plt.hist(gasdet.energies[:,2], 60,histtype='stepfilled')
        plt.setp(patches,'facecolor', 'b', 'alpha', 0.75)
        plt.title('Energy 21')
        plt.xlabel('Energy E[2]',horizontalalignment='left')
        
        ax4 = fig.add_subplot(224)
        n, bins, patches = plt.hist(self.gasdet_energies(gasdet, 3), 60, histtype='stepfilled')
        #n, bins, patches = plt.hist(gasdet.energies[:,3], 60,histtype='stepfilled')
        plt.setp(patches,'facecolor', 'm', 'alpha', 0.75)
        plt.title('Energy 22')
        plt.xlabel('Energy E[3]',horizontalalignment='left')
        return
            
    def plot_phasecavity(self,pc):
        print "Making plot of phasecavity, based on %d shots"%pc.shots.size

        # make figure
        fig = plt.figure(102, figsize=(12,8) )
        fig.clf()
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.suptitle("BldInfo:PhaseCavity data shot#%d"%self.event_number)
        
        ax1 = fig.add_subplot(231)
        n, bins, patches = plt.hist(pc.time[0], 60,histtype='stepfilled')
        plt.setp(patches,'facecolor', 'r', 'alpha', 0.75)
        plt.title('Time PC1')
        plt.xlabel('Time PC1',horizontalalignment='left')
        unit = bins[1] - bins[0]
        x1min, x1max = (bins[0]-unit), (bins[-1]+unit)
        plt.xlim(x1min,x1max)
        
        ax2 = fig.add_subplot(232)
        n, bins, patches = plt.hist(pc.time[1], 60,histtype='stepfilled')
        plt.setp(patches,'facecolor', 'r', 'alpha', 0.75)
        plt.title('Time PC2')
        plt.xlabel('Time PC2',horizontalalignment='left')
        unit = bins[1] - bins[0]
        x2min, x2max = (bins[0]-unit), (bins[-1]+unit)
        plt.xlim(x2min,x2max)
        
        ax3 = fig.add_subplot(233)
        plt.scatter(pc.time[0],pc.time[1])
        plt.title("Time PC1 vs. Time PC2")
        plt.xlabel('Time PC1',horizontalalignment='left')
        plt.ylabel('Time PC2',horizontalalignment='left')
        plt.xlim(x1min,x1max)
        plt.ylim(x2min,x2max)
        
        ax4 = fig.add_subplot(234)
        n, bins, patches = plt.hist(pc.charge[0], 60,histtype='stepfilled')
        plt.setp(patches,'facecolor', 'b', 'alpha', 0.75)
        plt.title('Charge PC1')
        plt.xlabel('Charge PC1',horizontalalignment='left')
        
        ax5 = fig.add_subplot(235)
        n, bins, patches = plt.hist(pc.charge[1], 60,histtype='stepfilled')
        plt.setp(patches,'facecolor', 'b', 'alpha', 0.75)
        plt.title('Charge PC2')
        plt.xlabel('Charge PC2',horizontalalignment='left')
        
        ax6 = fig.add_subplot(236)
        plt.scatter(pc.charge[0],pc.charge[1])
        plt.title("Charge PC1 vs. Charge PC2")
        plt.xlabel('Charge PC1',horizontalalignment='left')
        plt.ylabel('Charge PC2',horizontalalignment='left')
        

        
    def show_ipimb(self, datalist ):

        if len(datalist)==1:
            self.show_ipimb_single(datalist)
        else:
            self.show_ipimb_multiple(datalist)
        plt.draw()
            

    def show_ipimb_multiple(self,datalist):

        fig = plt.figure(200, figsize=(len(datalist)*5,10))
        fig.subplots_adjust(wspace=0.3,left=0.10,right=0.96,bottom=0.08)
        plt.clf()

        ndev = len(datalist)
        ax = plt.subplot(2,ndev,1) # two rows, n columns) 
        
        arrays = []
        titles = []
        gain = []
        for ipimb in datalist:
            titles.append(ipimb.name)
            gain.append("%s"%(ipimb.gain_settings))
            arrays.append(ipimb.fex_sum) 
            ax.plot(ipimb.fex_sum,'.',label="%s"%(ipimb.name))
        ax.set_xlabel("event count")
        ax.set_ylabel("Sum [V]")
        ax.legend()


        for j in range(0,ndev):
            tpos = j+1
            if tpos > 1 :
                #print "Plotting ipimb scatter of %d and %d in position %d"% (j-1,j, tpos)
                ax = plt.subplot(2,ndev,tpos)
                ax.scatter(arrays[j-1],arrays[j])
                ax.set_xlabel("%s sum [V]"%titles[j-1])
                ax.set_ylabel("%s sum [V]"%titles[j])

            bpos = j+1+ndev
            #print "Plotting ipimb %d in position %d"%( j,bpos)
            ax = plt.subplot(2,ndev,bpos)
            n,bins,patches = ax.hist(arrays[j], 100, histtype='stepfilled', color='b', label='Fex Sum')
            ax.set_xlabel("%s sum [V]"%titles[j])
            if gain[j]!="None":
                plt.text(0.6, 0.9,"Feedback capacitor settings:",
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform = ax.transAxes)
                plt.text(0.6, 0.8, gain[j],
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform = ax.transAxes)

            plt.suptitle(", ".join(titles)+"  shot#%d"%self.event_number)


    def show_ipimb_single(self,datalist):

        fig = plt.figure(200, figsize=(14,10))
        plt.clf()

        for ipimb in datalist:
            #ipimb.show()

            plt.suptitle("%s   %s   shot#%d"%(ipimb.name,ipimb.gain_settings,self.event_number))

            gs = GridSpec(4,4)
            gs.update(wspace=0.4,hspace=0.6,left=0.10,right=0.95)

            gains = ["","","",""]
            if ipimb.gain_settings is not None:
                gains = ["(%s)"%s for s in ipimb.gain_settings]
            ax1 = plt.subplot(gs[:2,:2])
            for ch in range (4):
                ax1.plot(ipimb.fex_channels[:,ch],label="Ch %d %s"%(ch,gains[ch]))
            ax1.set_xlabel("event count")
            ax1.set_ylabel("Intensity [V]")
            ax1.legend()
            ax1.set_title("Diode voltage per channel vs. event number")


            ax2 = plt.subplot( gs[:2,2:3] )
            ax2.plot(ipimb.fex_channels[:,0],ipimb.fex_channels[:,1],'o')
            ax2.set_xlabel("Ch0 [V]")
            ax2.set_ylabel("Ch1 [V]")
            ax2.set_title("Ch1 vs. Ch0")
            
            ax2.xaxis.major.formatter.set_powerlimits((0,0)) 
            ax2.yaxis.major.formatter.set_powerlimits((0,0)) 

            ax3 = plt.subplot( gs[:2,3:4]) #, sharey=ax2)
            ax3.plot(ipimb.fex_channels[:,2],ipimb.fex_channels[:,3],'o')
            ax3.set_xlabel("Ch2 [V]")
            ax3.set_ylabel("Ch3 [V]")
            ax3.set_title("Ch3 vs. Ch2")
            ax3.xaxis.major.formatter.set_powerlimits((0,0)) 
            ax3.yaxis.major.formatter.set_powerlimits((0,0)) 

            ax4 = plt.subplot( gs[2:,:2])
            ax4.plot(ipimb.fex_sum,label="FEX Sum")
            ax4.set_xlabel("event count")
            ax4.set_ylabel("Intensity [V]")
            ax4.legend()
            ax4.set_title("Diode voltage Sum vs. event number")
            
            ax5 = plt.subplot( gs[2:,2:] )
            n,bins,patches = ax5.hist(ipimb.fex_sum, 100, histtype='stepfilled', color='b', label='Fex Sum')
            ax5.set_xlabel("Sum of channels [V]")
            ax5.set_ylabel("#Events")
            ax5.set_title("Diode voltage sum, histogram")
    

    def show_image(self, datalist ):

        if self.image_disp is None: 
            self.image_disp = Plotter()
            self.image_disp.display_mode = self.display_mode
            self.image_disp.settings(7,7)
            
        i = 0
        for image_data in datalist:

            if image_data.image is not None:
                i+=1
                self.image_disp.add_frame("frame%d"%i,
                                          "%s image from shot #%d" % (image_data.name,self.event_number),
                                          (image_data.image,))
                #self.image_disp.frames["frame%d"%i].vmin = image_data.vrange[0]
                #self.image_disp.frames["frame%d"%i].vmax = image_data.vrange[1]
                if image_data.showProj: self.image_disp.frames["frame%d"%i].showProj = True

            if image_data.average is not None:
                i+=1
                self.image_disp.add_frame("frame%d"%i,
                                          "%s average of %d" % (image_data.name,image_data.counter),
                                          (image_data.average,))
                if image_data.showProj: self.image_disp.frames["frame%d"%i].showProj = True

            if image_data.avgdark is not None:
                i+=1
                self.image_disp.add_frame("frame%d"%i,
                                          "%s average dark of %d" % (image_data.name,image_data.ndark),
                                          (image_data.avgdark,))
                if image_data.showProj: self.image_disp.frames["frame%d"%i].showProj = True

            if image_data.maximum is not None:
                i+=1
                self.image_disp.add_frame("frame%d"%i,
                                          "%s maximum of %d" % (image_data.name,image_data.counter),
                                          (image_data.maximum,))
                if image_data.showProj: self.image_disp.frames["frame%d"%i].showProj = True

            self.image_disp.title = "Cameras shot#%d"%self.event_number

            print "Number of images in event ", self.event_number, ": ", len(self.image_disp.frames)
            
        newmode = self.image_disp.plot_all_frames(fignum=100,ordered=True)
        return newmode
    

                
#-------------------------------------------------------
# Plotter
#-------------------------------------------------------
class Frame(object):
    """Frame (axes/subplot) manager

    In principle one should think that the 'figure' and 'axes' 
    containers would be sufficient to navigate the plot, right?
    Yet, I find no way to have access to everything, like colorbar.
    # So I make my own container... let me know if you know a better way
    """
    def __init__(self, name="", title="", parent = None):
        self.name = name
        self.title = title
        self.plotter = parent

        self.data = None       # the array to be plotted
        self.ticks = {}        # tick marks for any axes if needed

        self.axes = None       # the patch containing the image
        self.axesim = None     # the image AxesImage

        self.colb = None       # the colorbar
        self.projx = None      # projection onto the horizontal axis
        self.projy = None      # projection onto the vertical axis

        # options
        self.plot_type = None 
        self.showProj = False  # only relevant for 2d plots
        self.extent = None
        self.axis_values = None
        self.aspect = 'auto'

        # threshold associated with this plot (image)
        self.threshold = None

        self.first = True

        # display-limits for this plot (image)
        self.vmin = None
        self.vmax = None
        self.orglims = None


    def myticks(self, x, pos):
        'The two args are the value and tick position'
        if self.axis_values is None:
            return x # change nothing

        try:
            val = self.axis_values[x]
            return '%1.0f'%val
        except:
            print "axis values out of range, %1.1f not in %s"%( x, str(self.axis_values.shape))
            return x


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


    def imshow( self, image, fignum=1, position=1):
        """ extension of plt.imshow with
        - interactive colorbar
        - optional axis projections
        - threshold management
        - display mode management
        """
        self.orglims = image.min(), image.max()

        if ( self.vmin is None) and (self.vmin is not None ):
            self.vmin = self.vmin
        if ( self.vmax is None) and (self.vmax is not None ):
            self.vmax = self.vmax

        # AxesImage
        myorigin = 'upper'
        if self.showProj>0 : myorigin = 'lower'
        self.axesim = self.axes.imshow( image,
                                        origin=myorigin,
                                        extent=self.extent,
                                        #interpolation='bilinear',
                                        vmin=self.vmin,
                                        vmax=self.vmax )

        divider = make_axes_locatable(self.axes)
        
        self.axes.set_title(self.title)

        if self.showProj>0 :
            self.projx = divider.append_axes("top", size="15%", pad=0.03,sharex=self.axes)
            self.projy = divider.append_axes("left", size="15%", pad=0.03,sharey=self.axes)
            self.projx.set_title( self.axes.get_title() )
            self.axes.set_title("") # clear the axis title

            maskedimage = np.ma.masked_array(image, mask=(image==0) )

            # --- average along each axis (or optionally maximum for special purposes)
            proj_vert, proj_horiz = None, None
            if self.showProj == 1:
                proj_vert = np.ma.average(maskedimage,1) # for each row, average of elements
                proj_horiz = np.ma.average(maskedimage,0) # for each column, average of elements
            elif self.showProj == 2: 
                proj_vert = np.ma.max(maskedimage,1) # for each row, max of elements
                proj_horiz = np.ma.max(maskedimage,0) # for each column, max of elements
               
            x1,x2,y1,y2 = self.axesim.get_extent()
            start_x = x1
            start_y = y1

            # vertical and horizontal dimensions, axes, projections
            vdim,hdim = self.axesim.get_size()        
            hbins = np.arange(start_x, start_x+hdim, 1)
            vbins = np.arange(start_y, start_y+vdim, 1)

            self.projx.plot(hbins,proj_horiz)
            self.projy.plot(proj_vert, vbins)
            self.projx.get_xaxis().set_visible(False)
        
            self.projx.set_xlim( start_x, start_x+hdim)
            self.projy.set_ylim( start_y+vdim, start_y)

            self.set_ticks()
            

        #cax = divider.append_axes("right",size="5%", pad=0.05)
        cax = divider.append_axes("right",size="5%", pad=0.05)
        self.colb = plt.colorbar(self.axesim,cax=cax, orientation='vertical')
        # colb is the colorbar object



        # show the active region for thresholding
        if self.threshold and self.threshold.area is not None:
            xy = [self.threshold.area[0],self.threshold.area[2]]
            w = self.threshold.area[1] - self.threshold.area[0]
            h = self.threshold.area[3] - self.threshold.area[2]
            self.thr_rect = plt.Rectangle(xy,w,h, facecolor='none', edgecolor='red', picker=10)
            self.axes.add_patch(self.thr_rect)
            print "Plotting the red rectangle in area ", self.threshold.area


        if self.plotter.display_mode == 2: # only if Interactive
            slider_vmin_ax = divider.append_axes("bottom",size="2%",pad=0.45)
            #slider_vmin_ax.patch.set_facecolor('blue')
            slider_vmax_ax = divider.append_axes("bottom",size="2%",pad=0.05)

            clim = self.colb.get_clim()
            edges = self.orglims

            # slider?!
            self.slider_vmin = Slider(slider_vmin_ax, 'min value', edges[0], edges[1],
                                      valinit=clim[0],
                                      facecolor='red', edgecolor='black')
            self.slider_vmax = Slider(slider_vmax_ax,'max value', edges[0], edges[1],
                                      valinit=clim[1], slidermin=self.slider_vmin,
                                      facecolor='red',edgecolor='black')
        
            def update(val):            
                self.axesim.set_clim(self.slider_vmin.val, self.slider_vmax.val)
                plt.draw()
            
            self.slider_vmin.on_changed(update)        
            self.slider_vmax.on_changed(update)
        


class Plotter(object):    
    """Figure (canvas) manager
    """
    def __init__(self):
        self.fig = None
        self.fignum = None
        # a figure has one or more plots/frames
        
        self.settings() # defaults

        self.frames = {} # dictionary / hash table to access the Frames

        self.display_mode = None
        # flag if interactively changed

        # global title
        self.title = ""
        
        self.threshold = None
        self.vmin = None
        self.vmax = None

        self.first = True
        self.cid1 = None
        self.cid2 = None

        self.set_ROI = False

        # matplotlib backend is set to QtAgg, and this is needed to avoid
        # a bug in raw_input ("QCoreApplication::exec: The event loop is already running")
        #QtCore.pyqtRemoveInputHook()

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
        

    def add_image_frame(self, name="", title="",contents=None, plot_type="image", aspect='auto'):
        """Forward to add_frame"""
        return self.add_frame(name,title,contents,plot_type,aspect)


    def add_frame(self, name="", title="",contents=None, plot_type=None, aspect='auto'):
        """Add a frame to the plotter. 
        @param  name       name of frame (must be unique, else returns the existing frame)
        @param  title      current title, may be different from event to event
        @param  contents   tuple of data arrays to be plotted (in one frame).
        @param  type       type of plot. Defaults based on contents
        @param  aspect     set aspect ratio of this frame (doesn't work)
        """
        aframe = None

        if name == "":
            name = "frame%d",len(self.frames)+1

        # Add this frame to plotter's list of frames.
        # If one with this name already exists, fetch it, don't make a new one
        if name in self.frames:
            aframe = self.frames[name]
        else :
            self.frames[name] = Frame(name, parent=self)
            aframe = self.frames[name]

        # copy any threshold as default
        aframe.threshold = self.threshold        
        
        aframe.title = title
        aframe.data = contents
        aframe.plot_type = plot_type
        aframe.aspect = aspect
        return aframe

    def plot_all_frames(self, fignum=1, ordered=False):
        """Draw all frames
        """
        nplots = len(self.frames)
        self.fignum = fignum

        ncol = int(np.ceil( np.sqrt(nplots) ))
        nrow = int(np.ceil( (1.0*nplots) / ncol ))

        figsize = (self.w*ncol, self.h*nrow)
        self.fig = plt.figure(fignum,figsize)
        self.fig.clf()

        self.fig.set_size_inches(self.w*ncol,self.h*nrow, forward=True)

        self.fig.suptitle(self.title)

        i = 1
        framenames = self.frames.iterkeys()
        if ordered : framenames = sorted(framenames)
        for name in framenames:

            frame = self.frames[name]
            frame.axes = self.fig.add_subplot(nrow,ncol,i)
            i+=1

            frame.axes.set_aspect(frame.aspect)

            titles = frame.title.split(';')
            frame.axes.set_title(titles[0])
            try:
                frame.axes.set_xlabel(titles[1])
                frame.axes.set_ylabel(titles[2])
            except:
                pass


            if frame.plot_type == "hist":
                n, bins, patches = plt.hist(frame.data, 60, histtype='stepfilled')

            elif frame.plot_type == 'image':
                frame.imshow(frame.data, fignum=fignum)

            elif frame.plot_type == 'scatter':
                points = plt.scatter(frame.data[0],frame.data[1])

            else :

                # single image (2d array)
                narrs = len(frame.data)
                ndims = frame.data[0].ndim
                
                if ndims == 2 :
                    if narrs == 1 :
                        frame.imshow(frame.data[0], fignum=fignum)
                    else :
                        print "utilities: Plotter:plot_all_frames: unsure what to do about this"
                    
                elif ndims == 1 :
                    # line plots
                
                    if narrs == 1 or narrs == 2:
                        plt.plot(*frame.data)

                    if narrs > 2 :
                        xaxis = frame.data[0]
                        others = frame.data[1:]
                        myargs = []
                        for o in others :
                            myargs.append( xaxis )
                            myargs.append( o )
                            myargs.append('-o')
                        
                        plt.plot(*myargs)

                    if len(frame.data)>1:
                        frame.axes.set_xlim(frame.data[0][0],frame.data[0][-1])

            if frame.axis_values is not None:
                formatter = ticker.FuncFormatter(frame.myticks)
                frame.axes.xaxis.set_major_formatter(formatter)
                

                
        self.fig.subplots_adjust(left=0.05,   right=0.95,
                                 bottom=0.05, top=0.90,
                                 wspace=0.2,  hspace=0.2 )
        self.connect()
        
        # for backward compatibility
        return self.display_mode
    
        
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


        self.fig = plt.figure(fignum)#,(self.w*ncol,self.h*nrow))
        self.fignum = fignum
        self.fig.clf()
        #self.fig.set_size_inches(self.w*ncol,self.h*nrow)

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
            self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.cid2 = self.fig.canvas.mpl_connect('pick_event', self.onpick)
        else :
            self.cid1 = self.fig.canvas.mpl_connect('button_press_event', plot.onclick)
            self.cid2 = self.fig.canvas.mpl_connect('pick_event', plot.onpick)
        #print "Now connected? ", self.cid1, self.cid2

    def onpick(self, event):
        print "The following clickable artist object was picked: ", event.artist

        # in which Frame?
        for aplot in self.frames.itervalues() :
            if aplot.axes == event.artist.axes : 

                print "Current   threshold = ", aplot.threshold.value
                print "          active area [xmin xmax ymin ymax] = ", aplot.threshold.area
                print "To change the Region of Interest, Left-click"
                if event.mouseevent.button==1:
                    self.set_ROI = True
                    print "************************************"
                    print "You can now select a new ROI        "
                    print "To cancel, right-click.             "
                    print "************************************"

                if event.mouseevent.button==3:
                    self.set_ROI = False
                    print "*************************************"
                    print "ROI selction has now been deactivated "
                    print "To select ROI again, left-click on the rectangle "
                    print "************************************"
                    

                """
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

                """            



    # define what to do if we click on the plot
    def onclick(self, event) :

        if self.first : 
            self.first = False
            print """
            To change the color scale, click on the color bar:
            - left-click sets the lower limit
            - right-click sets higher limit
            - middle-click resets to original
            """
        
        # -------------- clicks outside axes ----------------------
        # can we open a dialogue box here?
        if not event.inaxes and event.button == 3 :
            print "can we open a menu here?"
            #print "Close all mpl windows"
            #plt.close('all')
            

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
                    try:
                        aplot.slider_vmin.set_val(aplot.vmin)
                        aplot.slider_vmax.set_val(aplot.vmax)
                    except:
                        pass
                    plt.draw()

                
                if aplot.axes == event.inaxes: 
                    if self.set_ROI:
                        print "New area ", event.xdata, event.ydata
                        self.ROI_coordinates = plt.ginput(n=0) 
                        print "New coordinates", self.ROI_coordinates

    def plot_image(self, image, fignum=1, title="", showProj=0, extent=None):
        """ plot_image
        utility function for when plotting a single image outside of pyana
        """
        self.create_figure(fignum,1)
        self.fig.suptitle(title)
        self.drawframe(image, showProj=showProj, extent=extent)

        plt.draw()
        if self.display_mode == "Interactive" :
            plt.show()

############# This won't be missed (I think)
#
#    def plot_several(self, list_of_arrays, fignum=1, title="" ):
#        """ Draw several frames in one canvas
#        
#        @list_of_arrays          a list of tuples (title, array)
#        @fignum                  figure number, i.e. fig = plt.figure(num=fignum)
#        @return                  new display_mode if any (else return None)
#        """
#        #if self.fig is None: 
#        self.create_figure(fignum, nplots=len(list_of_arrays))
#        self.fig.suptitle(title)
            
#        pos = 0
#        for tuple in list_of_arrays :
#            pos += 1
#            ad = tuple[0]
#            im = tuple[1]
#            xt = None
#            if len(tuple)==3 : xt = tuple[2]

#            if type(im)==np.ndarray:
#                if len( im.shape ) > 1:
#                    self.drawframe(im,title=ad,fignum=fignum,position=pos)
#                else :
#                    plt.plot(im)
#            elif type(im)==tuple:
#                print "tuple"
#                pass
                
#        plt.draw()
#        return self.display_mode

    def draw_figurelist(self, fignum, event_display_images, title="",showProj=0,extent=None ) :
        """ Draw several frames in one canvas
        
        @fignum                  figure number, i.e. fig = plt.figure(num=fignum)
        @event_display_images    a list of tuples (name,title,image,extent=None)
        @return                  new display_mode if any (else return None)
        """
        #if self.fig is None: 
        self.create_figure(fignum, nplots=len(event_display_images))
        self.fig.suptitle(title)
            
        pos = 0
        for tuple in event_display_images :
            pos += 1
            name  = tuple[0]
            title = tuple[1]
            img   = tuple[2]

            xt = None
            if len(tuple)==4 : xt = tuple[3]
            
            self.drawframe(img,title=title,fignum=fignum,position=pos,showProj=showProj,extent=xt)
            
        plt.draw()
        return self.display_mode

    def draw_figure( self, frameimage, title="", fignum=1,position=1, showProj = 0,extent=None):
        """ Draw a single frame in one canvas
        """
        self.create_figure(fignum)
        self.fig.suptitle(title)
        self.drawframe(frameimage,title,fignum,position,showProj,extent)

        plt.draw()
        return self.display_mode


        
        
    def drawframe( self, frameimage, title="", fignum=1,position=1, showProj=0, extent=None):
        """ Draw a single interactive frame with optional projections and threshold
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
        myorigin = 'upper'
        if showProj>0 : myorigin = 'lower'

        aplot.axesim = aplot.axes.imshow( frameimage,
                                          origin=myorigin,
                                          extent=extent,
                                          interpolation='bilinear',
                                          vmin=aplot.vmin,
                                          vmax=aplot.vmax )

        divider = make_axes_locatable(aplot.axes)

        if showProj>0:
            aplot.projx = divider.append_axes("top", size="20%", pad=0.03,sharex=aplot.axes)
            aplot.projy = divider.append_axes("left", size="20%", pad=0.03,sharey=aplot.axes)
            aplot.projx.set_title( aplot.axes.get_title() )
            aplot.axes.set_title("") # clear the axis title

            # --- sum or average along each axis, 
            maskedimage = np.ma.masked_array(frameimage, mask=(frameimage==0) )

            proj_vert, proj_horiz = None, None
            if showProj == 1:
                proj_vert = np.ma.average(maskedimage,1) # for each row, average of elements
                proj_horiz = np.ma.average(maskedimage,0) # for each column, average of elements
            elif showProj == 2: 
                proj_vert = np.ma.max(maskedimage,1) # for each row, max of elements
                proj_horiz = np.ma.max(maskedimage,0) # for each column, max of elements
                           
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
            #print "Plotting the red rectangle in area ", aplot.threshold.area

        aplot.axes.set_title(title)
        
        
