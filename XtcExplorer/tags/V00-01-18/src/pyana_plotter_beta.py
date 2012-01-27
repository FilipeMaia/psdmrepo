#--------------------------------------------------------------------------
# Description:
#  Pyana user analysis module pyana_plotter_beta...
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $

@author Ingrid Ofte
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 1095 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging
import time

#import matplotlib 
#matplotlib.use('Qt4Agg')

# alternative 1: pyplot (matlab-like)
import matplotlib.pyplot as plt

## alternative 2: object oriented matplotlib
#import matplotlib as mpl
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
#from matplotlib.figure import Figure

from IPython.Shell import IPShellEmbed

#-----------------------------
# Imports for other modules --
#-----------------------------
from utilities import PyanaOptions 
from utilities import Plotter

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
#class ImagePlot(FigureCanvas) :
#    def __init__(self, parent=None,
#                 width=10, height=8, dpi=100, bgcolor=None, num=1 ):
#        fig = Figure(figsize=(width,height),dpi=dpi,facecolor=bgcolor, edgecolor=bgcolor)
#        FigureCanvas.__init__(self, fig)
#        self.setParent(parent)
#        fig.suptitle("ImagePlot Changed")
#        self.axes = fig.add_subplot(111)
#        #self.toolbar = NavigationToolbar(self,self)

#    def draw_array(self,array):
#        self.axesim = self.axes.imshow( array, origin='lower' )
#        # axes image 
#        self.show()


class pyana_plotter_beta (object) :
    """Class whose instance will be used as a user analysis module. """

    #--------------------
    #  Class variables --
    #--------------------
    
    # usual convention is to prefix static variables with s_
    s_staticVariable = 0

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self,
                   display_mode = "Interactive",
                   ipython         = "False",
                   printprogress   = "False" ):
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.

        @param display_mode        Interactive (1) or SlideShow (2) or NoDisplay (0)
        @param ipython             Drop into ipython at the end of the job
        """
        self.n_shots = 0

        self.display_mode = None
        if display_mode == "NoDisplay" : self.display_mode = 0
        if display_mode == "Interactive" : self.display_mode = 1
        if display_mode == "SlideShow" :   self.display_mode = 2
        if self.display_mode is None:
            print "Unknown display mode %s, using NoDisplay (0)"%display_mode
            self.display_mode = 0

        opt = PyanaOptions() # convert option string to appropriate type        
        self.ipython       = opt.getOptBoolean(ipython)
        self.printprogress = opt.getOptBoolean(printprogress)

        self.plotter = Plotter()        
        self.plotter.settings(7,7) # set default frame size

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job. It should
        do a one-time initialization possible extracting values from event
        data (which is a Configure object) or environment.

        @param evt    event data object
        @param env    environment object
        """
        self.starttime = time.time()
        
        # Preferred way to log information is via logging package
        logging.info( "pyana_plotter_beta.beginjob() called with displaymode %d"%self.display_mode )

        if self.display_mode == 0 :
            plt.ioff()
        if self.display_mode == 1 :
            plt.ioff()
        if self.display_mode == 2 :
            plt.ion()


    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "pyana_plotter_beta.beginrun() called" )

    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "pyana_plotter_beta.begincalibcycle() called" )

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        self.n_shots += 1

        # print progress
        if self.printprogress : self.progress_report()

        if evt.get('skip_event'):
            return
        
        # only call the plt.draw / plt.show is there's actually
        # something new to show, since it's slow. 
        if not evt.get('show_event'):
            return

        # only call plotter if this is the main thread
        if (env.subprocess()>0): return
            

        # OK, go ahead
        
        #----------------------------------------------------------
        # for xtcexplorernew: 
        #----------------------------------------------------------
        plot_data = evt.get('plot_data')

        if plot_data is not None: 
            self.make_plots(fignum=100,datalist=plot_data)
            plt.suptitle("Event#%d"%self.n_shots)

                
        #----------------------------------------------------------
        # for xtcexplorer (old)
        #----------------------------------------------------------
        event_display_images = evt.get( 'event_display_images')
        if event_display_images is not None: 
            print "pyana_plotter_beta: Plotting %d plots"%len(event_display_images)
            newmode = self.plotter.draw_figurelist(200,
                                                       event_display_images,
                                                       title="Cameras shot#%d"%self.n_shots,
                                                       showProj=False)
            if newmode is not None:
                self.display_mode = newmode
                    

            if self.ipython :
                plt.draw()
                self.launch_ipython(evt)
            
            if self.display_mode == 1:
                # Interactive
                plt.ioff()
                plt.show()
            
            elif self.display_mode == 2:
                # SlideShow
                plt.ion()
                plt.draw()            
            
            # wait for 'enter' before proceeding
            # raw_input('Please hit \'enter\' to proceed to the next event') 

    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """
        
        logging.info( "pyana_plotter_beta.endcalibcycle() called" )

    def endrun( self, env ) :
        """This optional method is called if present at the end of the run.
        
        @param env    environment object
        """        
        logging.info( "pyana_plotter_beta.endrun() called" )


    def endjob( self, evt, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        
        logging.info( "pyana_plotter_beta.endjob() called" )
        endtime = time.time()
        duration = endtime - self.starttime
        #print "Start: %.3f, Stop: %.3f, Duration: %.4f" %(self.starttime,endtime,duration)
        print "\nTiming as measured by pyana_plotter_beta endjob: %.4f s\n" %(duration)
                                
        if (env.subprocess()<1):        
            
            #----------------------------------------------------------
            # for xtcexplorernew: 
            #----------------------------------------------------------
            plot_data = evt.get('plot_data')

            if plot_data is not None: 
                self.make_plots(fignum=201,datalist=plot_data)
                plt.suptitle("End Job : Average of all events")
            

                
            #----------------------------------------------------------
            # for xtcexplorer (old)
            #----------------------------------------------------------
            event_display_images = evt.get( 'event_display_images')
            if event_display_images: 
                print "pyana_plotter_beta: Plotting %d plots"%len(event_display_images)
                self.plotter.draw_figurelist(201,
                                             event_display_images,
                                             title="End job",
                                             showProj=False)
            
                plt.draw()
            
                if self.ipython :
                    self.launch_ipython(evt)
            
                print "Pyana will exit once you close all the MatPlotLib windows"            
                if self.display_mode > 0 :
                    plt.ioff()
                    plt.show()


        print "-------------------"
        print "Done running pyana."
        print "To run pyana again, edit config file if needed and hit \"Run pyana\" button again"
        print "Send any feedback on this program to ofte@slac.stanford.edu"
        print "Thank you!"
        print "-------------------"


    def launch_ipython(self, evt):
        """Launch an ipython session with access to data stored in the evt object
        """
        # get pointer to the data from each of the modules
        data_ipimb = evt.get('data_ipimb')
        if data_ipimb :  print "data_ipimb: ", data_ipimb
        else :           del data_ipimb
        
        data_bld = evt.get('data_bld')
        if data_bld :   print "data_bld: ", data_bld
        else :          del data_bld
        
        data_epics = evt.get('data_epics')
        if data_epics :   print "data_epics: ", data_epics
        else :          del data_epics
        
        data_scan = evt.get('data_scan')
        if data_scan :  print "data_scan: ", data_scan
        else :          del data_scan
        
        data_encoder = evt.get('data_encoder')
        if data_encoder :  print "data_encoder: ", data_encoder
        else :          del data_encoder
        
        data_waveform = evt.get('data_waveform')
        if data_waveform :  print "data_waveform: ", data_waveform
        else :          del data_waveform
        
        data_image = evt.get('data_image')
        if data_image:  print "data_image: ", data_image
        else:           del data_image
        
        data_cspad = evt.get('data_cspad')
        if data_cspad:  print "data_cspad: ", data_cspad
        else:           del data_cspad
        
        ipshell = IPShellEmbed(argv=['-pi1','In \\# >> ','-po','Out \\#: '], 
                               banner='--------- Dropping into iPython ---------',
                               exit_msg='--------- Leaving iPython -------------')
        
        ipshell("Called from endjob. \nTry 'whos' to see the workspace. " \
                "\nHit Ctrl-D to exit iPython and continue program.")
        
        
    def progress_report(self):

        if self.n_shots <= 10 :
            print "Event # ", self.n_shots
        elif self.n_shots < 100 :
            if (self.n_shots % 10)==0 :
                print "Event # ", self.n_shots
        elif self.n_shots < 1000 :
            if (self.n_shots % 100)==0 :
                print "Event # ", self.n_shots
        elif self.n_shots < 10000 :
            if (self.n_shots % 1000)==0 :
                print "Event # ", self.n_shots
        else:
            if (self.n_shots % 10000)==0 :
                print "Event # ", self.n_shots



    def make_plots(self,fignum=1,datalist=None):

        if datalist is None:
            print "Cannot plot nothing"
            return

        print "pyana_plotter event %d has found %d sources to plot from"%(self.n_shots,len(datalist))
        
        for data in datalist:
            for name,array in data.get_plottables().iteritems():
                title = "%s %s"%(data.name, name)
                contents = (array,) # a tuple

                #print title, [item.shape for item in contents]
                self.plotter.add_frame(name,title,contents)
                

        self.plotter.plot_all_frames(fignum=fignum,ordered=True)
                    
        if self.ipython :
            plt.draw()
            self.launch_ipython(evt)
            
        if self.display_mode == 1:
            # Interactive
            plt.ioff()
            plt.show()
                
        elif self.display_mode == 2:
            # SlideShow
            plt.ion()
            plt.draw()            

        else :
            print "Displaymode?? ", self.display_mode
