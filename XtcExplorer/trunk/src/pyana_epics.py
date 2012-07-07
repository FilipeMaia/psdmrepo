#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z salnikov $
#
# Description:
#  Pyana user analysis module pyana_epics...
#
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

#-----------------------------
# Imports for other modules --
#-----------------------------
import numpy as np
import matplotlib.pyplot as plt

from pypdsdata import xtc
from pypdsdata import epics

from utilities import PyanaOptions
from utilities import EpicsData
from utilities import ncol_nrow_from_nplots


#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class pyana_epics (object) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, pv_names = None,
                   plot_every_n  = None,
                   accumulate_n  = "0",
                   fignum = "1" ) :
        """Class constructor
        All parameters are passed as strings
        @param pv_names       List of name(s) of the EPICS PV(s) to plot
        @param plot_every_n   Frequency for plotting. If n=0, no plots till the end
        @param accumulate_n   Accumulate all (0) or reset the array every n shots
        @param fignum         Matplotlib figure number
        """

        opt = PyanaOptions()
        self.pv_names = opt.getOptStrings(pv_names)
        self.plot_every_n = opt.getOptInteger(plot_every_n)
        self.accumulate_n = opt.getOptInteger(accumulate_n)
        self.mpl_num = opt.getOptInteger(fignum)

        # other
        self.n_shots = None
        self.accu_start = None        

        # lists to fill numpy arrays
        self.initlists()
        
    def initlists(self):
        self.pv_value = {}
        self.pv_shots = {}
        self.pv_status = {}
        self.pv_severity = {} 
        self.prev_val = {}
        for pv_name in self.pv_names :
            self.pv_value[pv_name] = []
            self.pv_shots[pv_name] = []
            self.pv_status[pv_name] = []
            self.pv_severity[pv_name] = []
            self.prev_val[pv_name] = None

    def resetlists(self):
        self.accu_start = self.n_shots
        for pv_name in self.pv_names :
            del self.pv_value[pv_name][:]
            del self.pv_shots[pv_name][:]
            del self.pv_status[pv_name][:]
            del self.pv_severity[pv_name][:]
            self.prev_val[pv_name] = None
                
        
    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        # Preferred way to log information is via logging package
        logging.info( "pyana_epics.beginjob() called")

        self.n_shots = 0
        self.accu_start = 0
        

        # Use environment object to access EPICS data
        self.epics_data = {}
        for pv_name in self.pv_names :

            pv = env.epicsStore().value( pv_name )
            if not pv:
                logging.warning('EPICS PV %s does not exist', pv_name)
            else :

                # The returned value should be of the type epics.EpicsPvCtrl.
                #print "PV %s: id=%d type=%d size=%d status=%s severity=%s values=%s" % \
                #      (pv_name, pv.iPvId, pv.iDbrType, pv.iNumElements,
                #       pv.status, pv.severity, pv.values)
                
                message = "%s (Id=%d, Type=%d, Precision=%s, Unit=%s )\n" \
                          %(pv_name,pv.iPvId, pv.iDbrType, str(pv.precision), pv.units)

                message += "   Status = %d , Severity = %d \n"%(pv.status, pv.severity)
                try:
                    message += "(%s), (%s)\n" \
                               %(epics.epicsAlarmConditionStrings[pv.status],
                                 epics.epicsAlarmSeverityStrings[pv.severity])
                except:
                    pass
                
                message += "   Limits: " 
                message += "   Ctrl: [%s - %s]" % (pv.lower_ctrl_limit, pv.upper_ctrl_limit)
                message += "   Display: [%s - %s]" % (pv.lower_disp_limit, pv.upper_disp_limit) 
                message += "   Warning: [%s - %s]" % (pv.lower_warning_limit, pv.upper_warning_limit) 
                message += "   Alarm: [%s - %s]" % (pv.lower_alarm_limit, pv.upper_alarm_limit)
                
                
                print message
                self.epics_data[pv_name] = EpicsData(pv_name)

        
    def event( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        self.n_shots += 1
        logging.info( "pyana_epics.event() called (%d)"%self.n_shots )


        if evt.get('skip_event') :
            return

        # --------- Reset -------------
        if self.accumulate_n!=0 and (self.n_shots%self.accumulate_n)==0 :
            self.resetlists()


        # Use environment object to access EPICS data
        for pv_name in self.pv_names :

            ## The returned value should be of the type epics.EpicsPvTime.
            pv = env.epicsStore().value( pv_name )
            if not pv:
                logging.warning('EPICS PV %s does not exist', pv_name)
            else:
                # only add to list if it changed since last event
                if pv.value != self.prev_val[pv_name] :
                    self.prev_val[pv_name] = pv.value

                    self.pv_value[pv_name].append( pv.value )
                    self.pv_shots[pv_name].append( self.n_shots )
                    self.pv_status[pv_name].append( pv.status )
                    self.pv_severity[pv_name].append( pv.severity )


        # ----------------- Plotting ---------------------
        if self.plot_every_n != 0 and (self.n_shots%self.plot_every_n)==0 :
            header = "EpicsPV plots shots %d-%d" % (self.accu_start, self.n_shots)
            self.make_plots(header)

            # flag for pyana_plotter
            evt.put(True, 'show_event')
            
            data_epics = []
            for pv_name in self.pv_names :
                data_epics.append( self.epics_data[pv_name] )
                # send it to the evt object
            evt.put(data_epics, 'data_epics')
                        

    def endjob( self, evt, env ) :
        """
        @param evt    optional
        @param env    environment object
        """
        
        logging.info( "pyana_epics.endjob() called" )

        # ----------------- Plotting ---------------------
        self.make_plots("EpicsPV plots at endjob")
        data_epics = []
        for pv_name in self.pv_names :
            data_epics.append( self.epics_data[pv_name] )
        # send it to the evt object
        evt.put(data_epics, 'data_epics')
 

    def make_plots(self, title=""):
        
        # -------- Begin: move this to beginJob
        """ This part should move to begin job, but I can't get
        it to update the plot in SlideShow mode when I don't recreate
        the figure each time. Therefore plotting is slow...
        """
        nplots = len(self.pv_names)
        if nplots == 0:
            return
        (ncols, nrows) = ncol_nrow_from_nplots(nplots)
            
        height=3.5
        if nrows * 3.5 > 12 : height = 12/nrows
        width=height*1.3
        
        fig = plt.figure(num=self.mpl_num, figsize=(width*ncols,height*nrows) )
        fig.clf()
        fig.subplots_adjust(wspace=0.45, hspace=0.45, top=0.85, bottom=0.15)
        fig.suptitle(title)

        if self.n_shots < 2:
            return
        # -------- End: move this to beginJob
            
        i = 0
        for pv_name in self.pv_names :
            i+=1

            fig.add_subplot(nrows, ncols, i) 

            # append one more bin to the shots array (bin boundaries)
            if self.n_shots != self.pv_shots[pv_name][-1] :
                self.pv_shots[pv_name].append( self.n_shots )

            shots_array = np.float_( self.pv_shots[pv_name] )
            values_array = np.float_( self.pv_value[pv_name] )
            status_array = np.float_( self.pv_status[pv_name] )
            severity_array = np.float_( self.pv_severity[pv_name] )


            self.epics_data[pv_name].values = values_array
            self.epics_data[pv_name].shotnr = shots_array
            self.epics_data[pv_name].status = status_array
            self.epics_data[pv_name].severity = severity_array

            nbins = values_array.size
            if nbins == 0 :
                print "No bins for ", pv_name, self.n_shots, self.accu_start
                continue
            
            ymax = np.amax( values_array )
            ymin = np.amin( values_array )
            span = ymax - ymin

            try:
                (n,bins,patches) = plt.hist(shots_array[:nbins],
                                            weights=values_array,
                                            bins=shots_array,
                                            histtype='step')
            except Exception, err:
                if str(err) == "zero-size array to ufunc.reduce without identity":
                    # For some reason, hist() removes all 0 values,
                    # but if all the values are zero, this results
                    # in an empty array and thus the error above.
                    # So add a tiny number to each value and try again.
                    values_array += 1e-20
                    (n,bins,patches) = plt.hist(shots_array[:nbins],
                                                weights=values_array,
                                                bins=shots_array,
                                                histtype='step')
                else:
                    raise

            plt.xlim(shots_array[0],shots_array[-1])
            if span > 0:
                plt.ylim(ymin-0.3*span,ymax+0.3*span)

            plt.plot(shots_array[:nbins], values_array,'bo')

            plt.title(pv_name)
            plt.xlabel("Shot number",horizontalalignment='left') # the other right
            plt.ylabel("PV value ",horizontalalignment='right')
            plt.draw()
        plt.draw()
        
