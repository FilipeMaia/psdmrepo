import pylab
import time
import logging
import toolbox
import event_process
import os

class epics_trend(event_process.event_process):
    def __init__(self):
        self.logger                      = logging.getLogger(__name__+'.epics_trend')
        self.output = event_process.event_process_output()
        self.reducer_rank                = 0
        self.period_window               = 1.
        self.channels_to_trend           = []
        self.output['in_report']         = None
        self.output['in_report_title']   = None

    def replicate_info(self):
        return ('add_pv_trend',tuple(self.channels_to_trend),{})

    def beginJob(self):
        self.epics       = self.parent.ds.env().epicsStore()
        self.allPvs      = self.epics.names()
        self.trends      = {}
        self.chans_to_remove = []
        for chan in self.channels_to_trend:
            if chan not in self.allPvs:
                self.logger.warning('channel {:} not in PV list, removing from trending'.format(chan))
                self.chans_to_remove.append(chan)

        for chan in self.chans_to_remove:
            self.channels_to_trend.remove(chan)

        for chan in self.channels_to_trend:
            self.trends[chan] = toolbox.mytrend(self.period_window)
        return

    def add_pv_trend(self,*chans):
        for chan in chans:
            if chan not in self.channels_to_trend:
                self.channels_to_trend.append(chan)
        return

    def event(self,evt):
        this_ts = self.parent.shared['timestamp'][0] + self.parent.shared['timestamp'][1]/1.0e9
        for chan in self.channels_to_trend:
            val = self.epics.value(chan)     
            #self.logger.debug('{:} = {:}'.format(chan,val))
            self.trends[chan].add_entry( this_ts, val )
        return


    def endJob(self):
        self.reduced_trends = {}
        for chan in self.channels_to_trend:
            self.logger.info('mpi reducing {:}'.format(chan))
            self.reduced_trends[chan] = self.trends[chan].reduce(self.parent.comm,reducer_rank=self.reducer_rank,tag=55,ranks=self.reduce_ranks)

        if self.parent.rank == self.reducer_rank:
            self.output['text'].append('All available PVs in the EPICS store: <select><option>--</option>\n')
            for chan in self.allPvs:
                self.output['text'][-1] += '<option>{:}</option>\n'.format(chan)
            self.output['text'][-1]     += '</select>\n'
            self.output['text'].append('PVs trended below the fold: <br/>\n<pre>')
            for chan in self.channels_to_trend:
                self.output['text'][-1] += chan+'\n'
            self.output['text'][-1]     += '</pre>'
            fig = pylab.figure()
            for chan in self.channels_to_trend:
                self.output['figures'][chan] = {}
                fig.clear()
                thisxs    = self.reduced_trends[chan].getxs()
                thismeans = self.reduced_trends[chan].getmeans()
                thismins = self.reduced_trends[ chan].getmins()
                thismaxs = self.reduced_trends[ chan].getmaxs()
                pylab.plot(thisxs,thismeans,'k')
                pylab.plot(thisxs,thismaxs,'r')
                pylab.plot(thisxs,thismins,'b')
                pylab.title(chan)
                pylab.xlabel('time [sec]')
                pylab.ylabel('value [min/mean/max]')
                pylab.savefig( os.path.join( self.output_dir, 'figure_trend_{:}.png'.format(chan.replace(':','_')) ))
                self.output['figures'][chan]['png'] = os.path.join( self.output_dir, 'figure_trend_{:}.png'.format( chan.replace(':','_') ))
                # finish this section
            del fig
            self.parent.output.append(self.output)
        return
