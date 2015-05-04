import event_process
import psana
import os
import time
import logging
import toolbox
import pylab
from common import strtype

__version__ = '00.00.06'

class simple_trends(event_process.event_process):
    def __init__(self):
        self.output = event_process.event_process_output()
        self.reducer_rank = 0
        self.logger = logging.getLogger(__name__+'.simple_trends')
        return

    def set_stuff(self,psana_src,psana_device,device_attrs,period_window,in_report=None,in_report_title=None):
        self.src         = psana.Source(psana_src)
        self.dev         = psana_device
        self.dev_attrs   = device_attrs
        self.period_window = period_window
        self.output['in_report'] = in_report
        self.output['in_report_title'] = in_report_title
        return

    def replicate_info(self):
        args = ( str(self.src), strtype(self.dev), self.dev_attrs, self.period_window )
        kwargs = { 'in_report': self.output['in_report'], 'in_report_title': self.output['in_report_title'] }
        self.logger.info('args: {:}'.format(repr(args)))
        self.logger.info('kwargs: {:}'.format(repr(kwargs)))
        return ('set_stuff',args,kwargs)

    def beginJob(self):
        self.trends = {}
        for attr in self.dev_attrs:
            self.trends[attr] = toolbox.mytrend(self.period_window)
        return

    def event(self, evt):
        self.gas = evt.get(self.dev,self.src)
        if self.gas is None:
            return
        this_ts = self.parent.shared['timestamp'][0] + self.parent.shared['timestamp'][1]/1.0e9
        for attr in self.dev_attrs:
            val = getattr(self.gas,attr)()     
            self.trends[attr].add_entry( this_ts, val )

    def endJob(self):
        self.reduced_trends = {}
        for attr in self.dev_attrs:
            self.logger.info('mpi reducing {:}'.format(attr))
            self.logger.info('reducer_rank={:} ranks={:}'.format(repr(self.reducer_rank),repr(self.reduce_ranks)))
            self.reduced_trends[attr] = self.trends[attr].reduce(self.parent.comm,ranks=self.reduce_ranks,reducer_rank=self.reducer_rank,tag=34)

        if self.parent.rank == self.reducer_rank:
            fig = pylab.figure()
            for attr in self.dev_attrs:
                self.output['figures'][attr] = {}
                fig.clear()
                thisxs    = self.reduced_trends[attr].getxs()
                thismeans = self.reduced_trends[attr].getmeans()
                thismins = self.reduced_trends[ attr].getmins()
                thismaxs = self.reduced_trends[ attr].getmaxs()
                pylab.plot(thisxs,thismeans,'k')
                pylab.plot(thisxs,thismaxs,'r')
                pylab.plot(thisxs,thismins,'b')
                pylab.xlabel('time [sec]')
                pylab.ylabel('value [min/mean/max]')
                pylab.title(attr)
                pylab.savefig( os.path.join( self.output_dir, 'figure_trend_{:}.png'.format(attr) ))
                self.output['figures'][attr]['png'] = os.path.join( self.output_dir, 'figure_trend_{:}.png'.format( attr ))
                # finish this section
            del fig
            self.parent.output.append(self.output)
        return

