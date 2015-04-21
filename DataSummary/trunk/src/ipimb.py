import os
import event_process
import logging
import time
import toolbox
import pylab
import psana
from common import strtype

__version__ = '00.00.06'

class ipimb(event_process.event_process):
    def __init__(self):
        self.output = event_process.event_process_output()
        self.reducer_rank = 0
        self.logger = logging.getLogger(__name__+'.ipimb')

        return

    def set_stuff(self,psana_src,psana_device,in_report=None,in_report_title=None,period_window=1.):
        self.src         = psana.Source(psana_src)
        self.dev         = psana_device
        self.output['in_report'] = in_report
        self.output['in_report_title'] = in_report_title
        self.period_window = period_window
        return

    def replicate_info(self):
        args = ( str(self.src), strtype(self.dev))
        kwargs = { 'in_report': self.output['in_report'], 'in_report_title': self.output['in_report_title'], 
                'period_window': self.period_window }
        self.logger.info('args: {:}'.format(repr(args)))
        self.logger.info('kwargs: {:}'.format(repr(kwargs)))
        return ('set_stuff',args,kwargs)
    
    def beginJob(self):
        #print "rank {:}".format(self.parent.rank)
        self.trends = {}
        for chan in ['channel 0','channel 1','channel 2','channel 3','sum','xpos','ypos']:
            self.trends[chan] = toolbox.mytrend(self.period_window)
        return

    def beginRun(self):
        return

    def event(self, evt):
        this_ts = self.parent.shared['timestamp'][0] + self.parent.shared['timestamp'][1]/1.0e9
        self.raw_ipimb        = evt.get(self.dev,self.src)
        self.sum_ipimb        = self.raw_ipimb.sum()
        self.channels_ipimb   = self.raw_ipimb.channel()
        self.xpos_ipimb       = self.raw_ipimb.xpos()
        self.ypos_ipimb       = self.raw_ipimb.ypos()
        self.trends['xpos'].add_entry(this_ts,self.xpos_ipimb)
        self.trends['ypos'].add_entry(this_ts,self.ypos_ipimb)
        self.trends['sum'].add_entry(this_ts,self.sum_ipimb)
        self.trends['channel 0'].add_entry(this_ts,self.channels_ipimb[0])
        self.trends['channel 1'].add_entry(this_ts,self.channels_ipimb[1])
        self.trends['channel 2'].add_entry(this_ts,self.channels_ipimb[2])
        self.trends['channel 3'].add_entry(this_ts,self.channels_ipimb[3])
        return

    def endRun(self):
        return

    def endJob(self):
        self.reduced_trends = {}
        for chan in self.trends:
            self.logger.info('mpi reducing {:}'.format(chan))
            self.reduced_trends[chan] = self.trends[chan].reduce(self.parent.comm,reducer_rank=self.reducer_rank,tag=44,ranks=self.reduce_ranks)

        if self.parent.rank == self.reducer_rank:
            self.output['text'].append('PVs trended below the fold: <br/>\n<pre>')
            for chan in self.trends:
                self.output['text'][-1] += chan+'\n'
            self.output['text'][-1]     += '</pre>'
            fig = pylab.figure()
            for chan in self.trends:
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
