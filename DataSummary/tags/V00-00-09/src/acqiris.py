import event_process
import psana
import os
import logging
import toolbox
import pylab
from common import strtype

__version__ = '00.00.06'

class acqiris(event_process.event_process):
    def __init__(self):
        self.data = {}
        self.output = event_process.event_process_output()
        self.logger = logging.getLogger(__name__+'.acqiris')
        self.reducer_rank = 0
        return

    def set_stuff(self,src,psana_device,in_report=None,in_report_title=None,histmin=400,histmax=450):
        self.src = psana.Source(src)
        self.dev = psana_device
        self.output['in_report'] = in_report
        self.output['in_report_title'] = in_report_title
        self.histmin = histmin
        self.histmax = histmax
        return

    def replicate_info(self):
        args = ( str(self.src), strtype(self.dev))
        kwargs = { 'in_report': self.output['in_report'], 'in_report_title': self.output['in_report_title'], 
                'histmin': self.histmin, 'histmax': self.histmax }
        self.logger.info('args: {:}'.format(repr(args)))
        self.logger.info('kwargs: {:}'.format(repr(kwargs)))
        return ('set_stuff',args,kwargs)

    def event(self,evt):
        """
        this should be expanded to look at all traces in general, and specify which ones to look
        at with the init command
        """
        self.raw_traces = evt.get(self.dev,self.src)
        if self.raw_traces is None:
            self.logger.error('No acqiris found in event {:}'.format(self.parent.eventN))
            return
        #self.logger.debug( 'acqiris traces = {:}'.format(self.raw_traces.data_shape() ))
        self.traces = { tr: list(self.raw_traces.data(tr).waveforms()[0]) for tr in xrange( self.raw_traces.data_shape()[0] ) }
        self.peaks  = { tr: self.traces[tr].index( max(self.traces[tr]) ) for tr in self.traces }
        for evr in self.parent.shared['evr']:
            #print "rank {:} evr {:} peak {:}".format(self.parent.rank, evr, peak )
            for tr in self.peaks:
                self.data.setdefault( "{:}-{:0.0f}".format(evr,tr), toolbox.myhist(5000,0,20000)).fill( self.peaks[tr] )
        return

    def endJob(self):
        # do the stuff on all the other nodes (e.g. send stuff to reducer_rank)
        self.reduced_data = {}
        for evr in self.data:
            self.logger.info('mpi reducing {:}'.format(evr))
            self.logger.info('reducer_rank={:} ranks={:}'.format(repr(self.reducer_rank),repr(self.reduce_ranks)))
            self.reduced_data[evr] = self.data[evr].reduce(self.parent.comm,ranks=self.reduce_ranks,reducer_rank=self.reducer_rank,tag=11)

        if self.parent.rank == self.parent._reducer_rank[(self.__class__.__name__, repr( self.describe_self() ) )]:
            fig = pylab.figure()
            for evr in sorted(self.reduced_data):
                fig.clear()
                newdata = self.reduced_data[evr]
                self.logger.info( "{:} mean: {:0.2f}, std: {:0.2f}, min {:0.2f}, max {:0.2f}".format( evr, newdata.mean(), newdata.std(), newdata.minval, newdata.maxval ) )
                self.output['table'][evr] = {}
                self.output['table'][evr]['Mean Arrival Bin'] = newdata.mean()
                self.output['table'][evr]['RMS'] = newdata.std()
                self.output['table'][evr]['min'] = newdata.minval
                self.output['table'][evr]['max'] = newdata.maxval

                self.output['figures'][evr] = {}
                #hh = pylab.hist(newdata,bins=100,range=(0,100))
                newX, newY = self.reduced_data[evr].mksteps()
                hh = pylab.fill_between( newX, 0, newY[:-1] )
                pylab.title( 'acqiris arrival bin for evr '+repr(evr) )
                pylab.xlabel('value')
                pylab.ylabel('count [per bin]')
                self.logger.info("hist min, hist max: ignored")
                #pylab.xlim(self.histmin,self.histmax)
                pylab.ylim( 0 , max(self.reduced_data[evr].binentries)*1.1 )
                #pylab.savefig( os.path.join( self.parent.output_dir, 'figure_evr_{:}.pdf'.format( evr ) ) )
                #self.output['figures'][evr]['pdf'] = os.path.join( self.parent.output_dir, 'figure_evr_{:}.pdf'.format( evr ) )
                pylab.savefig( os.path.join( self.output_dir, 'figure_evr_{:}.png'.format( evr ) ) )
                self.output['figures'][evr]['png'] = os.path.join( self.output_dir, 'figure_evr_{:}.png'.format( evr ) )
                self.logger.info( 'figure saved: {}'.format( self.output['figures'][evr]['png']) )
            del fig
            self.parent.output.append(self.output)
        return

