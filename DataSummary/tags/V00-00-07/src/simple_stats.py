import event_process
import psana
import os
import logging
import toolbox
import pylab
from common import strtype

__version__ = '00.00.06'

class simple_stats(event_process.event_process):
    def set_stuff(self,psana_src,psana_device,device_attrs,hist_ranges,in_report=None,in_report_title=None):
        self.src         = psana.Source(psana_src)
        self.dev         = psana_device
        if len(device_attrs) != len(hist_ranges):
            raise Exception('lenght mismatch between psana_device and hist_ranges')
        self.dev_attrs   = device_attrs # must be a list
        self.hist_ranges = hist_ranges  # must be a dict of the same length as dev_attrs
        self.output['in_report'] = in_report
        self.output['in_report_title'] = in_report_title
        self.logger = logging.getLogger(__name__+'.simple_stats')

    def replicate_info(self):
        args = ( str(self.src), strtype(self.dev), tuple(self.dev_attrs), self.hist_ranges )
        kwargs = { 'in_report': self.output['in_report'], 'in_report_title': self.output['in_report_title'] }
        self.logger.info('args: {:}'.format(repr(args)))
        self.logger.info('kwargs: {:}'.format(repr(kwargs)))
        return ('set_stuff',args,kwargs)

    def beginJob(self):
        self.histograms = {}
        for attr in self.dev_attrs:
            self.histograms[attr] = toolbox.myhist(*self.hist_ranges[attr])
        return


    def event(self,evt):
        self.gas = evt.get(self.dev,self.src)
        if self.gas is None:
            return
        for attr in self.dev_attrs:
            val = getattr(self.gas,attr)()
            #print attr, val
            self.histograms[attr].fill( getattr(self.gas,attr)() )

    def endJob(self):
        self.reduced_histograms = {}
        for attr in self.dev_attrs:
            self.logger.info('mpi reducing {:}'.format(attr))
            self.logger.info('reducer_rank={:} ranks={:}'.format(repr(self.reducer_rank),repr(self.reduce_ranks)))
            self.reduced_histograms[attr] = self.histograms[attr].reduce(self.parent.comm,ranks=self.reduce_ranks,reducer_rank=self.reducer_rank,tag=33)

        if self.parent.rank == self.reducer_rank:
            # plot those histograms
            self.output['figures'] = {}
            self.output['table'] = {}
            fig = pylab.figure()
            for attr in self.dev_attrs:
                self.output['figures'][attr] = {}
                fig.clear()
                #self.step = pylab.step( self.histograms[attr].edges, self.reduced_histograms[attr])
                # do this to make a filled step plot of the histogram
                newX, newY = self.reduced_histograms[attr].mksteps()
                pylab.fill_between( newX, 0, newY[:-1] )
                pylab.title( attr )
                pylab.xlabel('value')
                pylab.ylabel('count [per bin]')
                pylab.xlim( self.histograms[attr].minrange, self.histograms[attr].maxrange )
                pylab.ylim( 0 , max(self.reduced_histograms[attr].binentries)*1.1 )
                pylab.savefig( os.path.join( self.output_dir, 'figure_{:}.png'.format( attr ) ))
                self.output['figures'][attr]['png'] = os.path.join( self.output_dir, 'figure_{:}.png'.format( attr ))

                self.output['table'][attr] = {}
                self.output['table'][attr]['Mean'] = self.reduced_histograms[attr].mean()
                self.output['table'][attr]['RMS'] = self.reduced_histograms[attr].std()
                self.output['table'][attr]['min'] = self.reduced_histograms[attr].minval
                self.output['table'][attr]['max'] = self.reduced_histograms[attr].maxval
            del fig
            self.parent.output.append(self.output)
        return


