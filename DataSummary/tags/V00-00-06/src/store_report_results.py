import logging
import event_process
import os
import pprint

class store_report_results(event_process.event_process):
    def __init__(self):
        self.output = event_process.event_process_output()
        self.reducer_rank = 0 # this has no effect
        self.logger = logging.getLogger(__name__+'.store_report_results')
        return

    def endJob(self):
        self.parent.gather_output()
        if self.parent.rank == 0:
            self.output['in_report'] = 'meta'
            self.output['in_report_title'] = 'Report Results'
            self.output['text'] = ["<a href=report.py>Report.py</a>",]
            self.parent.output.append(self.output)
            self.parent.gathered_output.append(self.output)
            outfile = open( os.path.join(self.parent.output_dir,'report.py'), 'w' )
            outfile.write( 'report = ' )
            outfile.write( pprint.pformat( self.parent.gathered_output ) )
            outfile.close()
        return





