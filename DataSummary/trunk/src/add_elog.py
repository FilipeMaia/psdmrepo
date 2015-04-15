import logging
import event_process

class add_elog(event_process.event_process):
    def __init__(self):
        self.output = event_process.event_process_output()
        self.reducer_rank = 0
        self.logger = logging.getLogger(__name__+'.add_available_data')
        return
    
    def endJob(self):
        if self.parent.rank == self.reducer_rank:
            self.expNum = self.parent.ds.env().expNum()
            self.output['in_report']= 'meta'
            self.output['in_report_title'] = 'Elog'
            self.output['text'] = [     "<a href='https://pswww.slac.stanford.edu/apps/portal/index.php?exper_id={:0.0f}'>Elog</a>".format(self.expNum),]
            self.output['text'].append( "<a href='https://pswww.slac.stanford.edu/apps/portal/index.php?exper_id={:0.0f}&app=elog:search&params=run:{:0.0f}'>Elog Run</a> (broken)".format(self.expNum, self.parent.run))
            self.parent.output.append(self.output)
        return

