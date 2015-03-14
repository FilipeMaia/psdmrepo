import event_process
import pprint
import logging

class add_available_data(event_process.event_process):
    def __init__(self):
        self.output = event_process.event_process_output()
        self.reducer_rank = 0
        self.event_keys = []
        self.config_keys = []
        self.all_events_keys = {}
        self.logger = logging.getLogger(__name__+'.add_available_data')
        return

    def event(self,evt):
        for kk in evt.keys():
            if str(kk) not in self.all_events_keys:
                self.all_events_keys[str(kk)] = 0
            else:
                self.all_events_keys[str(kk)] += 1
        if len(self.event_keys) == 0:
            self.event_keys = evt.keys()
            self.parent.shared['event_keys'] = self.event_keys
        if len(self.config_keys) == 0:
            self.config_keys = self.parent.ds.env().configStore().keys()
            self.parent.shared['config_keys'] = self.config_keys
        return

    def endJob(self):
        self.gathered_all_events_keys = self.parent.comm.gather( self.all_events_keys, root=self.reducer_rank )
        if self.parent.rank == self.reducer_rank:
            self.output['in_report'] =  'meta'
            self.output['in_report_title'] = 'Available Data Sources'
            self.output['text'].append('Event Keys:<br/>\n<pre>' + pprint.pformat( self.event_keys )  + '</pre>')
            self.output['text'].append('Config Keys:<br/>\n<pre>' + pprint.pformat( self.config_keys )  + '</pre>')
            self.merged_all_events_keys = {}
            for gg in self.gathered_all_events_keys:
                for kk in gg:
                    if kk not in self.merged_all_events_keys:
                        self.merged_all_events_keys[kk] = gg[kk]
                    else :
                        self.merged_all_events_keys[kk] += gg[kk]

            self.output['text'].append("Event Keys Counter:<br/>\n<pre>" + pprint.pformat( self.merged_all_events_keys ) + '</pre>')

            self.parent.output.append(self.output)
        return

