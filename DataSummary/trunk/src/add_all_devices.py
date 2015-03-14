import event_process
import logging
from simple_stats import simple_stats
from simple_trends import simple_trends
from acqiris      import acqiris
from epics_trend  import epics_trend
from ipimb        import ipimb
from cspad        import cspad


class add_all_devices(event_process.event_process):
    def __init__(self,devs):
        self.devs = devs
        self.logger = logging.getLogger(__name__+'.add_all_devices')
        self.done = False
        self.done_counter = 0
        self.done_counter_max = 100
        self.inserted = []
        self.newsubjobs = []

    def setup_subjob(self,alias,nsj,evt):
        if alias == '':
            self.logger.error('alias is empty!')
        self.logger.info('setting up new subjob')
        self.inserted.append( alias )
        self.newsubjobs.append(nsj)
        self.parent.subjobs.insert( self.myindex, nsj )
        ranks = range(self.parent.size)
        nsj.set_parent(self.parent)
        nsj.logger = logging.getLogger( self.parent.logger.name + '.' + nsj.logger.name.split('.')[-1] )
        #nsj.reducer_rank = ranks[ len(self.inserted) % len(ranks) ]
        nsj.reducer_rank = 0
        nsj.beginJob()
        nsj.beginRun()
        nsj.event(evt)

    def add_device(self,kk,evt):
        alias = kk.alias()
        if alias == '':
            src = str(kk.src())
            self.logger.debug('alias is empty, falling back to src: "{:}"'.format(src))
            if src == 'BldInfo(FEEGasDetEnergy)':
                alias = 'Gasdet'
            elif src == 'BldInfo(EBeam)':
                alias = 'EBeam'
            self.logger.debug('setting alias to {:}'.format(alias))
        if (alias in self.devs) and (alias not in self.inserted):
            self.logger.debug(alias)
            if 'summary_report' in self.devs[alias] :
                self.logger.info('adding {:} to event processing'.format(alias))
                self.logger.info(self.devs[alias]['summary_report'])
                if alias=='Acqiris':
                    thisjob = acqiris()
                    thisjob.set_stuff(kk.src(),kk.type(),*self.devs[alias]['summary_report']['set_stuff']['args'],**self.devs[alias]['summary_report']['set_stuff']['kwargs'])
                    self.setup_subjob(alias,thisjob,evt)
                elif 'Ipimb' in alias:
                    self.logger.info('setting up Ipimb')
                    if 'epics_trend' in self.devs[alias]['summary_report']:
                        thisjob = epics_trend()
                        thisjob.add_pv_trend(self.devs[alias]['pvs']['targ']['base']+'.RBV')
                        thisjob.add_pv_trend(self.devs[alias]['pvs']['x']['base']+'.RBV')
                        thisjob.add_pv_trend(self.devs[alias]['pvs']['y']['base']+'.RBV')
                        thisjob.output['in_report'] = 'detectors'
                        thisjob.output['in_report_title'] = '{:} Epics Trends'.format(alias)
                        self.setup_subjob(alias,thisjob,evt)
                        thisjob = ipimb()
                        thisjob.set_stuff(kk.src(),kk.type(),*self.devs[alias]['summary_report']['ipimb']['args'],**self.devs[alias]['summary_report']['ipimb']['kwargs'])
                        thisjob.output['in_report'] = 'detectors'
                        thisjob.output['in_report_title'] = '{:} Diode Voltages'.format(alias)
                        self.setup_subjob(alias,thisjob,evt)
                elif 'CsPad' in alias:
                    self.logger.info('setting up CSPAD')
                    if 'summary_report' in self.devs[alias]:
                        thisjob = cspad()
                        thisjob.set_stuff(kk.src(),kk.type(),*self.devs[alias]['summary_report']['set_stuff']['args'], **self.devs[alias]['summary_report']['set_stuff']['kwargs'])
                        thisjob.output['in_report'] = 'analysis'
                        thisjob.output['in_report_title'] = '{:}'.format(alias)
                        self.setup_subjob(alias,thisjob,evt)
                else:
                    if 'simple_stats' in self.devs[alias]['summary_report'] and self.devs[alias]['summary_report']['simple_stats']:
                        self.logger.info('setting up simple stats')
                        thisjob = simple_stats()
                        thisjob.set_stuff(kk.src(),kk.type(),*self.devs[alias]['summary_report']['simple_stats']['args'],**self.devs[alias]['summary_report']['simple_stats']['kwargs'])
                        self.setup_subjob(alias,thisjob,evt)
                    if 'simple_trends' in self.devs[alias]['summary_report'] and self.devs[alias]['summary_report']['simple_trends']:
                        self.logger.info('setting up simple trends')
                        thisjob = simple_trends()
                        thisjob.set_stuff(kk.src(),kk.type(),*self.devs[alias]['summary_report']['simple_trends']['args'],**self.devs[alias]['summary_report']['simple_trends']['kwargs'])
                        self.setup_subjob(alias,thisjob,evt)
                # do something
        return


    def event(self,evt):
        if not self.done:
            self.myindex = self.parent.subjobs.index(self) + 1
            self.logger.debug('current subjob index is {:}'.format(self.myindex))
            for kk in evt.keys():
                self.add_device(kk,evt) 

            self.done_counter += 1

        if self.done_counter == self.done_counter_max:
            self.done = True
            self.logger.info('finished adding new subjobs')
            self.done_counter += 1


