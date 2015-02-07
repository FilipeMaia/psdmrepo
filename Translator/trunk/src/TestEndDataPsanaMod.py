import numpy as np

class TestEndDataPsanaMod(object):
    def __init__(self):
        for be in ['begin','end']:
            for trans in ['job','run','calibcycle']:
                for tp in ['str','ndarray']:
                    for loc in ['cfgstore','event']:
                        param = be + trans + '_' + tp + '_' + loc
                        paramValue = self.configStr(param,param)
                        setattr(self,param, paramValue)
        self.putarray = np.zeros(3)
        self.putstr = "configuration: threshold=5.2"

    def beginjob(self, evt, env):
        print "beginjob"
#        env.configStore().put(self.putstr, self.beginjob_str_cfgstore)
#        evt.put(self.putstr, self.beginjob_str_event)
#        env.configStore().put(self.putarray, self.beginjob_ndarray_cfgstore)
#        evt.put(self.putarray, self.beginjob_ndarray_event)
        
    def beginrun(self, evt, env):
        print "beginrun"
#        env.configStore().put(self.putstr, self.beginrun_str_cfgstore)
#        evt.put(self.putstr, self.beginrun_str_event)
#        env.configStore().put(self.putarray, self.beginrun_ndarray_cfgstore)
#        evt.put(self.putarray, self.beginrun_ndarray_event)
        
    def begincalibcycle(self, evt, env):
        print "begincalibcycle"
#        env.configStore().put(self.putstr, self.begincalibcycle_str_cfgstore)
        evt.put(self.putstr, self.begincalibcycle_str_event)
        env.configStore().put(self.putarray, self.begincalibcycle_ndarray_cfgstore)
        evt.put(self.putarray, self.begincalibcycle_ndarray_event)

    def event(self,evt,env):
        print "event"

    def endcalibcycle(self, evt, env):
        print "end calib cycle"
#        env.configStore().put(self.putstr, self.endcalibcycle_str_cfgstore)
        evt.put(self.putstr, self.endcalibcycle_str_event)
        env.configStore().put(self.putarray, self.endcalibcycle_ndarray_cfgstore)
        evt.put(self.putarray, self.endcalibcycle_ndarray_event)

    def endrun(self, evt, env):
        print "end run"
#        env.configStore().put(self.putstr, self.endrun_str_cfgstore)
#        evt.put(self.putstr, self.endrun_str_event)
#        env.configStore().put(self.putarray, self.endrun_ndarray_cfgstore)
#        evt.put(self.putarray, self.endrun_ndarray_event)

    def endjob(self, evt, env):
        print "end job"
#        env.configStore().put(self.putstr, self.endjob_str_cfgstore)
#        evt.put(self.putstr, self.endjob_str_event)
#        env.configStore().put(self.putarray, self.endjob_ndarray_cfgstore)
#        evt.put(self.putarray, self.endjob_ndarray_event)
