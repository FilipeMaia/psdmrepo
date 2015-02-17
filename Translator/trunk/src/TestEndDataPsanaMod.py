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
        self.putarray = np.array([0,1,2], np.float)
        self.putstr = "configuration: threshold=5.2"
        self.verbose = self.configBool('verbose',True)
        self.addedToCalib = False

    def beginjob(self, evt, env):
        if self.verbose: print "TestEndDataPsanaMod: beginjob"
        env.configStore().put(self.putstr, self.beginjob_str_cfgstore)
        evt.put(self.putstr, self.beginjob_str_event)
        env.configStore().put(self.putarray, self.beginjob_ndarray_cfgstore)
        evt.put(self.putarray, self.beginjob_ndarray_event)
        
    def beginrun(self, evt, env):
        if self.verbose: print "TestEndDataPsanaMod: beginrun"
        env.configStore().put(self.putstr, self.beginrun_str_cfgstore)
        evt.put(self.putstr, self.beginrun_str_event)
        env.configStore().put(self.putarray, self.beginrun_ndarray_cfgstore)
        evt.put(self.putarray, self.beginrun_ndarray_event)
        
    def begincalibcycle(self, evt, env):
        if self.verbose: print "TestEndDataPsanaMod: begincalibcycle"

        # right now, Python can't replace things that C++ sees in the event store, so we'll 
        # have each key for the config store include the calib number
        evt.put(self.putstr, self.begincalibcycle_str_event)
        evt.put(self.putarray, self.begincalibcycle_ndarray_event)
        if not self.addedToCalib:
            env.configStore().put(self.putstr, self.begincalibcycle_str_cfgstore)
            env.configStore().put(self.putarray, self.begincalibcycle_ndarray_cfgstore)

    def event(self,evt,env):
        if self.verbose: print "TestEndDataPsanaMod: event"

    def endcalibcycle(self, evt, env):
        if self.verbose: print "Translator.TestEndDataPsanaMod: end calib cycle"
        evt.put(self.putstr, self.endcalibcycle_str_event)
        evt.put(self.putarray, self.endcalibcycle_ndarray_event)
        if not self.addedToCalib:
            env.configStore().put(self.putstr, self.endcalibcycle_str_cfgstore)
            env.configStore().put(self.putarray, self.endcalibcycle_ndarray_cfgstore)
            self.addedToCalib = True

    def endrun(self, evt, env):
        if self.verbose: print "TestEndDataPsanaMod: end run"
        env.configStore().put(self.putstr, self.endrun_str_cfgstore)
        evt.put(self.putstr, self.endrun_str_event)
        env.configStore().put(self.putarray, self.endrun_ndarray_cfgstore)
        evt.put(self.putarray, self.endrun_ndarray_event)

    def endjob(self, evt, env):
        if self.verbose: print "TestEndDataPsanaMod: end job"
        env.configStore().put(self.putstr, self.endjob_str_cfgstore)
        evt.put(self.putstr, self.endjob_str_event)
        env.configStore().put(self.putarray, self.endjob_ndarray_cfgstore)
        evt.put(self.putarray, self.endjob_ndarray_event)
