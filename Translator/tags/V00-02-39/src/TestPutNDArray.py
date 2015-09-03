import psana
import numpy as np

class TestPutNDArray(object):
    def __init__(self):        
        self.src = self.configSrc('cspadsrc')
        self.cc=-1
        self.allevts = -1

    def begincalibcycle(self, evt, env):
        self.cc += 1
        self.ii=-1
        
    def event(self, evt, env):
        self.ii += 1
        self.allevts += 1
        cspad = evt.get(psana.CsPad.DataV2, self.src)
        if cspad is None: 
            print "Translator.TestPutNDArray: cc=%5d evt=%5d allevt=%7d: NO CSPAD" % (self.cc, self.ii, self.allevts)
            return
        arr = np.ones((2,3),np.float32)
        evt.put(arr,self.src, "myarray")
        print "Translator.TestPutNDArray: cc=%5d evt=%5d allevt=%7d" % (self.cc, self.ii, self.allevts)
