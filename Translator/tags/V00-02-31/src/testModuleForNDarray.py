import psana
import numpy as np

def makeArray(data,shape):
    X = np.array(data,dtype=np.float)
    X = X.reshape(shape)
    return X

class testModuleForNDarray(object):
    def __init__(self):
        pass

    def readConfigParameters(self):
        self.add_to_calib_shape = self.configListInt("add_to_calib_shape")
        self.add_to_calib_data = self.configListFloat("add_to_calib_data")
        self.add_to_calib_src = self.configStr("add_to_calib_src","")
        
        if len(self.add_to_calib_shape) == 0:
            self.add_to_calib_shape = (2,2)
        if len(self.add_to_calib_data) == 0:
            self.add_to_calib_data = [0,1,2,3]
        
        self.add_to_event_src = self.configSrc("add_to_event_src","")
        self.add_to_event_shape = self.configListInt("add_to_event_shape")
        self.add_to_event_data = self.configListFloat("add_to_event_data")
        if len(self.add_to_event_shape) == 0:
            self.add_to_event_shape = (2,2)
        if len(self.add_to_event_data) == 0:
            self.add_to_event_data = [0,1,2,3]
        self.add_to_event_key = self.configStr("add_to_event_key","array")
        
    def checkConfigParameters(self):
        size = reduce(lambda x,y:x*y, self.add_to_calib_shape)
        datalen = len(self.add_to_calib_data)
        assert size == datalen , "length of add_to_calib_data (%d) != shape size (%d)" % \
            (datalen, size)
        size = reduce(lambda x,y:x*y, self.add_to_event_shape)
        datalen = len(self.add_to_event_data)
        assert size == datalen , "length of add_to_event_data (%d) != shape size (%d)" % \
            (datalen, size)
                      
        
    def beginjob(self, evt, env):
        self.readConfigParameters()
        X = makeArray(self.add_to_calib_data, self.add_to_calib_shape)
        if self.add_to_calib_src != '':
            env.calibStore().put(X, psana.Source(self.add_to_calib_src))
        print env.calibStore().keys()

    def event(self, evt, env):
        X = makeArray(self.add_to_event_data, self.add_to_event_shape)
        evt.put(X, self.add_to_event_src, self.add_to_event_key)

        
        
        
