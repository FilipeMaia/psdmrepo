import psana
import numpy as np

class gsc:
    # put the "run" in the interface, so that the user knows that a new
    # object should be made for every run (to get the correct configuration)
    def __init__(self,dataSource,run,source):
        self.source=source
        cstore = dataSource.env().configStore()
        cfg = cstore.get(psana.Gsc16ai.ConfigV1, source)
        vrange = cfg.voltageRange()
        self.format = cfg.dataFormat()
        if vrange==vrange.VoltageRange_10V:
            self.voltsMin = -10.0
            self.voltsPerCount = 20.0 / 0xffff
        elif vrange==vrange.VoltageRange_5V:
            self.voltsMin = -5.0;
            self.voltsPerCount = 10.0 / 0xffff
        elif vrange==vrange.VoltageRange_2_5V:
            self.voltsMin = -2.5;
            self.voltsPerCount = 5.0 / 0xffff
        else:
            print 'Error: gsc16ai data voltage range not recognized'
        
    def voltages(self, evt):
        data = evt.get(psana.Gsc16ai.DataV1, self.source)
        if self.format == self.format.DataFormat_TwosComplement:
            return self.voltsPerCount * data.channelValue().astype(np.int16);
        elif self.format == self.format.DataFormat_OffsetBinary:
            return self.voltsMin + self.voltsPerCount * data.channelValue();
        else:
            return None
