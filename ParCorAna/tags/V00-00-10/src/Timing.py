import time

###############################
# timing

enableTiming = True
timingDictInsertOrder=[]  # list of names in order inserted
timingDict = {}  # entries will be 'foo':[totalTimeSec, numberOfCalls, callDescription]
                 # such as 'fooRamp' and 'fooSat', use two functions to time calls in
                 # different regions

class timecall(object):
    def __init__(self, counterDescription, secInUnit=1e-3, avgReportUnits='ms', 
             timingDict=timingDict, timingDictInsertOrder=timingDictInsertOrder):
        self.counterDescription = counterDescription
        self.secInUnit = secInUnit
        self.avgReportUnits = avgReportUnits
        self.timingDict = timingDict
        self.timingDictInsertOrder = timingDictInsertOrder

    def __call__(self, f):
        def null_wrap_f(*args, **kwargs):
            return f(*args, **kwargs)

        if not enableTiming:
            return null_wrap_f

        funcName = f.__name__
        if funcName not in self.timingDict:
            self.timingDict[funcName]=[0.0, 0, self.counterDescription,
                                       self.secInUnit, self.avgReportUnits]
            self.timingDictInsertOrder.append(funcName)
        def time_wrap_f(*args, **kwargs):
            t0 = time.time()
            res = f(*args, **kwargs)
            self.timingDict[funcName][0] += time.time()-t0
            self.timingDict[funcName][1] += 1
            return res

        return time_wrap_f

def reportOnTimingDict(logger, hdr, footer, timingDict=timingDict, keyOrder=timingDictInsertOrder):
    msg = '\n%s\n' % hdr
    keyWidth = max([len(k) for k in timingDict.keys()])
    for key in keyOrder:
        totalSec, n, nDescr, secToUnit, unitsText = timingDict[key]
        if n <= 0: continue
        avgPerCall = totalSec/float(secToUnit*n)
        msg += "%s: %8.3f%s per %s (%d total calls)\n" % (key, avgPerCall, unitsText, nDescr, n)
    msg += footer
    logger.info(msg)
