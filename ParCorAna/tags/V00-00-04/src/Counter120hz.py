class Counter120hz(object):
    '''make counter

    '''
    MAXFIDUCIALS = (1<<17)-32
    FIDCYCLESECONDS = int(MAXFIDUCIALS/360.0)
    def __init__(self, sec0, nsec0, fid0):
        assert fid0 >=0 and fid0 <= Counter120hz.MAXFIDUCIALS, "fid0=%s not in [%d,%d]" % (fid0, 0, Counter120hz.MAXFIDUCIALS)
        assert nsec0 >=0 and nsec0 <= 1e9
        fractionIntoCycle = fid0/float(Counter120hz.MAXFIDUCIALS)
        secondsIntoCycle = Counter120hz.FIDCYCLESECONDS*fractionIntoCycle
        self.approxFirstCycleStartTime = (sec0-secondsIntoCycle) + 1e-9 * nsec0
        self.firstFiducialCounter = fid0/3

    def getCounter(self, sec,fid):
        assert fid >=0 and fid <= Counter120hz.MAXFIDUCIALS, "fid=%s not in [%d,%d]" % (fid, 0, Counter120hz.MAXFIDUCIALS)
        counterInCurrentCycle = fid/3

        fractionIntoCurrentCycle = fid/float(Counter120hz.MAXFIDUCIALS)
        secondsIntoCurrentCycle = Counter120hz.FIDCYCLESECONDS*fractionIntoCurrentCycle
        approxSecondsStartCurrentCycle = sec - secondsIntoCurrentCycle
        secondsBetweenCycles = approxSecondsStartCurrentCycle - self.approxFirstCycleStartTime
        numberCyclesSinceStart = round(secondsBetweenCycles/float(Counter120hz.FIDCYCLESECONDS))
        counter = counterInCurrentCycle + (numberCyclesSinceStart * 120 * Counter120hz.FIDCYCLESECONDS)
        counter -= self.firstFiducialCounter
        return counter
        
        
        

