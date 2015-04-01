#!/usr/bin/env python

import pytopsana
import sys
from psana import *

ds  = DataSource('exp=amoc6914:run=225')
evt = ds.events().next()
env = ds.env()
src = Source('DetInfo(Camp.0:pnCCD.0)')
det = pytopsana.Detector() # src)

#print evt.keys()

peds = det.pedestals(src,evt,env)
print peds[0,0,1:20]

#raw_data = detector.raw(src,evt,env)
#print raw_data
#print detector.calib(src,evt,env)
#print detector.calib(raw_data)
#print detector.env(env)

sys.exit(0)
