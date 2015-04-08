#!/usr/bin/env python

import sys
from psana import *
import pytopsana


ds  = DataSource('exp=amoc6914:run=225')
evt = ds.events().next()
env = ds.env()

src = Source('DetInfo(Camp.0:pnCCD.1)')
#src = Source('Camp.0:pnCCD.1')

det = pytopsana.Detector(src,0) # , 0xffff)
# src)

#print evt.keys()

peds = det.pedestals(evt,env)
print '\npedestals:\n', peds[0:20]

prms = det.pixel_rms(evt,env)
print '\npixel_rms:\n', prms[0:20]

pgain = det.pixel_gain(evt,env)
print '\npixel_gain:\n', pgain[0:20]

pmask = det.pixel_mask(evt,env)
print '\npixel_mask:\n', pmask[0:20]

pbkgd = det.pixel_bkgd(evt,env)
print '\npixel_bkgd:\n', pbkgd[0:20]

pstat = det.pixel_status(evt,env)
print '\npixel_status:\n', pstat[0:20]

pcmod = det.common_mode(evt,env)
print '\ncommon_mode:\n', pcmod

print '\nInstrument: ', det.inst(env)

#raw_data = det.raw(src,evt,env)
#print raw_data
#print det.calib(src,evt,env)
#print det.calib(raw_data)

sys.exit(0)
