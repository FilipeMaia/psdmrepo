#!/usr/bin/env python

import os
import sys
from pyimgalgos.EventViewer import EventViewer

import psana

#------------------------------
import CalibManager.AppDataPath as apputils
#------------------------------

def do_work() :

    path_psana_cfg = apputils.AppDataPath('pyimgalgos/scripts/psana-cspad-ds1-image-producer.cfg').path()
    print 'Path to psana cfg file: %s' % path_psana_cfg
    psana.setConfigFile(path_psana_cfg)

    ds = psana.DataSource('exp=cxii9415:run=33:idx')
    run = ds.runs().next()
    src = psana.Source('DetInfo(CxiDs1.0:Cspad.0)')
    key ='cspad_img'

    list_of_times = run.times()
    #list_of_times = list_of_times_selected(run)
    EventViewer(run, list_of_times, src, key)
    
#------------------------------

if __name__ == '__main__' :
    proc_name = os.path.basename(sys.argv[0])

    do_work()

    sys.exit('The End')

#------------------------------
