#!/usr/bin/env python

import os
import sys
from pyimgalgos.EventViewer import EventViewer

import psana

#------------------------------

class EventViewerApp (EventViewer) :
    def __init__(self, run, list_of_times=None, src=None, key=None):
        EventViewer.__init__(self, run, list_of_times, src, key)

#------------------------------

    def get_img(self, evt, src=None, key=None) :
        """Interface call-back method can be re-implemented in sub-class
        """
        img = evt.get(psana.ndarray_float32_2, src, key)
        if img is not None : return img

#------------------------------

def list_of_times_selected(run):
    print 'List of selected times'

    lst_sel_t = []

    times = run.times()

    lst_fids =(0x19aa9 \
             , 0x19aca \
             , 0x19bd5 \
             , 0x19c0b \
             , 0x19cbc \
             , 0x19ce0 \
             , 0x19d3a \
             , 0x19ea8 \
             , 0x1a01c \
             , 0x1a172 \
             )
    
    for i,t in enumerate(times) :
        if t.fiducial() == lst_fids[-1] : break
    
        if t.fiducial() in lst_fids :
            print '%5d:  fid: %d  %d(sec)  %d(nsec)' % (i, t.fiducial(), t.seconds(), t.nanoseconds())

            lst_sel_t.append(t)

    return lst_sel_t

#------------------------------
#------------------------------
#------------------------------
import CalibManager.AppDataPath as apputils
#------------------------------

def do_work() :

    path_psana_cfg = apputils.AppDataPath('pyimgalgos/scripts/psana-cspad-ds2-image-producer.cfg').path()
    print 'Path to psana cfg file: %s' % path_psana_cfg
    psana.setConfigFile(path_psana_cfg)

    ds = psana.DataSource('exp=cxif5315:run=169:idx')
    run = ds.runs().next()
    src = psana.Source('DetInfo(CxiDs2.0:Cspad.0)')
    key ='cspad_img'

    #list_of_times = run.times()
    list_of_times = list_of_times_selected(run)
    EventViewerApp(run, list_of_times, src, key)
    
#------------------------------

if __name__ == '__main__' :
    proc_name = os.path.basename(sys.argv[0])

    do_work()

    sys.exit('The End')

#------------------------------
