#!/usr/bin/env python

# --------------------------------------------------------------
# This script will obtain values of PVs representing a status of
# the LCLS X-Ray beam and put them into a database using an
# external application.
# --------------------------------------------------------------

import os
import sys
import time
import subprocess

import pyca
import Pv

ctrl = False
connect_timeout_sec   =  1.0
get_timeout_sec       =  1.0
sampling_interval_sec =  2.0
force_interval_sec    = 10.0
num_hearbeats2keep    =  1

def get_pv(pvname):
    """
    Fetch and return a value of the specified EPICS variable if it's
    available. Return None otherwise. Some operations invoked by
    this function may through exceptions. Please, take care of them
    in a calling code.
    """
    value = None
    pv = Pv.Pv(pvname)
    pv.connect(connect_timeout_sec)
    pv.get(ctrl,get_timeout_sec)
    if pv.status == pyca.NO_ALARM:
        value = pv.value
    pv.disconnect()
    return value

def store_pv(pvname,value,force=False,num2keep=None):
    cmd = ['python',"%s/LCLSBeamTimeStore.py" % os.path.dirname(__file__),pvname,str(value)]
    if force: cmd.append('-force')
    if num2keep: cmd.extend(['-keep',str(num2keep)])
    subprocess.call(cmd)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        sampling_interval_sec = float(sys.argv[1])
        if sampling_interval_sec < 1.0:
            print "error: sampling interval should be longer than 1 second"
            sys.exit(1)

        if len(sys.argv) > 2:
            force_interval_sec = float(sys.argv[2])
            if force_interval_sec < sampling_interval_sec:
                print "error: force interval should be longer than the sampling one"
                sys.exit(1)

    pvnames = ['LIGHT:LCLS:STATE','XRAY_DESTINATIONS']

    since_last_force_sec = 0.0

    while True:

        force = False
        if since_last_force_sec > force_interval_sec:
            force = True
            since_last_force_sec = 0.0

        try:
            for pvname in pvnames:
                value = get_pv(pvname)
                if value is not None:
                    store_pv(pvname,value)

            store_pv('HEARTBEAT'," ".join(pvnames),force,num_hearbeats2keep)

        except pyca.pyexc, e:
            print 'pyca exception:', e
        except pyca.caexc, e:
            print 'channel access exception:', e
        except Exception, e:
            print 'Exception:', e

        time.sleep(sampling_interval_sec)

        since_last_force_sec = since_last_force_sec + sampling_interval_sec

    sys.exit(0)
