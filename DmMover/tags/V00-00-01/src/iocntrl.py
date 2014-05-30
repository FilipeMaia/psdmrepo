#!/usr/bin/env python 

import sys
import select
import time
import threading

class StdinCtrl(threading.Thread):

    def __init__(self, name=""):

        threading.Thread.__init__(self)
        self.status = 'run'
        self.stop = threading.Event()
        self.wait = threading.Event()

        self.cmdname = name

    def stop(self):
        return self.status == 'stop'

    def run(self):

        print "Starting stdin reader thread"

        while True:
            al,bl,cl = select.select([sys.stdin], [], [])        
            if len(al) <= 0:
                return
            line = al[0].readline().rstrip()
            #print "INPUT", line.rstrip()
            if line.startswith('wait'):
                self.wait.set()
            elif line.startswith('cont'):
                self.wait.clear()
            if line.startswith('help'):
                self.showhelp()
            if line.startswith('stop'):
                self.stop.set()
                print "stop from thread"
                return
            elif line.startswith('stat'):
                print "status:", self.status
            time.sleep(0.2)

    def showhelp(self):
        print " HELP: Commands to talk to", self.cmdname
        print "       Cmd: stop help"

    def stopnow(self):

        return self.stop.isSet()

    def wait_alittle(self):
        return self.wait.isSet()
    

if __name__ == "__main__":
    ctrl = StdinCtrl(name="test ctrl")
    ctrl.start()


    for i in xrange(300):

        print "next", i, ctrl.wait_alittle()
        time.sleep(3)


        if ctrl.stopnow():
            print "wait"
            break
    


    
