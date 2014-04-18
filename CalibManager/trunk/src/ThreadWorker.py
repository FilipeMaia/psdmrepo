
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ThreadWorker
#
#------------------------------------------------------------------------

"""ThreadWorker"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import random

from PyQt4 import QtGui, QtCore
from ConfigParametersForApp import confpars as cp

#---------------------
#  Class definition --
#---------------------

class ThreadWorker (QtCore.QThread) :

    def __init__ ( self, parent=None, dt_sec=5, print_bits=0 ) :
        QtCore.QThread.__init__(self, parent)        

        self.dt_sec     = dt_sec
        self.print_bits = print_bits
        self.thread_id  = random.random()

        cp.thread1 = self
        self.counter = 0

        #self.connect( self, QtCore.SIGNAL('update(QString)'), self.testConnection )


    def testConnection(self, text) :
        print 'ThreadWorker: Signal is recieved ' + str(text)


    def run( self ) :
        while True :
            self.counter += 1
            if self.print_bits & 2 : print '\nThreadWorker id, i:', self.thread_id, self.counter
            self.emitCheckStatusSignal()
            self.sleep(self.dt_sec)
            #time.sleep(1)


    def emitCheckStatusSignal( self ) :
        msg = 'from work thread ' + str(self.thread_id) + '  check counter: ' + str(self.counter)
        self.emit( QtCore.SIGNAL('update(QString)'), msg)

        if self.print_bits & 1 : print msg

        #self.emit( QtCore.SIGNAL('update(QString)'), \
        #           'from work thread ' + str(self.thread_id) +\
        #           '  check counter: ' + str(self.counter) )
        #print status_str

#---------------------
