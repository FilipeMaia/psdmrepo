
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ThreadCheckNewXtcFiles
#
#------------------------------------------------------------------------

"""ThreadCheckNewXtcFiles"""

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
from FileNameManager        import fnm

#---------------------
#  Class definition --
#---------------------

class ThreadCheckNewXtcFiles (QtCore.QThread) :

    def __init__ ( self, parent=None, dt_sec=60, print_bits=0 ) :
        QtCore.QThread.__init__(self, parent)

        self.dt_sec     = dt_sec
        self.print_bits = print_bits
        self.thread_id  = random.random()

        self.exp_name   = cp.exp_name.value()
        self.counter = 0
        self.list_of_runs_old = []

        cp.thread_check_new_xtc_files = self
        self.connect( self, QtCore.SIGNAL('update(QString)'), self.testConnection )


    def testConnection(self, text) :
        print 'ThreadCheckNewXtcFiles: Signal is recieved ' + str(text)


    def run( self ) :
        while True :
            self.counter += 1
            if self.print_bits & 2 : print 'ThreadCheckNewXtcFiles id: %f, counter: %d' % (self.thread_id, self.counter)
            if self.newXtcFileIsAvailable() :
                self.emitSignalNewXtc()
            self.sleep(self.dt_sec)


    def newXtcFileIsAvailable( self ) :
        list_of_runs = fnm.get_list_of_xtc_run_nums()
        if self.print_bits & 4 : print 'list_of_runs    : %s' % list_of_runs
        if self.print_bits & 8 : print 'list_of_runs_old: %s' % self.list_of_runs_old

        # If the experiment name has changed, it does not mean that new xtc is availble...
        if cp.exp_name.value() != self.exp_name :
            self.exp_name = cp.exp_name.value()
            self.list_of_runs_old = list(list_of_runs)            
            return False

        if list_of_runs != [] :
            if self.list_of_runs_old != []:
                if list_of_runs[-1] != self.list_of_runs_old[-1] :
                     self.list_of_runs_old = list(list_of_runs)
                     return True
            else :
                self.list_of_runs_old = list(list_of_runs)

        return False


    def emitSignalNewXtc( self ) :
        msg = 'thread_id:%f counter:%d last_run:%d' %(self.thread_id, self.counter, self.list_of_runs_old[-1])       
        self.emit( QtCore.SIGNAL('update(QString)'), msg )
        if self.print_bits & 1 : print 'New xtc file is available, msg: %s' % msg

#---------------------
