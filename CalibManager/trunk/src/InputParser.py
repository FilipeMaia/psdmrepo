#!/usr/bin/env python

#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIGrabSubmitELog...
#
#------------------------------------------------------------------------

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
#import pwd
#import tempfile
#from copy import deepcopy
#from time import localtime, strftime

#from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

from optparse import OptionParser

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

def input_options_parser() :

    parser = OptionParser(description='Optional input parameters.', usage ='usage: %prog [options] args')
    parser.add_option('-i', '--ins', dest='inssta', default=None,   action='store', type='string', help='the name of an instrument and station <INS>[:<station-number>]')
    parser.add_option('-e', '--exp', dest='exp',    default=None,   action='store', type='string', help='the name of some specific experiment')
    parser.add_option('-w', '--url', dest='url',    default=None,   action='store', type='string', help='the base URL of the LogBook web service')
    parser.add_option('-c', '--cmd', dest='cmd',    default=None,   action='store', type='string', help='the command for child message')
    parser.add_option('-f', '--cfg', dest='cfname', default=None,   action='store', type='string', help='the file name with configuration parameters')
    parser.add_option('-u', '--usr', dest='usr',    default=None,   action='store', type='string', help='the user name to connect to the web service')
    parser.add_option('-p', '--pas', dest='pas',    default='pcds', action='store', type='string', help='the password to connect to the web service')
    #parser.add_option('-v', '--verbose',      dest='verb',    default=True, action='store_true',           help='allows print on console')
    #parser.add_option('-q', '--quiet',        dest='verb',                  action='store_false',          help='supress print on console')

    (opts, args) = parser.parse_args()
    return (opts, args)

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

#---------------------------------
# python /reg/g/pcds/pds/grabber/bin/grelog.py -i AMO -e amodaq09 -u amoopr -p pcds -w https://pswww.slac.stanford.edu/ws-auth
# grelog.py -i AMO -e amodaq09 -u amoopr -w https://pswww.slac.stanford.edu/ws-auth
#---------------------------------
from LogBookWebService import LogBookWebService

def run_GUIGrabSubmitELog() :

    (opts, args) = input_options_parser()
    #print 'opts:\n', opts
    #print 'args:\n', args

    #-----------------------------
    print 'File name for I/O configuration parameters:', str(opts.cfname)

    sta = ''
    pos = opts.inssta.rfind(':')
    if pos==-1 : ins = opts.inssta
    else :
        ins = opts.inssta[:pos]
        if len(opts.inssta[pos:]) > 1 : sta = opts.inssta[pos+1:]

    #-----------------------------
    pars = { 'ins'    : ins, 
             'sta'    : sta, 
             'exp'    : opts.exp, 
             'url'    : opts.url, 
             'usr'    : opts.usr, 
             'pas'    : opts.pas,
             'cmd'    : opts.cmd
           }

    print 'Start grabber for ELog with input parameters:'
    for k,v in pars.items():
        if k is not 'pas' : print '%9s : %s' % (k,v)

    lbws = LogBookWebService(**pars)
    #lbws.print_experiments()

    #-----------------------------
    pars2 ={ 'ins'    : 'NEH', 
             'sta'    : sta, 
             'exp'    : ins + ' Instrument', 
             'url'    : opts.url, 
             'usr'    : opts.usr, 
             'pas'    : opts.pas,
             'cmd'    : opts.cmd
           }

    print 'Open web service for copy messages in the instrumental ELog:'
    for k,v in pars2.items():
        if k is not 'pas' : print '%9s : %s' % (k,v)

    lbws2 = LogBookWebService(**pars2)
    #lbws2.print_experiments()

    #-----------------------------
    app = QtGui.QApplication(sys.argv)
    #w  = GUIGrabSubmitELog.py(**pars)
    w = GUIGrabSubmitELog(cfname=opts.cfname, lbws=lbws, lbws2=lbws2)
    w.show()
    app.exec_()

    #del QtGui.qApp
    #QtGui.qApp=None

    #app.closeAllWindows()
    #QtGui.qApp=None
    print 'Exit application...'
    
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

if __name__ == "__main__" :

    run_GUIGrabSubmitELog()

    #test_test_Logger()
    #test_ConfigParametersForApp()
    #test_GUILogger() 
    #test_GUIImage()
    #test_GUIGrabSubmitELog()

    sys.exit (0)

    #try: sys.exit (0)
    #except SystemExit as err :
    #    print 'Xo-xo'

#-----------------------------
#-----------------------------
