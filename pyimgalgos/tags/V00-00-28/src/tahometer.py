#------------------------------
"""User analysis module 

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import os
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------

from time import time

class tahometer (object) :
    """Saves image array in file with specified in the name type."""

    def __init__ ( self ) :
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.

        @param dn  int, interval in number of events to print current statistics
        @param print_bits  int, bit-word for verbosity control 
        """

        self.m_dn         = self.configInt  ('dn', 100)
        self.m_print_bits = self.configInt  ('print_bits', 1)

        if self.m_print_bits & 1 : self.print_input_pars()

        self.counter    = 0
        self.counter_dn = 0

    def beginjob( self, evt, env ) :
        pass
        self.t_beginjob = time()
        self.t_dn       = self.t_beginjob
 
    def beginrun( self, evt, env ) :
        #logging.info( "tahometer.beginrun() called" )
        self.run = evt.run()


    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        self.counter    += 1
        self.counter_dn += 1

        if self.counter_dn == self.m_dn :
            if self.m_print_bits & 2 :
                self.print_event_record()
                self.counter_dn = 0
                self.t_dn = time()


    def endjob( self, evt, env ) :
        pass
        self.print_event_record()


    def print_input_pars( self ) :
        msg = '\n%s: List of input parameters\n  dn %6d\n  print_bits: %4d' % \
              (__name__, self.m_dn, self.m_print_bits)
        #logging.info( __name__+ ': ' + msg )
        print msg


    def print_event_record( self ) :

        t_curr = time()
        dt_sec = t_curr - self.t_dn
        t_sec  = t_curr - self.t_beginjob
        rate   = 0
        drate  = 0
        if  t_sec != 0 :  rate = self.counter/t_sec
        if dt_sec != 0 : drate = self.counter_dn/dt_sec

        rec = '%s: run:%04d  evt:%06d  t[sec]:%10.3f  dt[sec]:%10.3f  n/t[1/sec]:%10.3f  dn/dt[1/sec]:%10.3f' % \
              (__name__, self.run, self.counter, t_sec, dt_sec, rate, drate)
        #logging.info( rec )
        print rec

#-----------------------------
#-----------------------------
#-----------------------------
