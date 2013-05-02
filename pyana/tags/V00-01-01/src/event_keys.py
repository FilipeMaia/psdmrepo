#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module event_keys...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andy Salnikov
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

def _printConfigKeys(env):
    print "Config keys:"
    for item in env.configKeys():
        print "    %s" % (item,)

def _printEventKey(evt):
    print "Event keys:"
    for item in evt.keys():
        print "    %s" % (item,)

#---------------------
#  Class definition --
#---------------------
class event_keys (object) :
    """Pyana module which prints event keys. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        pass

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        logging.info( "event_keys.beginjob() called" )
        _printConfigKeys(env)
        _printEventKey(evt)

    def beginrun( self, evt, env ) :
        logging.info( "event_keys.beginrun() called" )
        _printConfigKeys(env)
        _printEventKey(evt)

    def begincalibcycle( self, evt, env ) :
        logging.info( "event_keys.begincalibcycle() called" )
        _printConfigKeys(env)
        _printEventKey(evt)

    def event( self, evt, env ) :
        logging.info( "event_keys.event() called" )
        _printEventKey(evt)

    def endcalibcycle( self, evt, env ) :
        logging.info( "event_keys.event() called" )
        _printEventKey(evt)

    def endrun( self, evt, env ) :
        logging.info( "event_keys.endrun() called" )
        _printEventKey(evt)

    def endjob( self, evt, env ) :
        logging.info( "event_keys.endjob() called" )
        _printEventKey(evt)
