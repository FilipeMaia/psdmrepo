#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module AppBase
#------------------------------------------------------------------------

""" This is a base class for the "application" objects. It encapsulates two 
tasks - command line parsing and logger instantiation/customization. Adopted
from BaBar BbrPyapp module.

This software was developed for the LCLS project.  If you
use all or part of it, please give an appropriate acknowledgement.

Copyright (C) 2006 SLAC

@version $Id$ 

@author Andy Salnikov
"""

#------------------------------
#  Interpreter version check --
#------------------------------

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
from optparse import OptionParser
import logging
import sys
import os

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

LOGGING_TRACE_LEVEL = 15

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class AppBase :

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, installLogger = True, 
                   usage = None, version = None, description = None, prog = None,
                   logfmt = '%(asctime)s %(levelname)-8s %(message)s' ) :
        """Constructor.

        @param installLogger  will install root logger if true
        @param usage          usage string for OptionParser, e.g. usage: %prog [options] arg1
        @param logfmt         logger format string
        """

        # define instance variables
        self._installLogger = installLogger
        self._parser = OptionParser ( usage = usage, version = version, description = description, prog = prog )
        self._parser.add_option ( '-v', "--verbose", dest="verbose", action="count", default=0, help="produce more noise" )
        self._log = None
        self._logfmt = logfmt

    #-------------------
    #  Public methods --
    #-------------------

    #
    #  called from mine, customize the app and run it
    #
    def run ( self, argv = sys.argv ) :
        """Customize and run the application

        @param argv   application arguments from the command line, usually sys.argv
        @return       return value, pass this to the shell
        """

        self._options, self._args = self._parser.parse_args(argv[1:])

        appName = self.appName()

        # setup logger
        if self._installLogger :
            logging.addLevelName ( LOGGING_TRACE_LEVEL, 'TRACE' )
            rootlog = logging.getLogger()
            hdlr = logging.StreamHandler( sys.stdout )
            formatter = logging.Formatter(self._logfmt)
            hdlr.setFormatter(formatter)
            rootlog.addHandler(hdlr)
            logLevels = { 0 : logging.WARNING, 1 : logging.INFO, 2 : LOGGING_TRACE_LEVEL }
            rootlog.setLevel( logLevels.get (self._options.verbose,logging.DEBUG) )
            self._log = logging.getLogger(appName)
        else :
            self._log = logging.getLogger()

        return self._run()

    #
    #  Logging methods
    # 
    def debug ( self, msg, *args, **kwargs ) :
        return self._log.log ( logging.DEBUG, msg, *args, **kwargs )
    def trace ( self, msg, *args, **kwargs ) :
        return self._log.log ( LOGGING_TRACE_LEVEL, msg, *args, **kwargs )
    def info ( self, msg, *args, **kwargs ) :
        return self._log.log ( logging.INFO, msg, *args, **kwargs )
    def warning ( self, msg, *args, **kwargs ) :
        return self._log.log ( logging.WARNING, msg, *args, **kwargs )
    def error ( self, msg, *args, **kwargs ) :
        return self._log.log ( logging.ERROR, msg, *args, **kwargs )
    def critical ( self, msg, *args, **kwargs ) :
        return self._log.log ( logging.CRITICAL, msg, *args, **kwargs )


    #
    #  get the application name, subclasses may override
    #
    def appName ( self ) :
        return os.path.basename(sys.argv[0])

    #
    #  Subclasses should override this method
    #
    def _run ( self ) :
        """Run the application

        @return       return value, pass this to the shell
        """
        raise NotImplementedError ( "AppBase: method _run() should be implemented in a subclass" )



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
