#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module config...
#
#------------------------------------------------------------------------

"""Job configuration for pyana

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging
import ConfigParser 
from optparse import OptionParser, make_option

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------
logging.basicConfig()
_log = logging.getLogger("pyana.config")

# maps option names in config file to command-line options names
# and the conversion functions
_unity = lambda x : x

def _str2bool(x):
    if x.lower in ('false', '0', 'no') : return False
    if x.lower in ('true', '1', 'yes') : return False
    raise ValueError('invalid value for boolean conversion: %s' % x)

_options = {
    'verbose'    : ( 'verbose',    int ),
    'file-list'  : ( 'file_list',  _unity),
    'num-events' : ( 'num_events', int),
    'skip-events' : ( 'skip_events', int),
    'modules'    : ( 'module',     lambda x : x.split()),
    'job-name'   : ( 'job_name',   _unity),
    'num-cpu'    : ( 'num_cpu',    int),
    'dg-ref'     : ( 'dg_ref',     _str2bool),
}


_cmdoptions = [
    make_option( '-v', "--verbose", action="count", help="increase verbosity" ),
    make_option( '-c', "--config", metavar="FILE", help="configuration file" ),
    make_option( '-C', "--config-name", metavar="STRING", help="configuration name, def: empty" ),
    make_option( '-l', "--file-list", metavar="FILE", help="file with a list of file names in it" ),
    make_option( '-n', "--num-events", metavar="NUMBER", type="int", help="maximum number of events to process" ),
    make_option( '-s', "--skip-events", metavar="NUMBER", type="int", help="number of events to skip" ),
    make_option( '-j', "--job-name", metavar="STRING", help="job name, default is deduced from file name(s)" ),
    make_option( '-m', "--module", metavar="NAME", action="append", help="user module name, multiple modules allowed" ),
    make_option( '-p', "--num-cpu", metavar="NUMBER", type="int", help="number grater than 1 enables multi-processing" ),
    make_option( '-r', "--dg-ref", action="store_true",help="pass datagrams by-reference to children processes" ),
]

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class config ( object ) :

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, args=None ) :

        self._parser = OptionParser(usage="%prog [options] [xtc-files ...]", option_list=_cmdoptions)
    
        self._options, self._args = self._parser.parse_args(args)
        
        # set logging level
        log_levels = { None: logging.WARNING, 0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG }
        level = log_levels.get ( self._options.verbose, logging.DEBUG )
        logging.getLogger().setLevel(level)

        self._config = None
        self._sections = ["pyana"]
        if self._options.config_name :
            self._sections.insert(0, "pyana."+self._options.config_name)

        # read config file
        configFile = self._options.config
        if configFile :
            _log.info("reading config file %s", configFile)
            self._config = ConfigParser.ConfigParser()
            self._config.readfp(file(configFile))
        else :
            _log.info("reading config file pyana.cfg")
            self._config = ConfigParser.ConfigParser()
            self._config.read("pyana.cfg")

        # verify that config file options have valid names
        for section in self._config.sections() :
            _log.debug("config section=%s", section)
            if section == 'pyana' or section.startswith('pyana.') :
                for option in self._config.options(section):
                    _log.debug("config section=%s option=%s", section, option)
                    if option not in _options :
                        raise ValueError("unknown option name '%s' in section [%s]" % (option, section) )

        # update logging level
        level = log_levels.get ( self.getJobConfig('verbose', 0), logging.DEBUG )
        logging.getLogger().setLevel(level)


    #-------------------
    #  Public methods --
    #-------------------

    def files( self ):
        """ Returns the list of input file names """
        # get file names
        file_list = self.getJobConfig('file-list')
        if not self._args and not file_list :
            self._parser.error("at least one file name or a file list required")
        if self._args and file_list :
            self._parser.error("file list cannot be used with the file names")
        if file_list :
            # read file names from file
            names = file(file_list).readlines()
            names = [ n.strip() for n in names ]
            names = [ n for n in names if n ]
        else :
            names = self._args
        return names

    def getJobConfig( self, option, default=None ):
        """ Gets configuration option from one of many sections """
        
        opt = _options.get(option)
        if not opt : raise ValueError('unknown option name: %s' % option)
        
        # first try command line option
        cmdopt = opt[0]
        if cmdopt :
            value = getattr(self._options, cmdopt)
            if value is not None: return value

        # then from config file
        for section in self._sections:
            try:
                _log.debug("getJobConfig section=%s option=%s", section, option)
                strval = self._config.get(section, option)
                # convert it to correct type
                return opt[1](strval)
            except ConfigParser.NoSectionError:
                pass
            except ConfigParser.NoOptionError:
                pass
            _log.debug("getJobConfig option not found")
    
        # otherwise default
        return default
    
    def getModuleConfig( self, module ):
        """ Gets configuration options for a module as a dictionary """
        
        config = {}
        try :
            
            if ':' in module :
                # read options from the main module section
                try:
                    base = module.split(':')[0]
                    config.update( self._config.items(base) )
                except ConfigParser.NoSectionError:
                    pass
            config.update( self._config.items(module) )
        except ConfigParser.NoSectionError:
            pass
        return config
    

    #--------------------------------
    #  Static/class public methods --
    #--------------------------------

    #--------------------
    #  Private methods --
    #--------------------


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
