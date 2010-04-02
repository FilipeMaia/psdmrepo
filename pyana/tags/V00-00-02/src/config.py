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

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------
_log = logging.getLogger("pyana.config")

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
    def __init__ ( self, configFile, jobName = None ) :

        self._config = None
        self._sections = ["pyana"]
        if jobName :
            self._sections.insert(0, "pyana."+jobName)


        # read config file
        if configFile :
            _log.info("reading config file %s", configFile)
            self._config = ConfigParser.ConfigParser()
            self._config.read(file(configFile))
        else :
            _log.info("reading config file pyana.cfg")
            self._config = ConfigParser.ConfigParser()
            self._config.read("pyana.cfg")
    

    #-------------------
    #  Public methods --
    #-------------------

    def getJobConfig( self, option ):
        """ Gets configuration option from one of many sections """
        
        for section in self._sections:
            try:
                #_log.debug("getJobConfig section=%s option=%s", section, option)
                return self._config.get(section, option)
            except ConfigParser.NoSectionError:
                pass
            except ConfigParser.NoOptionError:
                pass
            #_log.debug("getJobConfig option not found")
    
    def getModuleConfig( self, module ):
        """ Gets configuration options for a module as a dictionary """
        
        try :
            config = self._config.items(module)
            return dict(config)
        except ConfigParser.NoSectionError:
            return {}
    

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
