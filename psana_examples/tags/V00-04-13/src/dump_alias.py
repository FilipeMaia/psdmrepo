#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Psana user analysis module dump_alias...
#
#------------------------------------------------------------------------

"""Psana module which dumps alias configuration.

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
from psana import *

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#---------------------
#  Class definition --
#---------------------
class dump_alias (object) :

    #----------------
    #  Constructor --
    #----------------
    def __init__(self):
        '''Class constructor. Does not need any parameters'''

        self.m_src = self.configSrc('source', 'ProcInfo()')

    #-------------------
    #  Public methods --
    #-------------------

    def beginRun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        config = env.configStore().get(Alias.Config, self.m_src)
        if config:
        
            print "%s: %s" % (config.__class__.__name__, self.m_src)
            print "  numSrcAlias =", config.numSrcAlias()
            for i, alias in enumerate(config.srcAlias()):
                print "    {0}: {1} -> {2}".format(i, alias.aliasName(), alias.src())

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        pass

