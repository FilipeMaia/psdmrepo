#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Psana user analysis module dump_l3t...
#
#------------------------------------------------------------------------

"""Psana module to dump L3T information.

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
class dump_l3t (object) :
    '''Class whose instance will be used as a user analysis module.'''

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

        config = env.configStore().get(L3T.Config, self.m_src)
        if config:
        
            print "%s: %s" % (config.__class__.__name__, self.m_src)
            print "  module_id =\"{0}\"".format(config.module_id())
            print "  description =\"{0}\"".format(config.desc())

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """

        data = evt.get(L3T.Data, self.m_src)
        if data:
            print "{0}: accept={1}".format(data.__class__.__name__, data.accept())

