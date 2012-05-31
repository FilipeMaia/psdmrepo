#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: skip_stop_example.py 2941 2012-02-15 19:53:01Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  Pyana user analysis module skip_stop_example...
#
#------------------------------------------------------------------------

"""Simple module which demonstrates Skip/Stop/Terminate feature.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: skip_stop_example.py 2941 2012-02-15 19:53:01Z salnikov@SLAC.STANFORD.EDU $

@author Andy Salnikov
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 2941 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------
import pyana

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#---------------------
#  Class definition --
#---------------------
class skip_stop_example (object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :

        self.count = 0

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        pass

    def event( self, evt, env ) :

        self.count += 1

        if self.count % 7 == 0 :
            logging.info("Requesting Skip, event: %d", self.count)
            return pyana.Skip

        if self.count > 30 :
            logging.info("Requesting Stop, event: %d", self.count)
            return pyana.Stop

        # this piece of code is never executed, just for example
        if self.count < 0 :
            logging.info("Requesting Terminate, event: %d", self.count)
            return pyana.Terminate

    def endjob( self, env ) :
        pass