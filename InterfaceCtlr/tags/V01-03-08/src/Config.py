#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Config...
#
#------------------------------------------------------------------------

"""Configuration class for controller.

This software was developed for the SIT project.  If you use all or 
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
import types

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# make an object which cannot exist in configuration
_Missing = [None]
_Missing[0] = _Missing

def _subs(value, subs):
    "Do keyword substitution"
    
    if not subs: return value

    if type(value) in types.StringTypes:
        value = value % subs

    if type(value) == types.ListType:
        return [_subs(item) for item in value]

    return value

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class Config ( object ) :
    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor. """

        # define instance variables
        self._dict = {}
        
    #-------------------
    #  Public methods --
    #-------------------

    def add ( self, option, value, instrument=None, experiment=None ) :
        """Add one option with value.

        Stores option value, if option starts with 'list:' the value will be
        appended to the list. 
        @param option   configuration option name
        @param value    value for this option, usually number or string
        @param instrument if None then value used for any instrument
        @param experiment if None then value used for any experiment
        """

        key = (option, instrument, experiment)
        if option.startswith('list:') :
            self._dict.setdefault(key, []).append(value)
        else :
            self._dict[key] = value

    def merge(self, other):
        """Merge two configurations, adds content of other to this object,
        overrides local content with other if necessary, except for lists 
        that are joined""" 

        for key, value in other._dict.iteritems() :
            
            option = key[0]
            if option.startswith('list:') :
                self._dict.setdefault(key, []).extend(value)
            else :
                self._dict[key] = value
            
    def get(self, option, instrument=None, experiment=None, default=None, subs=None ) :
        """get the option value

        @param option     configuration option name
        @param instrument instrument name
        @param experiment experiment name
        @param default    default value to be returned
        @param subs       dictionary for keyword substitutions
        """

        if instrument and experiment:
            
            key = (option, instrument, experiment)
            val = self._dict.get(key, _Missing)
            if val is not _Missing: 
                return _subs(val, subs)
            
        if experiment:
            key = (option, None, experiment)
            val = self._dict.get(key, _Missing)
            if val is not _Missing: 
                return _subs(val, subs)

        if instrument:
            key = (option, instrument, None)
            val = self._dict.get(key, _Missing)
            if val is not _Missing: 
                return _subs(val, subs)

        key = (option, None, None)
        val = self._dict.get(key, _Missing)
        if val is not _Missing: 
            return _subs(val, subs)

        return _subs(default, subs)


    def __str__(self):
        res = []
        for key, value in self._dict.iteritems() :
            option = key[0]
            optline = "  %s: %s" % (option, value)
            if key[1] or key[2] :
                optline += " (instr=%s, exp=%s)" % (key[1], key[2])
            res.append(optline)
        return ', '.join(res)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
