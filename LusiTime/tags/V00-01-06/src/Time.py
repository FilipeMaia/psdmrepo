#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Time...
#
#------------------------------------------------------------------------

"""Brief one-line description of the module.

Following paragraphs provide detailed description of the module, its
contents and usage. This is a template module (or module template:)
which will be used by programmers to create new Python modules.
This is the "library module" as opposed to executable module. Library
modules provide class definitions or function definitions, but these
scripts cannot be run by themselves.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import time

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import TimeFormat

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class Time ( object ) :
    """Common time class. 
    
    Counts time since the standard UNIX epoch. Provides nanosecond precision.
    """

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, sec = None, nsec = 0 ) :
        """Constructor.

        @param sec    seconds since epoch
        @param nsec   nanoseconds
        """
        if nsec < 0 or nsec > 999999999 : raise ValueError("nanoseconds value out of range")
        self._sec = sec ;
        self._nsec = nsec ;

    #-------------------
    #  Public methods --
    #-------------------

    def sec(self) :
        return self._sec

    def nsec(self) :
        return self._nsec

    def isValid(self) :
        return self._sec is not None

    def to64(self):
        """ Pack time into a 64-bit number. """
        if self._sec is None : raise ValueError("Time.to64: converting invalid object")
        return self._sec*1000000000L + self._nsec

    def toString(self, fmt="%F %T%f%z" ):
        """ Format time according to format string """
        if self._sec is None : raise ValueError("Time.toString: converting invalid object")
        return TimeFormat.formatTime( self._sec, self._nsec, fmt )

    def __str__ ( self ):
        """ Convert to a string """
        return self.toString()

    def __repr__ ( self ):
        """ Convert to a string """
        if self._sec is None : return "LusiTime.Time()"
        return "<LusiTime.Time:%s>" % self.toString("S%s%f")

    def __cmp__ ( self, other ):
        """ compare two Time objects """
        if not isinstance(other,Time) : raise TypeError ( "Time.__cmp__: comparing to unknown type" )
        if self._sec is None or other._sec is None : raise ValueError ( "Time.__cmp__: comparing invalid times" )
        return cmp ( ( self._sec,self._nsec ), ( other._sec,other._nsec ) )

    def __hash__ ( self ):
        """ calculate hash value for use in dictionaries, returned hash value 
        should be 32-bit integer """
        if self._sec is None : raise ValueError ( "Time.__hash__: invalid time value" )
        return hash( (self._sec, self._nsec) )

    #--------------------------------
    #  Static/class public methods --
    #--------------------------------

    @staticmethod
    def now():
        """ Get current time, resolution can be lower than 1ns """
        t = time.time()
        sec = int(t)
        nsec = int( (t-sec) * 1e9 )
        return Time( sec, nsec )

    @staticmethod
    def from64( packed ):
        """ Unpack 64-bit time into time object """
        sec, nsec = divmod( packed, 1000000000 )
        return Time( sec, nsec )

    @staticmethod
    def parse( s ):
        """ Convert string presentation into time object """
        sec, nsec = TimeFormat.parseTime( s )
        return Time( sec, nsec )
    
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
