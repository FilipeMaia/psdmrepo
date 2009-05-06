#ifndef LUSITIME_TIME_H
#define LUSITIME_TIME_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Time.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "LusiTime/Exceptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace LusiTime {

/**
 *  Common time class. Counts time since the standard UNIX epoch.
 *  Provides nanosecond precision.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Time  {
public:

  /// Returns current time
  static Time now() ;

  /**
    * Parse an input string as the date and time.
    *
    * If successful then the method will return a valid object of
    * the current class. Otherwise an exception will be thrown.
    *
    * The parsed string is expected to have the following format:
    *
    *   <year>-<month>-<day> <hours>:<minutes>:<seconds>
    *
    * For example:
    *
    *   2009-JUN-05 10:12:32
    *   2009-06-05 10:12:32
    *
    * Other notes on a format of the input string:
    *
    *   - whitespaces, tabs, newline or carriage return symbols at the begining
    *   of the string are allowed
    *
    *   - whitespaces, tabs, newline or carriage return symbols at the end
    *   of the string are also allowed
    *
    *   - the date and time fields must be separated by at least one whitespace
    *   character. It's also allowed to have more then one whitespace in there.
    *
    *   - month can be given either as a number or as a name. The name can be
    *   short ('JAN', 'FEB', etc.) or complete ('JANUARY', 'FEBRUARY', etc.)
    *
    * TODO: The 'mktime' used in the current implementation of the method
    *       uses local timezone, therefore the code will run correctly only
    *       on a machine configured for PDT. We need to resolve this issue.
    */
  static Time parse(const std::string& inTimeStr) throw (Exception) ;

  // Default constructor, makes invalid time
  Time () {
    m_time.tv_sec = 0xFFFFFFFF ;
    m_time.tv_nsec = -1 ;
  }

  /// Build time from seconds and nanoseconds
  Time ( time_t sec, long nsec ) {
    m_time.tv_sec = sec ;
    m_time.tv_nsec = nsec ;
  }

  /// Destructor
  ~Time () {}

  /// return seconds
  time_t sec() const { return m_time.tv_sec ; }

  /// return nanoseconds
  time_t nsec() const { return m_time.tv_nsec ; }

  /// check if the time is valid
  bool isValid() const { return m_time.tv_nsec >= 0 ; }

  /**
    * Translate into the string representation.
    *
    * The output string will have the following format:
    *
    *   <year>-<month>-<day> <hours>:<minutes>:<seconds>
    *
    * Where the month will be represented by a number. For example:
    *
    *   2009-06-05 10:12:32
    *
    * Also note, that the result won't include nanoseconds. This is done
    * to make the output format compatible with the one expected by
    * the opposite method Time::parse().
    *
    * If the object is not valid then the method will throw an
    * exception.
    *
    * TODO: The 'localtime_r' used in the current implementation of the method
    *       uses local timezone, therefore the code will run correctly only
    *       on a machine configured for PDT. We need to resolve this issue.
    */
  std::string toString() const throw (Exception) ;

protected:

private:

  // Data members
  struct timespec m_time ;

};

/// comparisons
bool operator< ( const Time& t1, const Time& t2 ) ;
bool operator<= ( const Time& t1, const Time& t2 ) ;
bool operator> ( const Time& t1, const Time& t2 ) ;
bool operator>= ( const Time& t1, const Time& t2 ) ;
bool operator== ( const Time& t1, const Time& t2 ) ;
bool operator!= ( const Time& t1, const Time& t2 ) ;

} // namespace LusiTime

#endif // LUSITIME_TIME_H
