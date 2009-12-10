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
#include <iostream>
#include <stdint.h>

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
    * The input string is expected to have on of the the following formats:
    *
    *   <year>-<month>-<day> <hours>:<minutes>:<seconds>[.<fractions>][<tz>]
    *   <year>-<month>-<day> <hours>:<minutes>:<seconds>[.<fractions>][<tz>]
    *   <year>-<month>-<day> <hours>:<minutes>[<tz>]
    *   <year>-<month>-<day> <hours>[<tz>]
    *   <year>-<month>-<day>     - means 00:00:00 of the day
    *   <year>-<month>           - means 00:00:00 of the 1st day of the month
    *   <year>                   - means 00:00:00 of the January 1st of the year
    *   S<seconds-since-epoch>[.<fractions>]
    *
    * Date/time separators (- and :) are optional. Date and time can be 
    * separated by any number of spaces or a single 'T' character. Timezone 
    * offset <tz> can be either 'Z' for UTC, or {+|-}HH[[:]MM]
    *
    * Few examples of valid input:
    *
    *   2009-06-05
    *   2009-06-05 10:12:32
    *   2009-06-05T10:12:32Z
    *   2009-06-05T10:12:32.123-08
    *   2009-06-05T10:12-08:00
    *   20090605T101232.123+0100
    *   S1234567890.001
    */
  static Time parse(const std::string& inTimeStr) throw (Exception) ;

  /**
    * Unpack a 64-bit number into time
    *
    * A number on the input is supposed to be a previously packed value of
    * an object of the current class.
    *
    * The overflow is reported through the exception
    */
  static Time from64(uint64_t inNumber) throw (Exception) ;

  /**
    * Pack time into a 64-bit number
    *
    * An input timestamp must be valid. Otherwise an exception will be thrown.
    */
  static uint64_t to64(const Time& inTime) throw (Exception) ;

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
  long nsec() const { return m_time.tv_nsec ; }

  /// check if the time is valid
  bool isValid() const { return m_time.tv_nsec >= 0 ; }

  /**
    * Translate into the string representation.
    *
    * The output string will have the following format:
    *
    *   <year>-<month>-<day> <hours>:<minutes>:<seconds>.<nanoseconds><TZ>
    *
    * Where the month will be represented by a number. For example:
    *
    *   2009-06-05 10:12:32.123456789-0800
    *
    * If the object is not valid then the method will throw an
    * exception.
    */
  std::string toString() const throw (Exception) ;

  /**
    * Translate into the string representation according to given format.
    * Format specifiers supported:
    *
    * %F    = %Y-%m-%d
    * %f    fractions of second with preceding dot, default precision is 9,
    *       precision can be changed with %.Nf  format 
    * %H    hour (00...24)
    * %j    day of the year (001...366)
    * %m    month (01...12)
    * %M    minute (00...59)
    * %s    seconds since epoch
    * %S    seconds (00...60)
    * %T    = %H:%M:%S
    * %Y    year
    * %z    timezone
    * %%    literal %
    *
    * If the object is not valid then the method will throw an
    * exception.
    */
  std::string toString( const std::string& fmt ) const throw (Exception) ;

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

inline
std::ostream&
operator<< ( std::ostream& out, const Time& t ) {
  return out << t.toString() ;
}

} // namespace LusiTime

#endif // LUSITIME_TIME_H
