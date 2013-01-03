#ifndef PSTIME_TIME_H
#define PSTIME_TIME_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Time.
// 
//            The transformation to/from human-readable time format, required by I/O, 
//            is based on the standard ISO8601 time presentation.
// 
//            This is essentially a wrap-up for clock_gettime() [UNIX/POSIX time]
//            access to high-resolution time method,
//            utilizing "struct timespec" and "struct tm" from <time.h>    
//            Some arithmetics between PSTime and PSDuration objects is also available.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>
#include <iosfwd>
#include <string>
#include <stdint.h>

#include "PSTime/Duration.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  @defgroup PSTime PSTime package
 *  
 *  @brief Package defining time-related classes for psana framework.
 *  
 *  This package include several classes and functions which define
 *  time representation and operations with time. It is based on on 
 *  standard ISO8601 external representation (http://en.wikipedia.org/wiki/ISO_8601)
 *  but it does not provide complete support and in samoe cases it is
 *  made more human-oriented.
 *  
 *  Core of the package are two classes: Time and Duration. Other classes define 
 *  formatting functions and various utility functions.
 */


namespace PSTime {

/**
 *  @ingroup PSTime
 * 
 *  @brief Standard time class for use in LCLS software.
 *
 *  The transformation to/from human-readable time format, required by I/O, 
 *  is based on the standard ISO8601 time presentation.
 *
 *  This is essentially a wrap-up for clock_gettime() [UNIX/POSIX time]
 *  access to high-resolution time method,
 *  utilizing "struct timespec" and "struct tm" from <time.h>    
 *  Some arithmetics between PSTime and PSDuration objects is also available.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class Time  {
public:

  /**
   *  @brief Type for time zone designation.
   *  
   *  There are several methods in Time class which can produce time
   *  representation in either local or UTC time zones. By default 
   *  all these methods assume local time zone, use UTC enum value as 
   *  the parameter to those methods to produce UTC representation.
   */
  enum Zone { UTC, Local };

  /**
   *  @brief Default constructor makes zero (epoch) time.
   *  
   *  Use now() method if you need current time.
   */
  Time ();

  /** 
   *  @brief Construct from UNIX time.
   * 
   *  Constructs a time from an unsigned number of seconds since the
   *  the Unix epoch of 1970-01-01; nanoseconds are optional, if needed.
   */
  explicit Time (time_t sec_since_1970_01_01, time_t nsec = 0);

  /**
   * @brief Construct from broken-down representation.
   * 
   * @param[in] year  Calendar year C.E. (e.g., 2010).
   * @param[in] month Month in the range [1,12].
   * @param[in] day   Day of the month in the range [1,31].
   * @param[in] hour  Hours in the range [0,23].
   * @param[in] min   Minutes after the hour in the range [0,59].
   * @param[in] sec   Seconds in the range [0,59*] (*) might be up to 61 for leap seconds. 
   * @param[in] nsec  Nanoseconds after the second in the range [0,999999999].
   * @param[in] zone  Time zone, default is local time zone.
   */
  Time (int year,
        int month,
        int day,
        int hour,
        int min,
        int sec,
        int nsec = 0,
        Zone zone = Local);

  /** 
   * @brief Constructs a time from the timespec struct (for high resolution time).
   */
  explicit Time (struct timespec& ts);

  /** 
   * @brief Constructs a time from the tm struct (broken-down time) and zone.
   */
  explicit Time (struct tm& tms, Zone zone = Local);

  /** 
   * @brief Constructs a time from the string representation.
   * 
   * Accepts the time stamp in format:
   *     @li \<date\>, or
   *     @li \<date\> \<time\>[.\<fraction-of-sec\>][\<zone\>], or
   *     @li \<date\>T\<time\>[.\<fraction-of-sec\>][\<zone\>]
   *     
   *     where
   *     
   *     @li \<date\> should be in format @c YYYY-MM-DD, @c YYYYMMDD, @c YYYY-MM, @c YYYYMM, 
   *            or @c YYYY, if month or day is missing they are assumed to be 1;
   *     @li \<time\> should be in format @c HH:MM:SS, @c HHMMSS, @c HH:MM, @c HHMM, @c HH, 
   *            if minutes or seconds are missing they are assumed to be 0;
   *     @li \<fraction-of-sec\> may have up to 9 digits; if this field is missing,
   *            it is assumed equal to 0, fraction require seconds to be specified;
   *     @li \<zone\> should be @c Z for UTC or in format @c {sign}HH[:MM] or {sign}HHMM,
   *             if this field is missing time is assumed to be in local time zone.
   *             
   *  Examples of valid input are:
   *  @li "2000-01-01 00:00:00" - beginning of year 2000 in local time zone
   *  @li "2000-01-01 00:00:00Z" - beginning of year 2000 in UTC
   *  @li "2000-01-01 00:00:00-08" - beginning of year 2000 in PST
   *  @li "2000-01-01 00:00:00.000000001" - one nanosecond after the beginning of year 2000 
   *            in local time zone
   *  @li "20000101T000000.001" - one millisecond after the beginning of year 2000 in 
   *            local time zone in compact form
   *  @li "20000101" - beginning of year 2000 in local time zone
   *  @li "2000" - same as above
   *
   *  @throw TimeParseException
   */
  explicit Time (const std::string& str);

  /**
   *  @brief Subtract two times and return duration.
   *  
   *  Duration cannot be negative, so t1-t2 is always equal to t2-t1.
   */
  Duration operator-(  const Time& t ) const;               

  /// Add duration to time.
  Time& operator+=( const Duration& d );

  /// Subtract duration from time.
  Time& operator-=( const Duration& d );

  /// Compare two times
  bool operator!=( const Time& t ) const { 
    return ( m_utcSec != t.m_utcSec || m_utcNsec != t.m_utcNsec );
  }

  /// Compare two times
  bool operator==( const Time& t ) const { return !( *this != t ); }

  /// Compare two times
  bool operator<( const Time& t ) const { 
    return ( m_utcSec  < t.m_utcSec ) || 
           ( m_utcSec == t.m_utcSec && m_utcNsec < t.m_utcNsec );
  }

  /// Compare two times
  bool operator>( const Time& t ) const { 
    return ( m_utcSec  > t.m_utcSec ) || 
           ( m_utcSec == t.m_utcSec && m_utcNsec > t.m_utcNsec );
  }

  /// Compare two times
  bool operator>=( const Time& t ) const { return !( *this < t ); }

  /// Compare two times
  bool operator<=( const Time& t ) const { return !( *this > t ); }

  /// POSIX sec. since  00:00:00 Jan. 1, 1970 UTC
  time_t sec() const {return m_utcSec;}

  /// returns nanoseconds value
  time_t nsec() const {return m_utcNsec;}

  /// Returns the Time object content via high-resolution time timespec structure.
  struct timespec getTimeSpec() const;

  /// Returns the Time object content via broken-down time tm structure for indicated zone.
  struct tm gettm( Zone zone = Local ) const;

  /**
   *  @brief Return time representation in extended ISO8601 format.
   *  
   *  Output will look like "2011-01-30 17:52:50.123456789-08". This format 
   *  is acceptable for human-readable output.
   *  
   *  @param[in] zone  Time zone to use for string representation
   *  @param[in] nsecPrecision Precision of nanoseconds field, if 0 then nanoseconds
   *             will not be shown.
   */
  std::string asString( Zone zone = Local, int nsecPrecision = 9 ) const;  // YYYY-MM-DD HH:MM:SS.NNNNNNNNN+HH:MM

  /**
   *  @brief Return time representation in basic ISO8601 format.
   *  
   *  Output will look like "20110130T175250-08". This format is intended 
   *  for machine-readable output, such as timestamps in file names, etc.
   *  
   *  @param[in] zone  Time zone to use for string representation
   *  @param[in] nsecPrecision Precision of nanoseconds field, if 0 then nanoseconds
   *             will not be shown.
   */
  std::string asStringCompact( Zone zone = Local, int nsecPrecision = 0 ) const;  // HHMMSSTHHMMSS+HHMM

  
  /**
   *  @brief Format time according the supplied format string.
   *  
   *  For full description of the acceptable format codes see TimeFormat::format() method.
   *  
   *  @param[in] fmt   Format string
   *  @param[in] zone  Time zone to use for string representation
   */
  std::string asStringFormat( const std::string& fmt="%Y-%m-%d %H:%M:%S%f%z", Zone zone = Local ) const;

protected:

private:

  // Data members
  time_t m_utcSec;     ///< Number of seconds since UNIX epoch time
  uint32_t m_utcNsec;  ///< Number of nanoseconds within second (range 0..999999999)

//------------------
// Static Members --
//------------------

public:

  /**
   *  @brief Returns current time.
   */
  static Time now() ;
  
}; // class Time


/// Addition of duration and time objects
inline 
Time  
operator+(const Duration& d, const Time& t) 
{
  Time tmp = t;
  tmp += d;
  return tmp;
}

/// Addition of time and duration objects
inline
Time     
operator+(const Time& t, const Duration& d) 
{
  Time tmp = t;
  tmp += d;
  return tmp;
}

/// Subtract duration from time
inline
Time
operator-(const Time& t, const Duration& d )
{
  Time tmp = t;
  tmp -= d;
  return tmp;
}

/// Stream insertion operator, prints result of Time::asString() method.
std::ostream& 
operator<<( std::ostream& os, const Time& t );

}  // namespace PSTime

#endif // PSTIME_TIME_H
