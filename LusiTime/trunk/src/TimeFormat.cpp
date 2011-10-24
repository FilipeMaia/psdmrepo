//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeFormat...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TimeFormat.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <stdio.h>
#include <string.h>
#include <boost/regex.hpp>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // regular expression to match %f format
  boost::regex ffmtre( "%([.](\\d+))?f" ) ;

  // regular expressions for parsing date and time
#define DATE_RE "(\\d{4})(?:-?(\\d{2})(?:-?(\\d{2}))?)?"
#define TIME_RE "(\\d{1,2})(?::?(\\d{2})(?::?(\\d{2})(?:[.](\\d{1,9}))?)?)?"
#define TZ_RE "Z|(?:([-+])(\\d{2})(?::?(\\d{2}))?)"
  boost::regex dtre( "^" DATE_RE "(?:(?: +|T)(?:" TIME_RE ")?(" TZ_RE ")?)?$" ) ;

  // time specified as seconds.fractions
  boost::regex secre ( "^S(\\d{0,10})(?:[.](\\d{1,9}))?$" ) ;


  // Turn string into nanoseconds, strings is everything that
  // appears after decimal dot.
  // "1"    -> 100000000 ns
  // "123"  -> 123000000 ns
  // "123456789987654321" -> 123456789ns (truncation, no rounding)
  long getNsec ( const std::string& nsecStr )
  {
    char buf[10] ;
    unsigned ndig = nsecStr.length() ;
    if ( ndig > 9 ) ndig = 9 ;

    std::copy ( nsecStr.begin(), nsecStr.begin()+ndig, buf ) ;
    std::fill ( buf+ndig, buf+9, '0' ) ;
    buf[9] = '\0' ;
    return strtol ( buf, 0, 10 ) ;
  }

  // compare two tm structs
  bool cmp_tm ( struct tm* lhs, struct tm* rhs )
  {
    if ( lhs->tm_year != rhs->tm_year ) return false ;
    if ( lhs->tm_mon != rhs->tm_mon ) return false ;
    if ( lhs->tm_mday != rhs->tm_mday ) return false ;
    if ( lhs->tm_hour != rhs->tm_hour ) return false ;
    if ( lhs->tm_min != rhs->tm_min ) return false ;
    if ( lhs->tm_sec != rhs->tm_sec ) return false ;
    if ( lhs->tm_isdst >= 0 and rhs->tm_isdst >= 0 ) {
      if ( lhs->tm_isdst != rhs->tm_isdst ) return false ;
    }
    return true ;
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace LusiTime {

/**
 * Parse the time string and return time
 */
Time
TimeFormat::parse( const std::string& timeStr ) throw (Exception)
{
  time_t sec = 0 ;
  long nsec = 0 ;

  // match the string against the regular expression
  boost::smatch match ;
  if ( boost::regex_match( timeStr, match, secre ) ) {

    // time is given as S<sec>.<frac>
    sec = strtoul ( match.str(1).c_str(), 0, 10 ) ;
    if ( match[2].matched ) nsec = getNsec( match[2] ) ;

  } else if ( boost::regex_match( timeStr, match, dtre ) ) {

    // Close-to-ISO8601 time specification

    struct tm stm ;
    memset( &stm, 0, sizeof stm ) ;

    stm.tm_mday = 1 ;

    // parse the date
    stm.tm_year = strtoul ( match.str(1).c_str(), 0, 10 ) ;
    stm.tm_year -= 1900 ;
    if ( match[2].matched ) stm.tm_mon = strtoul ( match.str(2).c_str(), 0, 10 ) - 1 ;
    if ( stm.tm_mon < 0 or stm.tm_mon > 11 ) throw ParseException( "month out of range" ) ;
    if ( match[3].matched ) stm.tm_mday = strtoul ( match.str(3).c_str(), 0, 10 ) ;
    if ( stm.tm_mday < 1 or stm.tm_mday > 31 ) throw ParseException( "day out of range" ) ;

    // parse the time
    if ( match[4].matched ) stm.tm_hour = strtoul ( match.str(4).c_str(), 0, 10 ) ;
    if ( stm.tm_hour < 0 or stm.tm_hour > 23 ) throw ParseException( "hours out of range" ) ;
    if ( match[5].matched ) stm.tm_min = strtoul ( match.str(5).c_str(), 0, 10 ) ;
    if ( stm.tm_min < 0 or stm.tm_min > 59 ) throw ParseException( "minutes out of range" ) ;
    if ( match[6].matched ) stm.tm_sec = strtoul ( match.str(6).c_str(), 0, 10 ) ;
    if ( stm.tm_sec < 0 or stm.tm_sec > 60 ) throw ParseException( "seconds out of range" ) ;
    if ( match[7].matched ) nsec = getNsec( match[7] ) ;

    if ( match[8].matched ) {

      // timezone is specified

      int tzoffset_min = 0 ;
      if ( match[8] != "Z" ) {

        // we have proper offset, calculate offset in minutes, will adjust it later
        int tz_hour = strtol ( match.str(10).c_str(), 0, 10 ) ;
        int tz_min = 0 ;
        if ( match[11].matched ) tz_min = strtol ( match.str(11).c_str(), 0, 10 ) ;
        if ( tz_hour > 12 or tz_min > 59 ) throw ParseException( "timezone out of range" ) ;

        tzoffset_min = tz_hour * 60 ;
        tzoffset_min += tz_min ;
        if ( match[9] == "-" ) tzoffset_min = -tzoffset_min ;

      }

      struct tm vtm = stm ;
      sec = timegm( &stm ) ;
      if ( time_t(-1) == sec ) throw ParseException( "timegm() failed" ) ;

      // to validate the input compare the structures, if nothing was changed then input is OK
      if ( not ::cmp_tm( &stm, &vtm ) ) throw ParseException( "input time validation failed" ) ;

      // adjust for timezone
      sec -= tzoffset_min * 60 ;

    } else {

      // No timezone specified, we should assume the time is in the local timezone.
      // Let it guess the daylight saving time status.
      stm.tm_isdst = -1 ;
      struct tm vtm = stm ;
      sec = timelocal( &stm ) ;
      if ( time_t(-1) == sec ) throw ParseException( "timelocal() failed" ) ;

      // to validate the input compare the structures, if nothing was changed then input is OK
      if ( not ::cmp_tm( &stm, &vtm ) ) throw ParseException( "input time validation failed" ) ;

    }

  } else {

    throw ParseException( "failed to parse the string: "+timeStr ) ;

  }

  return Time( sec, nsec ) ;
}

/**
 * Convert time to string according to format
 */
std::string
TimeFormat::format ( const Time& time, const std::string& afmt ) throw (Exception)
{
  if ( not time.isValid()) return "<InvalidTime>" ;

  // replace %f (and its variations) with fractional seconds
  std::string fmt = afmt ;
  boost::smatch match ;
  while ( boost::regex_search( fmt, match, ffmtre ) ) {

    // get the precision
    unsigned int precision = 9 ;
    if ( match[2].matched ) {
      const std::string& prec_str = match[2] ;
      precision = strtoul ( prec_str.c_str(), 0, 10 ) ;
      // do not allow nonsensical values here
      if ( precision > 9 ) precision = 9 ;
      if ( precision < 1 ) precision = 1 ;
    }

    // make replacement string
    char subsec[16] ;
    snprintf ( subsec, sizeof subsec, ".%09ld", time.nsec() ) ;
    subsec[precision+1] = '\0' ;

    // replace %f with this string
    fmt.replace ( match.position(), match.length(), subsec ) ;
  }

  // Translate time into the broken down 'tm' structure.
  struct tm time2convert ;
  memset( &time2convert, 0, sizeof(time2convert)) ;

  time_t sec = time.sec() ;
  if ( localtime_r( &sec, &time2convert ) != &time2convert ) {
    throw Exception( "failed in localtime_r()" ) ;
  }

  // Note, that the result value returned by the function will not
  // include the terminating null symbol. But the buffer (in case of
  // success) will have the terminated string.
  //
  const size_t MAXSIZE = 256 ;  // including the terminating null symbol
  char buf[256] ;

  const size_t res = strftime( buf, MAXSIZE, fmt.c_str(), &time2convert );
  if (res == 0)
    throw Exception( "failed in strftime()" ) ;

  return std::string( buf ) ;
}


} // namespace LusiTime
