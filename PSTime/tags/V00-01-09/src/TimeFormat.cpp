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
#include "PSTime/TimeFormat.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <stdio.h>
#include <string.h>
#include <boost/regex.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSTime/Exceptions.h"
#include "PSTime/TimeUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;

namespace {

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

  
  // Format nanoseconds part as fractional seconds
  void formatNsec(uint32_t nsec, int prec, std::ostream& str)
  {
    // constrain precision
    if (prec < 1) prec = 1;
    if (prec > 9) prec = 9;

    // format it to temporary buffer
    char buf[16];
    snprintf(buf, sizeof buf, ".%09d", int(nsec));
    
    // print dot and then first 'prec' digits.
    buf[prec+1] = '\0';
    str << buf;
  }
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSTime {
namespace TimeFormat {

/*
 * Parse the time string and return time
 */
Time
parseTime( const std::string& timeStr )
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
    if ( stm.tm_mon < 0 or stm.tm_mon > 11 ) throw TimeParseException( ERR_LOC, "month out of range" ) ;
    if ( match[3].matched ) stm.tm_mday = strtoul ( match.str(3).c_str(), 0, 10 ) ;
    if ( stm.tm_mday < 1 or stm.tm_mday > 31 ) throw TimeParseException( ERR_LOC, "day out of range" ) ;

    // parse the time
    if ( match[4].matched ) stm.tm_hour = strtoul ( match.str(4).c_str(), 0, 10 ) ;
    if ( stm.tm_hour < 0 or stm.tm_hour > 23 ) throw TimeParseException( ERR_LOC, "hours out of range" ) ;
    if ( match[5].matched ) stm.tm_min = strtoul ( match.str(5).c_str(), 0, 10 ) ;
    if ( stm.tm_min < 0 or stm.tm_min > 59 ) throw TimeParseException( ERR_LOC, "minutes out of range" ) ;
    if ( match[6].matched ) stm.tm_sec = strtoul ( match.str(6).c_str(), 0, 10 ) ;
    if ( stm.tm_sec < 0 or stm.tm_sec > 60 ) throw TimeParseException( ERR_LOC, "seconds out of range" ) ;
    if ( match[7].matched ) nsec = getNsec( match[7] ) ;

    if ( match[8].matched ) {

      // timezone is specified

      int tzoffset_min = 0 ;
      if ( match[8] != "Z" ) {

        // we have proper offset, calculate offset in minutes, will adjust it later
        int tz_hour = strtol ( match.str(10).c_str(), 0, 10 ) ;
        int tz_min = 0 ;
        if ( match[11].matched ) tz_min = strtol ( match.str(11).c_str(), 0, 10 ) ;
        if ( tz_hour > 12 or tz_min > 59 ) throw TimeParseException( ERR_LOC, "timezone out of range" ) ;

        tzoffset_min = tz_hour * 60 ;
        tzoffset_min += tz_min ;
        if ( match[9] == "-" ) tzoffset_min = -tzoffset_min ;

      }

      struct tm vtm = stm ;
      sec = TimeUtils::timegm( &stm ) ;
      if ( time_t(-1) == sec ) throw TimeParseException( ERR_LOC, "timegm() failed" ) ;

      // to validate the input compare the structures, if nothing was changed then input is OK
      if ( not ::cmp_tm( &stm, &vtm ) ) throw TimeParseException( ERR_LOC, "input time validation failed" ) ;

      // adjust for timezone
      sec -= tzoffset_min * 60 ;

    } else {

      // No timezone specified, we should assume the time is in the local timezone.
      // Let it guess the daylight saving time status.
      stm.tm_isdst = -1 ;
      struct tm vtm = stm ;
      sec = mktime( &stm ) ;
      if ( time_t(-1) == sec ) throw TimeParseException( ERR_LOC, "mktime() failed" ) ;

      // to validate the input compare the structures, if nothing was changed then input is OK
      if ( not ::cmp_tm( &stm, &vtm ) ) throw TimeParseException( ERR_LOC, "input time validation failed" ) ;

    }

  } else {

    throw TimeParseException( ERR_LOC, "failed to parse the string: "+timeStr ) ;

  }

  return Time( sec, nsec ) ;
}

/*
 * Convert time to string according to format
 */
std::string
format ( const Time& time, const std::string& afmt, Time::Zone zone )
{
  std::ostringstream str;
  format ( str, time, afmt, zone );
  return str.str();
}

void
format ( std::ostream& str, const Time& time, const std::string& afmt, Time::Zone zone )
{

  // broken down time
  struct tm stm = time.gettm(zone);
  
  char fill = str.fill('0'); 
  
  for (std::string::size_type p = 0; p != afmt.size(); ++p) {
    if (afmt[p] != '%') {
      // non-percent, copy and move forward
      str << afmt[p];
    } else if (p+1 == afmt.size()) {
      // percent, but it is the last char, just copy
      str << afmt[p];
    } else {
      
      // go to next char
      ++ p;
      
      switch (afmt[p]) {
      case '%' :
        str << afmt[p];
        break;
      case 'd' :
        // The day of the month as a decimal number (range 01 to 31).
        str << setw(2) << stm.tm_mday ;
        break;
      case 'F' :
        // Equivalent to %Y-%m-%d (the ISO 8601 date format).
        str << setw(4) << stm.tm_year+1900 << '-' << setw(2) << stm.tm_mon+1 
            << '-' << setw(2) << stm.tm_mday;
        break;
      case 'H' :
        // The hour as a decimal number using a 24-hour clock (range 00 to 23).
        str << setw(2) << stm.tm_hour;
        break;
      case 'm' :
        // The month as a decimal number (range 01 to 12).
        str << setw(2) << stm.tm_mon+1;
        break;
      case 'M' :
        // The minute as a decimal number (range 00 to 59).
        str << setw(2) << stm.tm_min;
        break;
      case 's' :
        // The number of seconds since the Epoch, i.e., since 1970-01-01 00:00:00 UTC.
        str << time.sec();
        break;
      case 'S' :
        // The second as a decimal number (range 00 to 60).
        str << setw(2) << stm.tm_sec;
        break;
      case 'T' :
        // The time in 24-hour notation (%H:%M:%S).
        str << setw(2) << stm.tm_hour << ':' << setw(2) << stm.tm_min 
            << ':' << setw(2) << stm.tm_sec;
        break;
      case 'Y' :
        // The year as a decimal number including the century.
        str << setw(4) << stm.tm_year+1900;
        break;
      case 'z' :
        // 'Z' if time is printed in UTC zone, or offset from UTC to local time.
        if (zone == Time::UTC) {
          str << 'Z';
        } else {
          // Non-portable code, tm_gmtoff is glibc extension
          char sign = stm.tm_gmtoff > 0 ? '+' : '-' ;
          int offmin = abs(stm.tm_gmtoff / 60);
          int offhour = offmin / 60;
          offmin %= 60;
          str << sign << setw(2) << offhour;
          if (offmin) str << setw(2) << offmin;
        }
        break;
      case '.' :
        // might be .<N>f
        if (p+2 < afmt.size() and isdigit(afmt[p+1]) and afmt[p+2] == 'f') {
          ::formatNsec(time.nsec(), afmt[p+1]-'0', str);
          p += 2;
        }
        break;
      case 'f' :
        // Equivalent to %.9f
        ::formatNsec(time.nsec(), 9, str);
        break;
      default :
        // unknown, copy both % and this guy
        str << '%' << afmt[p];
        break;
      }
    }
  }
  
  str.fill(fill); 
}


} // namespace TimeFormat
} // namespace PSTime
