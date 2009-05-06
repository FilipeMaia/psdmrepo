//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Time...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "LusiTime/Time.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace LusiTime {

/* Date and time formats supported by the current implementation.
 */
const char* const fmt[2] = {
  "%Y-%m-%d %T",    // yyyy-mm-dd        hh:mm:ss (month number)
  "%Y-%b-%d %T"     // yyyy-monthname-dd hh:mm:ss (short or full month name)
} ;
const int fmt_len = 2 ;

/// Returns current time
Time Time::now()
{
  struct timespec ts ;
  int res = clock_gettime( CLOCK_REALTIME, &ts );
  if ( res < 0 ) {
    throw ExceptionErrno( "clock_gettime failed" );
  }
  return Time ( ts.tv_sec, ts.tv_nsec ) ;
}

/// Parse an input string as the date and time
Time Time::parse( const std::string& inTimeStr ) throw (Exception)
{
  // Truncate white spaces, tabs and newlines (if any are present) on
  // the both ends of the input string. Then set up a clean context for
  // the time translation.
  //
  const char* inTime = inTimeStr.c_str() ;
  while ( (*inTime == ' ' ) ||
          (*inTime == '\t') ||
          (*inTime == '\r') ||
          (*inTime == '\n')) inTime++ ;

  const char* inTimeLast = inTime + strlen (inTime) ;
  while ( (inTimeLast > inTime) &&
          ((*(inTimeLast-1) == ' ' )) ||
          ((*(inTimeLast-1) == '\t')) ||
          ((*(inTimeLast-1) == '\r')) ||
          ((*(inTimeLast-1) == '\n'))) inTimeLast-- ;

  // Now translate a string into the broken down structure 'tm'
  // using one of the formats supported by the current implementation.
  //
  struct tm parsed_time ;

  for ( int i = 0 ; i < fmt_len ; ++i ) {

    memset( &parsed_time, 0, sizeof(parsed_time)) ;

    parsed_time.tm_isdst = -1 ; // Set the DST state as unknown, and let
                                // the function to derive its status based
                                // on a configuration of a target system.

    const char* last = strptime( inTime, fmt[i], &parsed_time );
    if ( inTimeLast == last ) break ;
    if ( i == ( fmt_len - 1 ))
      throw Exception( "strptime() failed" ) ;
  }

  // Translate the broken down structure into the number of seconds
  // since UNIX Epoch. The number of nanoseconds will stay equal to 0.
  //
  // TODO: The 'mktime' uses local timezone, therefore the code
  //       will run correctly only on a machine configured
  //       for PDT. We need to resolve this issue.
  //
  struct timespec ts ;
  ts.tv_sec = mktime( &parsed_time ) ;
  if (( time_t)-1 == ts.tv_sec )
    throw Exception( "mktime() failed" ) ;

  return Time ( ts.tv_sec, 0 ) ;
}

/// Translate into the string representation.

std::string Time::toString() const throw (Exception)
{
  if ( !this->isValid())
    throw Exception( "toString(): invalid time" ) ;

  // Translate the curent object's time into the broken down 'tm'
  // structure.
  //
  struct tm time2convert ;
  memset( &time2convert, 0, sizeof(time2convert)) ;

  if ( localtime_r( &m_time.tv_sec, &time2convert ) != &time2convert )
    throw Exception( "failed in localtime_r()" ) ;

  // Note, that the result value returned by the function will not
  // include the terminating null symbol. But the buffer (in case of
  // success) will have the terminated string.
  //
  const size_t MAXSIZE = 256 ;  // including the terminating null symbol
  char buf[256] ;

  const size_t res = strftime( buf, MAXSIZE, fmt[0], &time2convert );
  if (res == 0)
    throw Exception( "failed in strftime()" ) ;

  return std::string( buf ) ;
}

/// comparisons
bool
operator< ( const Time& t1, const Time& t2 )
{
  if ( not ( t1.isValid() and t2.isValid() ) ) {
    throw Exception( "operator<(Time,Time): invalid time" ) ;
  }
  if ( t1.sec() < t2.sec() ) return true ;
  if ( t1.sec() > t2.sec() ) return false ;
  if ( t1.nsec() < t2.nsec() ) return true ;
  return false ;
}

bool
operator<= ( const Time& t1, const Time& t2 )
{
  if ( not ( t1.isValid() and t2.isValid() ) ) {
    throw Exception( "operator<=(Time,Time): invalid time" ) ;
  }
  if ( t1.sec() < t2.sec() ) return true ;
  if ( t1.sec() > t2.sec() ) return false ;
  if ( t1.nsec() <= t2.nsec() ) return true ;
  return false ;
}

bool
operator> ( const Time& t1, const Time& t2 )
{
  if ( not ( t1.isValid() and t2.isValid() ) ) {
    throw Exception( "operator>(Time,Time): invalid time" ) ;
  }
  if ( t1.sec() > t2.sec() ) return true ;
  if ( t1.sec() < t2.sec() ) return false ;
  if ( t1.nsec() > t2.nsec() ) return true ;
  return false ;
}

bool
operator>= ( const Time& t1, const Time& t2 )
{
  if ( not ( t1.isValid() and t2.isValid() ) ) {
    throw Exception( "operator>=(Time,Time): invalid time" ) ;
  }
  if ( t1.sec() > t2.sec() ) return true ;
  if ( t1.sec() < t2.sec() ) return false ;
  if ( t1.nsec() >= t2.nsec() ) return true ;
  return false ;
}

bool
operator== ( const Time& t1, const Time& t2 )
{
  if ( not ( t1.isValid() and t2.isValid() ) ) {
    throw Exception( "operator==(Time,Time): invalid time" ) ;
  }
  return t1.sec() == t2.sec() and t1.nsec() == t2.nsec() ;
}

bool
operator!= ( const Time& t1, const Time& t2 )
{
  if ( not ( t1.isValid() and t2.isValid() ) ) {
    throw Exception( "operator!=(Time,Time): invalid time" ) ;
  }
  return t1.sec() != t2.sec() or t1.nsec() != t2.nsec() ;
}

} // namespace LusiTime
