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

//-----------------------
// This Class's Header --
//-----------------------
#include "LusiTime/Time.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "TimeFormat.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace LusiTime {

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
  return TimeFormat::parse ( inTimeStr ) ;
}

/// Translate into the string representation.

std::string Time::toString() const throw (Exception)
{
  // use default format for string presentation including nanoseconds 
  // and timezone offset
  return TimeFormat::format( *this, "%F %T%f%z" ) ;
}

/// Translate into the string representation.

std::string Time::toString( const std::string& fmt ) const throw (Exception)
{
  return TimeFormat::format( *this, fmt ) ;
}

// Translate into/from 64-bit unsigned integer

const long long unsigned NSEC_IN_ONE_SEC = 1*1000*1000*1000ULL ;
const long long unsigned MAX_NSEC = (2*1024*1024*1024ULL - 1) * NSEC_IN_ONE_SEC ;

Time Time::from64( uint64_t inNumber ) throw (Exception)
{
  if (inNumber > MAX_NSEC)
    throw Exception( "Time::from64(number): invalid number" ) ;

  return Time( inNumber / NSEC_IN_ONE_SEC,
               inNumber % NSEC_IN_ONE_SEC ) ;
}

uint64_t Time::to64( const Time& inTime ) throw (Exception)
{
  if (!inTime.isValid())
    throw Exception( "Time::to64(Time): invalid time" ) ;
  return inTime.sec() * NSEC_IN_ONE_SEC + inTime.nsec() ;
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
