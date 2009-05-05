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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "LusiTime/Exceptions.h"

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
