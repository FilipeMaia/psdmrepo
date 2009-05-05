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

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

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
