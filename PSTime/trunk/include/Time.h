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
#include <stdio.h>
#include <string>
//#include <types.h> // defines time_t
//#include <iostream>

using namespace std;


//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//#include "PSTime/Duration.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSTime {

/**
 *  C++ source file code template. The first sentence is a brief summary of 
 *  what the class is for. It is followed by more detailed information
 *  about how to use the class. This doc comment must immediately preceed the 
 *  class definition.
 *
 *  Additional paragraphs with more details may follow; separate paragraphs
 *  with a blank line. The last paragraph before the tags (preceded by @) 
 *  should be the identification and copyright, as below.
 *
 *  Please note that KDOC comments must start with (a forward slash
 *  followed by TWO asterisks). Interface members should be documented
 *  with KDOC comments, as should be protected members that may be of interest
 *  to those deriving from your class. Private implementation should
 *  be commented with C++-style // (double forward slash) comments.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class Time  {
public:

  enum Zone { UTC, Local, PST };

  // Default constructor
  Time ();

  // Copy constructor 
  Time (const Time& t);

  Time (time_t sec_since_1970_01_01, time_t nsec = 0);

  Time (int year,
        int month,
        int day,
        int hour,
        int min,
        int sec,
        int nsec = 0,
        Zone zone = Local);

  Time (struct timespec& ts, Zone zone = UTC); // Constructs from POSIX/UNIX high-resolution time structure

  Time (struct tm& tms, Zone zone = UTC); // Constructs from POSIX/UNIX "broken-down" time structure

  //  Time (const std::string& date, const std::string& time, Zone zone = UTC); // from formatted date and time

  //  Time (const std::string&, Zone zone = UTC); // from formatted time stamp


  // Destructor
  virtual ~Time ();

  // Operators
  // Assignment operator

  Time& operator = ( const Time& t );

  // Arithmetic operators with Duration

  /*
  Duration operator-(  const Time& t ) const;               

  Time&    operator+=( const Duration& d );
  Time     operator+(  const Duration& d ) const;

  Time&    operator-=( const Duration& d );
  Time     operator-(  const Duration& d ) const;
  */

  // Comparison operators

  bool operator!=( const Time& t ) const
    { 
      return ( m_utcSec != t.m_utcSec || m_utcNsec != t.m_utcNsec );
    }

  bool operator==( const Time& t ) const
    { 
      return !( *this != t );
    }

  bool operator<( const Time& t ) const
    { 
      return ( m_utcSec  < t.m_utcSec ) || 
	     ( m_utcSec == t.m_utcSec && m_utcNsec < t.m_utcNsec );
    }

  bool operator>( const Time& t ) const
    { 
      return ( m_utcSec  > t.m_utcSec ) || 
	     ( m_utcSec == t.m_utcSec && m_utcNsec > t.m_utcNsec );
    }

  bool operator>=( const Time& t ) const
    { 
      return !( *this < t );      
    }

  bool operator<=( const Time& t ) const
    { 
      return !( *this > t );      
    }

  // Methods

  void Print() const;
 
  void getTimeSpec( struct timespec& ts, Zone zone = UTC ) const;
  void gettm( struct tm& stm, Zone zone = UTC ) const;

  string strZoneHuman( Zone zone = UTC ) const;  // +HH:MM or Z for UTC
  string strZoneBasic( Zone zone = UTC ) const;  // +HHMM  or Z for UTC

  string strDateHuman( Zone zone = UTC ) const;  // YYYY-MM-DD
  string strDateBasic( Zone zone = UTC ) const;  // YYYYMMDD

  string strTimeHuman( Zone zone = UTC ) const;  // HH:MM:SS
  string strTimeBasic( Zone zone = UTC ) const;  // HHMMSS

  string strDateTimeFreeFormat( const char* format="%Y-%m-%d %H:%M:%S", Zone zone = UTC) const; // See help for strftime 

  string strNsec(int nsecPrecision) const;       // .NNNNN where the number of "N" is equal to nsecPrecision

  string strTimeStampHuman( Zone zone = UTC, int nsecPrecision = 9 ) const;  // YYYY-MM-DD HH:MM:SS.NNNNNNNNN+HH:MM
  string strTimeStampBasic( Zone zone = UTC, int nsecPrecision = 0 ) const;  // HHMMSSTHHMMSS+HHMM
  string strTimeStampFreeFormat( const char* fmt="%Y-%m-%d %H:%M:%S", Zone zone = UTC, int nsecPrecision = 0 ) const; // Using free format

  // Friends
  //friend Time  operator+( const PSDuration& d, const Time& t     );
  //friend Time  operator+( const Time& t,     const PSDuration& d );
  //friend std::ostream& operator<<( std::ostream& os, const Time& t );

//protected:

private:

  // Data members  // private members start with m_
  
  time_t m_utcSec;  // number of seconds since 00:00:00 Jan. 1, 1970 UTC
  time_t m_utcNsec; // number of nanoseconds

  //  struct timespec m_time;    
  //  struct tm       m_timeinfo;



//------------------
// Static Members --
//------------------

public:

  // Selectors (const)

  time_t getUTCSec()  const {return m_utcSec;}  // POSIX sec. since  00:00:00 Jan. 1, 1970 UTC
  time_t getUTCNsec() const {return m_utcNsec;}

  time_t dtSecLocToUTC() const;

  // Modifiers

  // Methods

  static Time getTimeNow();

  static time_t getZoneTimeOffsetSec(Zone zone);

  static time_t getLocalZoneTimeOffsetSec();
  static time_t getPSTZoneTimeOffsetSec();

  static void getZoneTimeOffset(Zone zone, int &zoneHour, int &zoneMin);

  //  static bool parseTime( const std::string& sdate, const std::string& stime, 
  //                         Zone zone,
  //                         Time& time );

  //  static bool parseTime( const std::string& sdatetime, 
  //                       Zone zone,
  //                       Time& time );



private:

  static time_t evaluateLocalZoneTimeOffsetSec();

  // Data members
  // static int s_staticVariable;     // Static data member starts with s_.

}; // class Time

}  // namespace Time

#endif // PSTIME_TIME_H
