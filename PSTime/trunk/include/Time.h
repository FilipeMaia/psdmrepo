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
#include <stdio.h>  // for printf 
#include <iostream> // for cout
#include <string>
//#include <types.h> // defines time_t

using namespace std;


//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "PSTime/Duration.h"

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

  enum ParseStatus {
    PARSE_IS_OK,
    TOO_SHORT_TIME_STAMP,
    TOO_LONG_TIME_STAMP,
    WRONG_DATE_FORMAT_A,
    WRONG_DATE_FORMAT_B,
    WRONG_DATE_FORMAT_C,
    TOO_SHORT_TIME_RECORD,
    TOO_LONG_TIME_RECORD,
    WRONG_TIME_FORMAT_A,
    WRONG_TIME_FORMAT_B,
    WRONG_SEC_FRACTION_FORMAT,
    TOO_SHORT_ZONE_RECORD,
    TOO_LONG_ZONE_RECORD,
    WRONG_ZONE_FORMAT_A,
    WRONG_ZONE_FORMAT_B,
    WRONG_ZONE_FORMAT_C
  };

/** Default constructor */
  Time ();

/** Copy constructor */
  Time (const Time& t);


/** 
 *  Constructs a time from an unsigned number of seconds since the
 *  the Unix epoch of 1970-01-01; nanoseconds are optional, if needed.
 */
  Time (time_t sec_since_1970_01_01, time_t nsec = 0);

/** 
 * Constructs a time from human-undestandable numbers;
 * year  - is the calendar year C.E. (e.g., 2010),
 * month - is the months in the range [1,12],
 * day   - is the day of the month in the range [1,31],
 * hour  - is the hours in the range [0,23],
 * min   - is the minutes after the hour in the range [0,59],
 * sec   - is the seconds in the range [0,59*] (*) might be up to 61 for leap seconds, 
 * nsec  - is the nanoseconds after the second in the range [0,999999999].
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
 * Constructs a time from the timespec struct (for high resolution time) and zone.
 * Zone can be Time::UTC (Universal Time Coordinated, a.k.a. Greenvitch meridian time), 
 *             Time::PST (Pasific Standard Time) for SLAC, 
 *             Time::Local - for local computer time zone. 
 */
  Time (struct timespec& ts, Zone zone = UTC); // Constructs from POSIX/UNIX high-resolution time structure

/** 
 * Constructs a time from the tm struct (for low resolution time) and zone.
 * Zone can be Time::UTC (Universal Time Coordinated, a.k.a. Greenvitch meridian time), 
 *             Time::PST (Pasific Standard Time) for SLAC, 
 *             Time::Local - for local computer time zone. 
 */
  Time (struct tm& tms, Zone zone = UTC); // Constructs from POSIX/UNIX "broken-down" time structure

  // Do we need this? Next constructor should cover all needs
  //Time (const std::string& date, const std::string& time, Zone zone = UTC); // from formatted date and time

/** 
 * Constructs a time from the string time stamp in format:
 * <date> <time>[.<fraction-of-sec>][<time-zone>]
 *   or
 * <date>T<time>[.<fraction-of-sec>][<time-zone>]
 * where 
 * <date> should be in format YYYY-MM-DD or YYYYMMDD
 * <time> should be in format HH:MM:SS   or HHMMSS
 * <fraction-of-sec> may have up to 9 digits for nsec; 
 *                   if this field is missing, it is assumed equal to 0. 
 *                   the dot-separator '.' should not be used without following digit(s)
 * <time> should be Z for UTC or in format <sign + or ->HH[:MM] or <sign + or ->HHMM
 *        if this field is missing, it is assumed equal to 0 for UTC  
 */
  Time (const std::string&); // from formatted time stamp


/** Destructor */
  virtual ~Time ();

  // Operators
/** Assignment operator */
  Time& operator = ( const Time& t );

/** Arithmetic operators with Duration */

  Duration operator-(  const Time& t ) const;               

  Time&    operator+=( const Duration& d );
  Time     operator+(  const Duration& d ) const;

  Time&    operator-=( const Duration& d );
  Time     operator-(  const Duration& d ) const;

/** Comparison operators returns bool-ean true or false statement. */

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

/** Prints entity of the Time object and human-readable time presentation in UTC */
  void Print() const;
 
/** Returns the Time object contetnt via high-resolution time timespec structure for indicated zone.*/
  void getTimeSpec( struct timespec& ts, Zone zone = UTC ) const;

/** Returns the Time object contetnt via low-resolution time tm structure for indicated zone.*/
  void gettm( struct tm& stm, Zone zone = UTC ) const;

/** Methods below return content of the Time object 
 *  for date, time, zone, date and time, nanoseconds, and the time stamp 
 *  as a 'Human', 'Basic', or specifically formatted string for indicated zone.
 */ 
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
/** Operators between Time and Duration objects */
// t = d + t1 complementary for earlier defined t = t1 + d 
  friend Time  operator+( const Duration& d, const Time& t1 );
  friend std::ostream& operator<<( std::ostream& os, const Time& t );

//protected:

private:

// private members start with m_ 
/** Data members:
 * number of seconds since 00:00:00 Jan. 1, 1970 UTC
 * number of nanoseconds
 */ 
  time_t m_utcSec; 
  time_t m_utcNsec;

  //  struct timespec m_time;    
  //  struct tm       m_timeinfo;


//------------------
// Static Members --
//------------------

public:

  // Selectors (const)
/** Access methods to get member data. */
  time_t getUTCSec()  const {return m_utcSec;}  // POSIX sec. since  00:00:00 Jan. 1, 1970 UTC
  time_t getUTCNsec() const {return m_utcNsec;}


  time_t dtSecLocToUTC() const;

  // Modifiers

  // Methods

/** Updates the time entity of the object. */
  static Time getTimeNow();


/** Returns the time offset in seconds for specific zone.*/
  static time_t getZoneTimeOffsetSec(Zone zone);
  static time_t getLocalZoneTimeOffsetSec();
  static time_t getPSTZoneTimeOffsetSec();

/** Converts zone time offset for specified zone (in secconds) 
 *  to the zone time offset in hours and minutes
 */
  static void getZoneTimeOffset(Zone zone, int &zoneHour, int &zoneMin);

  //  static int parseTime( const std::string& sdate, const std::string& stime, 
  //                        Zone zone,
  //                        Time& time );

/** The parsification engine, 
 *  see description of the Time (const std::string&) constructor.
 */ 
  static int parseTimeStamp( const std::string& sdatetime, Time& time );


private:

/** Evaluates and returns the time offset in seconds for local zone w.r.t. UTC. 
 *  This evaluation is called only once for static data.
 */
  static time_t evaluateLocalZoneTimeOffsetSec();

  // Data members
  // Static data member starts with s_.
  // static int s_staticVariable;     

}; // class Time

}  // namespace Time

#endif // PSTIME_TIME_H
