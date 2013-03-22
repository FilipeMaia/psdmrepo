
//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PSTime...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSTime/Time.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSTime/Exceptions.h"
#include "PSTime/TimeFormat.h"
#include "PSTime/TimeUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // number of nanoseconds in a second
  const uint32_t nsecInASec = 1000000000U;
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSTime {

//----------------
// Constructors --
//----------------

Time::Time () 
  : m_utcSec(0)
  , m_utcNsec(0)
{
}

Time::Time (time_t utcSec, uint32_t utcNsec)
  : m_utcSec(utcSec)
  , m_utcNsec(utcNsec)
{
}

Time::Time ( int  year,
             int  month,
             int  day,
             int  hour,
             int  min,
             int  sec,
             uint32_t  nsec,
             Zone zone)
  : m_utcSec(0)
  , m_utcNsec(nsec)
{
  struct tm stm;
  stm.tm_year  = year - 1900; // Human: 2010,   tm year from: 1900
  stm.tm_mon   = month - 1;   // Human: [1,12], tm and POSIX: [0,11]
  stm.tm_mday  = day; 
  stm.tm_hour  = hour;
  stm.tm_min   = min;
  stm.tm_sec   = sec;
  stm.tm_isdst = -1;          // Let mktime figure out whether DST is in force

  if (zone == UTC) {
    m_utcSec  = TimeUtils::timegm( &stm );
  } else {
    m_utcSec  = mktime( &stm );
  }
}

Time::Time (struct timespec& ts)
{
  m_utcSec  = ts.tv_sec;
  m_utcNsec = ts.tv_nsec;
}

Time::Time (struct tm& stm, Zone zone) 
  : m_utcSec(0)
  , m_utcNsec(0)
{
  if (zone == UTC) {
    m_utcSec  = TimeUtils::timegm( &stm );
  } else {
    m_utcSec  = mktime( &stm );
  }
}

// tstamp can be presented in formats understood by in parse(...) method
Time::Time (const std::string& tstamp)
  : m_utcSec(0)
  , m_utcNsec(0)
{
  Time t = TimeFormat::parseTime(tstamp);
  m_utcSec  = t.m_utcSec; // or t.getUTCSec();
  m_utcNsec = t.m_utcNsec; // or t.getUTCNsec();
}

//--------------
//  -- Operators
//--------------

// Arithmetic operators with Duration

// d = |t2 - t1|
Duration 
Time::operator-( const Time& t1 ) const
{
    // This code forms |t2-t1| without having to use signed intergers.

    time_t t2Sec;
    time_t t2Nsec;
    time_t t1Sec;
    time_t t1Nsec;

    if ( *this > t1 )
        {
            t2Sec  = m_utcSec;
            t2Nsec = m_utcNsec;
            t1Sec  = t1.m_utcSec;
            t1Nsec = t1.m_utcNsec;
        }
    else
        {
            t2Sec  = t1.m_utcSec;
            t2Nsec = t1.m_utcNsec;
            t1Sec  = m_utcSec;
            t1Nsec = m_utcNsec;
        }

    if ( t2Nsec < t1Nsec )
        {
            // borrow a second from t2Sec
            t2Nsec += ::nsecInASec;
            t2Sec--;
        }

    time_t sec  = t2Sec  - t1Sec;
    time_t nsec = t2Nsec - t1Nsec;

    Duration diff( sec, nsec );

    return diff;
}



// t -= d
Time& 
Time::operator-=( const Duration& d )
{
  // if t1 - d < 0 then return t = 0
  if ( ( m_utcSec  < d.getSec() ) ||
       ( m_utcSec == d.getSec()   && m_utcNsec < d.getNsec() ) ) {
    m_utcSec  = 0;
    m_utcNsec = 0;
  }
  else {
    time_t tempSec  = m_utcSec;
    time_t tempNsec = m_utcNsec;
    
    if ( tempNsec < d.getNsec() ) {
      // if t1.m_utcNsec < d._nsec borrow a second from t1.m_utcSec
      tempNsec += ::nsecInASec;
      tempSec--;
    }
    
    m_utcSec  = tempSec  - d.getSec();
    m_utcNsec = tempNsec - d.getNsec();
  }
  
  return *this;
}

// t += d
Time& 
Time::operator+=( const Duration& d )
{
  time_t totalSec  = m_utcSec  + d.getSec();
  time_t totalNsec = m_utcNsec + d.getNsec();
  
  if ( totalNsec >= ::nsecInASec ) {
    // carry nanoseconds over into seconds
    time_t extraSec   = totalNsec / ::nsecInASec;
    time_t remainNsec = totalNsec % ::nsecInASec;
    totalSec         += extraSec;
    totalNsec         = remainNsec;
  }
  
  m_utcSec  = totalSec;
  m_utcNsec = totalNsec;

  return *this;
}


//--------------------
//  -- Public methods
//--------------------

struct timespec
Time::getTimeSpec() const
{
  struct timespec ts;
  ts.tv_sec  = m_utcSec;
  ts.tv_nsec = m_utcNsec;
  return ts;
}

struct tm
Time::gettm( Zone zone ) const
{
  struct tm stm;
  memset( &stm, 0, sizeof(stm)) ;

  if (zone == UTC) {
    gmtime_r(&m_utcSec, &stm);
  } else {
    localtime_r(&m_utcSec, &stm);    
  }
  
  return stm;
}

// Time stamp in fromat YYYY-MM-DD HH:MM:SS.NNNNNNNNN+HH:MM
std::string 
Time::asString( Zone zone, int nsecPrecision ) const
{
  std::string fmt;
  if (nsecPrecision < 1) {
    fmt = "%F %T%z";
  } else {
    if (nsecPrecision > 9) nsecPrecision = 9;
    fmt = "%F %T%.9f%z";
    fmt[7] = char('0'+nsecPrecision);
  }
  return TimeFormat::format(*this, fmt, zone);
}


// Time stamp in fromat HHMMSSTHHMMSS+HHMM
std::string 
Time::asStringCompact( Zone zone, int nsecPrecision  ) const
{
  std::string fmt;
  if (nsecPrecision < 1) {
    fmt = "%Y%m%dT%H%M%S%z";
  } else {
    if (nsecPrecision > 9) nsecPrecision = 9;
    fmt = "%Y%m%dT%H%M%S%.9f%z";
    fmt[15] = char('0'+nsecPrecision);
  }
  return TimeFormat::format(*this, fmt, zone);
}


// Time stamp in free fromat
std::string 
Time::asStringFormat( const std::string& fmt, Zone zone ) const
{
  return TimeFormat::format(*this, fmt, zone);
}

/**
 *  Returns current time.
 */
Time 
Time::now() 
{
  struct timespec ts;
  int status = clock_gettime( CLOCK_REALTIME, &ts );

  if( status == 0 ) {
    return Time(ts.tv_sec, ts.tv_nsec);
  } else {
    throw ErrnoException( ERR_LOC, "clock_gettime failed");
  }
  
}

std::ostream& 
operator <<( std::ostream& os, const Time& t ) 
{
  TimeFormat::format(os, t, "%F %T%f%z", Time::Local);
  return os;
}

//--------------
} // namespace PSTime

