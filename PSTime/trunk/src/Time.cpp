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
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "PSTime/Time.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//using std::ostream;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSTime {

//----------------
// Constructors --
//----------------

Time::Time () : m_utcSec(0), m_utcNsec(0) // default
{
  struct timespec ts;
  int status = clock_gettime( CLOCK_REALTIME, &ts ); // Get LOCAL time

  if( status == 0 ){
    m_utcSec  = ts.tv_sec - getLocalZoneTimeOffsetSec(); 
    m_utcNsec = ts.tv_nsec;
  }
}

Time::Time (const Time& t) : m_utcSec(t.m_utcSec), m_utcNsec(t.m_utcNsec)
{
}

Time::Time (time_t utcSec, time_t utcNsec) : m_utcSec(utcSec), m_utcNsec(utcNsec)
{
}

Time::Time ( int  year,
             int  month,
             int  day,
             int  hour,
             int  min,
             int  sec,
             int  nsec,
             Zone zone)
{
  struct tm stm;
  stm.tm_year  = year - 1900; // Human: 2010,   tm year from: 1900
  stm.tm_mon   = month - 1;   // Human: [1,12], tm and POSIX: [0,11]
  stm.tm_mday  = day; 
  stm.tm_hour  = hour;
  stm.tm_min   = min;
  stm.tm_sec   = sec;
  stm.tm_isdst = -1;          // Let mktime figure out whether DST is in force

  m_utcSec  = mktime( &stm ) - getZoneTimeOffsetSec(zone);
  m_utcNsec = nsec;
}

Time::Time (struct timespec& ts, Zone zone) // HR-time ts is defined for zone (UTC as default)
{
  m_utcSec  = ts.tv_sec - getZoneTimeOffsetSec(zone);
  m_utcNsec = ts.tv_nsec;
}

Time::Time (struct tm& stm, Zone zone) : m_utcNsec(0) // stm is defined for zone (UTC as default)
{
  m_utcSec  = mktime( &stm ) - getZoneTimeOffsetSec(zone);
  // nsec are ignored for low resolution tm structure ...
}

//--------------
// Destructor --
//--------------
Time::~Time ()
{
  
}

//--------------
//  -- Operators
//--------------

// t2 = t1
Time& Time::operator = ( const Time& t1 )
{
  //if ( this == &t1 ) return *this; // it should be faster not doing this check
  m_utcSec  = t1.m_utcSec;
  m_utcNsec = t1.m_utcNsec;
  return *this;
}


//--------------
//  -- Public methods
//--------------

void Time::Print() const
  {
    printf ( "Time:: m_utcSec = %ld   m_utcNsec = %ld \n", m_utcSec, m_utcNsec);
    struct tm *stm = localtime ( &m_utcSec );  // localtime for UTC seconds gives UTC time...
    printf ( "UTC time (asctime) : %s", asctime (stm) );
  //printf ( "UTC time (ctime)   : %s\n", ctime (&m_utcSec) ); 
  }


void Time::getTimeSpec( struct timespec& ts, Zone zone ) const
{
  ts.tv_sec  = m_utcSec + getZoneTimeOffsetSec(zone);
  ts.tv_nsec = m_utcNsec;
}

void Time::gettm( struct tm& stm, Zone zone ) const
{
  // nsec are ignored for low resolution tm structure ...
  time_t secondsInZone = m_utcSec + getZoneTimeOffsetSec(zone);
  stm =*localtime(&secondsInZone);  // convert local to local, store as stm
}


// Zone in format: +HH:MM or Z for UTC 
string Time::strZoneHuman( Zone zone ) const
{
  string strZone = "Z";
  if(zone != UTC) 
    { 
      int    HH, MM;
      getZoneTimeOffset(zone,HH,MM);
      char   charZone[8];
      int    len = sprintf(charZone,"%+03d:%02d",HH,MM);
      strZone = (len==6) ? charZone : "Zone?";
    }
  return strZone;
}


// Zone in format: +HHMM or Z for UTC 
string Time::strZoneBasic( Zone zone ) const
{
  string strZone = "Z";
  if(zone != UTC) 
    {
      int    HH, MM;
      getZoneTimeOffset(zone,HH,MM);
      char   charZone[8];
      int    len = sprintf(charZone,"%+03d%02d",HH,MM);
      strZone = (len==5) ? charZone : "Zone?";
    }
  return strZone;
}


// Date in format: YYYY-MM-DD 
string Time::strDateHuman( Zone zone ) const
{
  struct tm stm;
  gettm(stm, zone);
  string strDate = "Date Error";
  char   charDate[16];
  size_t maxsize =16;
  int len = strftime(charDate, maxsize, "%Y-%m-%d", &stm);
  if (len==10) { strDate = charDate; }
  else         { printf("\n  Time::strDateHuman len = %d\n", len);}
  return strDate;
}


// Date in format: YYYYMMDD 
string Time::strDateBasic( Zone zone ) const
{
  struct tm stm;
  gettm(stm, zone);
  string strDate = "Date Error";
  char   charDate[16];
  size_t maxsize =16;
  int len = strftime(charDate, maxsize, "%Y%m%d", &stm);
  if (len==8) strDate = charDate; 
  else        printf("\n  Time::strDateBasic len = %d\n", len); 
  return strDate;
}


// Time in format: HH:MM:SS 
string Time::strTimeHuman( Zone zone ) const
{
  struct tm stm;
  gettm(stm, zone);
  string strTime = "Time Error";
  char   charTime[16];
  size_t maxsize =16;
  int len = strftime(charTime, maxsize, "%H:%M:%S", &stm);
  if (len==8) strTime = charTime; 
  else        printf("\n  Time::strTimeHuman len = %d\n", len); 
  return strTime;
}


// Time in format: HHMMSS 
string Time::strTimeBasic( Zone zone ) const
{
  struct tm stm;
  gettm(stm, zone);
  string strTime = "Time Error";
  char   charTime[16];
  size_t maxsize =16;
  int len = strftime(charTime, maxsize, "%H%M%S", &stm);
  if (len==6) strTime = charTime; 
  else        printf("\n  Time::strTimeBasic len = %d\n", len); 
  return strTime;
}


// Date and Time in free format 
string Time::strDateTimeFreeFormat( const char* format, Zone zone ) const
{
  struct tm stm;
  gettm(stm, zone);
  string strDateTime = "DateTime Error";
  char   charDateTime[255];
  size_t maxsize =255;
  int len = strftime(charDateTime, maxsize, format, &stm);
  if (len>0) strDateTime = charDateTime; 
  else        printf("\n  Time::strDateTimeFreeFormat len = %d\n", len); 
  return strDateTime;
}


// Nsec in format of decimal fraction of sec : .NNNNN from .0 to .999999999
string Time::strNsec(int nsecPrecision) const
{
  int len = (nsecPrecision>9) ? 9 : nsecPrecision;
  if(nsecPrecision<1) {string strnsec = ""; return strnsec;} // return empty string

  string strnsec = "nsec Error";  
  char   nsecFormat[10];
  sprintf(nsecFormat,"%%%-d.%df",len+2,len); // should be something like "%-7.6f"
  double nsecInSec = (double)m_utcNsec;
         nsecInSec *= 1e-9; 
	 //printf("\n Time::strNsec format for nsec : %s\n", nsecFormat); 
  char   charNsec[16];
  len = sprintf(charNsec,nsecFormat,nsecInSec);
         strnsec = (len>0) ? &charNsec[1] : "nsec?"; // [1] - do not return "0" preceeding the dot
  return strnsec;
}


// Time stamp in fromat YYYY-MM-DD HH:MM:SS.NNNNNNNNN+HH:MM
string Time::strTimeStampHuman( Zone zone, int nsecPrecision ) const
{
  return strDateHuman(zone) + " " + strTimeHuman(zone) + strNsec(nsecPrecision) + strZoneHuman(zone);
}


// Time stamp in fromat HHMMSSTHHMMSS+HHMM
string Time::strTimeStampBasic( Zone zone, int nsecPrecision  ) const
{
  return strDateBasic(zone) + "T" + strTimeBasic(zone) + strNsec(nsecPrecision) + strZoneBasic(zone);
}


// Time stamp in free fromat
string Time::strTimeStampFreeFormat( const char* fmt, Zone zone, int nsecPrecision  ) const
{
  return strDateTimeFreeFormat(fmt,zone) + strNsec(nsecPrecision) + strZoneBasic(zone);
}


Time Time::getTimeNow()  // update time in object
{
    struct timespec ts;
    int gettimeStatus = clock_gettime(CLOCK_REALTIME, &ts);
     if(gettimeStatus != 0) 
      {
        return Time(0,0); // In case of 
      }
    else
      {
        return Time(ts.tv_sec - getLocalZoneTimeOffsetSec(), ts.tv_nsec);
      }
}


time_t Time::getZoneTimeOffsetSec(Zone zone)
{
    switch  (zone)
      {
        case Local : {return getLocalZoneTimeOffsetSec();} // Local zone offset w.r.t. UTC
        case UTC   : {return 0;}                           // 0-offset for UTC 
        case PST   : {return getPSTZoneTimeOffsetSec();}   // PST zone offset w.r.t. UTC  
        default    : {return 0;}                           // 0-offset for UTC 
      } 
}


time_t Time::getPSTZoneTimeOffsetSec()
{
    static time_t PSTZoneTimeOffsetSec = -8 * 3600; 
    return        PSTZoneTimeOffsetSec;
}


time_t Time::getLocalZoneTimeOffsetSec()
{
    // Initialization for static is done at 1st call
    static time_t localZoneTimeOffsetSec = Time::evaluateLocalZoneTimeOffsetSec(); 
    return        localZoneTimeOffsetSec;
}


// Converts zone (sec) to zone in hours and minutes
void Time::getZoneTimeOffset(Zone zone, int &zoneHour, int &zoneMin)
{
      time_t      zoneTimeOffsetSec = getZoneTimeOffsetSec(zone);
      zoneHour =  zoneTimeOffsetSec/3600;
      zoneMin  = abs(zoneTimeOffsetSec%3600)/60;
}


//--------------
//  -- Private methods
//--------------


time_t Time::evaluateLocalZoneTimeOffsetSec()
{
    time_t locSec =time(NULL);             // get current local time_t time
    tm     utcTime=*gmtime(&locSec);       // convert local time_t to GMT, store as tm
    time_t utcSec =(mktime(&utcTime));     // convert GMT tm to GMT time_t
    time_t offset = locSec - utcSec;       // difference between local and UTC seconds
    printf("Time::evaluateLocalZoneTimeOffsetSec() : Local zone time offset w.r.t. UTC = %d h.\n", (int)offset/3600);
    return offset;
}
//--------------
} // namespace PSTime


