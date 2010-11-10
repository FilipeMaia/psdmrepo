
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
#include "PSTime/TimeConstants.h"

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

// tstamp can be presented in formats described in parseTimeStamp(...) method
Time::Time (const std::string& tstamp) : m_utcSec(0), m_utcNsec(0) // from formatted time stamp
{
  Time t(0,0);
  int status = parseTimeStamp(tstamp, t);
  if( status == PARSE_IS_OK ) {
    m_utcSec  = t.m_utcSec; // or t.getUTCSec();
    m_utcNsec = t.m_utcNsec; // or t.getUTCNsec();
  }
  // else leave time of the Christ birth = (0,0)
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

// Arithmetic operators with Duration

// d = |t2 - t1|
Duration Time::operator-( const Time& t1 ) const
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
            t2Nsec += TimeConstants::s_nsecInASec;
            t2Sec--;
        }

    time_t sec  = t2Sec  - t1Sec;
    time_t nsec = t2Nsec - t1Nsec;

    Duration diff( sec, nsec );

    return diff;
}



// t = t1 - d
Time Time::operator-( const Duration& d ) const
{
  return Time(*this) -= d;
}

// t -= d
Time& Time::operator-=( const Duration& d )
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
      tempNsec += TimeConstants::s_nsecInASec;
      tempSec--;
    }
    
    m_utcSec  = tempSec  - d.getSec();
    m_utcNsec = tempNsec - d.getNsec();
  }
  
  return *this;
}

// t = t1 + d
Time Time::operator+( const Duration& d ) const
{
  return Time(*this) += d;
}

// t += d
Time& Time::operator+=( const Duration& d )
{
  time_t totalSec  = m_utcSec  + d.getSec();
  time_t totalNsec = m_utcNsec + d.getNsec();
  
  if ( totalNsec >= TimeConstants::s_nsecInASec ) {
    // carry nanoseconds over into seconds
    time_t extraSec   = totalNsec / TimeConstants::s_nsecInASec;
    time_t remainNsec = totalNsec % TimeConstants::s_nsecInASec;
    totalSec         += extraSec;
    totalNsec         = remainNsec;
  }
  
  m_utcSec  = totalSec;
  m_utcNsec = totalNsec;

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


// Accepts the time stamp in format:
// <date> <time>[.<fraction-of-sec>][<time-zone>]
//   or
// <date>T<time>[.<fraction-of-sec>][<time-zone>]
// where 
// <date> should be in format YYYY-MM-DD or YYYYMMDD
// <time> should be in format HH:MM:SS   or HHMMSS
// <fraction-of-sec> may have up to 9 digits for nsec; 
//                   if this field is missing, it is assumed equal to 0. 
// <time> should be Z for UTC or in format <sign + or ->HH[:MM] or <sign + or ->HHMM
//        if this field is missing, it is assumed equal to 0 for UTC
//
int Time::parseTimeStamp(const std::string& tstamp, Time& time_from_tstamp)
{
//bool printForTest = true;
  bool printForTest = false;

  struct tm tm_tstamp;
  struct tm tm_zone;

  char char_date[16]; size_t len_date=0;
  char char_time[16]; size_t len_time=0;
  char char_nsec[16]; size_t len_nsec=0;
  char char_zone[16]; size_t len_zone=0;

  time_t zoneTimeOffsetSec  = 0;

  if(printForTest)
  printf ( "\n\n\nParse the time stamp %s\n", tstamp.data());

  // Check time stamp size
  size_t len_time_stamp = tstamp.size();
  if( len_time_stamp < 15 ) {return TOO_SHORT_TIME_STAMP;}
  if( len_time_stamp > 35 ) {return TOO_LONG_TIME_STAMP;}

  // Find position of the <date> and <time> separator field 'T' or ' ' (space)
    size_t pos_time_sep = tstamp.find(' ');
      if ( pos_time_sep ==string::npos ) 
           pos_time_sep = tstamp.find('T');
      if ( pos_time_sep ==string::npos ) return WRONG_DATE_FORMAT_A;

  // Find position of the '.' (dot) before the second fraction
    size_t pos_nsec_sep = tstamp.find('.',pos_time_sep);

  // Find position of the time zone record, the sign '+','-' or 'Z'
    size_t pos_zone_sep = tstamp.find('Z');

      if ( pos_zone_sep ==string::npos ) 
	{
	   pos_zone_sep = tstamp.find('+',pos_time_sep);
	}
      if ( pos_zone_sep ==string::npos )
        {
           pos_zone_sep = tstamp.find('-',pos_time_sep); 
        }

      if(printForTest)
      cout << "Length of the time stamp:" << (int)len_time_stamp
           << " Position of sep. : " 
	   << " time:" << (int)pos_time_sep 
	   << " nsec:" << (int)pos_nsec_sep 
	   << " zone:" << (int)pos_zone_sep 
           << endl;

  // further assume that pos separates date and time
  // Parse <date> and fill the structure tm_tstamp

      if ( pos_time_sep == 8 ){       // for YYYYMMDD
        len_date=tstamp.copy(char_date,8,0); char_date[len_date] = '\0';
	if( strptime(char_date,"%Y%m%d",&tm_tstamp) == NULL) return WRONG_DATE_FORMAT_B;        
      }
      else if ( pos_time_sep == 10 ){ // for YYYY-MM-DD
        len_date=tstamp.copy(char_date,10,0); char_date[len_date] = '\0';
	if( strptime(char_date,"%Y-%m-%d",&tm_tstamp) == NULL) return WRONG_DATE_FORMAT_C;        
      }

      if(printForTest)
      cout << "char_date : " << char_date << endl;

  // Parse <time-zone>
      if ( pos_zone_sep ==string::npos   // If There is no time-zone info
        || tstamp[pos_zone_sep] == 'Z' ) // or UTC time zone, do nothing
	{
           char_zone[0] = 'Z';
           char_zone[1] = '\0';
           zoneTimeOffsetSec = 0;
	}
      else // parse <zone> record
	{
          len_zone = len_time_stamp - 1 - pos_zone_sep;

          tm_zone.tm_hour = 0;
	  tm_zone.tm_min  = 0;
          tm_zone.tm_isdst = 0;

          if(len_zone < 2)                                    return TOO_SHORT_ZONE_RECORD;
          else if(len_zone == 2){
	     len_zone = tstamp.copy(char_zone,2,pos_zone_sep+1); char_zone[len_zone] = '\0';
             if( strptime(char_zone,"%H",&tm_zone) == NULL)   return WRONG_ZONE_FORMAT_A;
	  }
          else if(len_zone == 4){
	    len_zone = tstamp.copy(char_zone,4,pos_zone_sep+1); char_zone[len_zone] = '\0';
	    if( strptime(char_zone,"%H%M",&tm_zone) == NULL)  return WRONG_ZONE_FORMAT_B;
          }
          else if(len_zone == 5){
            len_zone = tstamp.copy(char_zone,5,pos_zone_sep+1); char_zone[len_zone] = '\0';
            if( strptime(char_zone,"%H:%M",&tm_zone) == NULL) return WRONG_ZONE_FORMAT_C;
          }
          else                                                return TOO_LONG_ZONE_RECORD;

                    zoneTimeOffsetSec = tm_zone.tm_hour * 3600
	    	                      + tm_zone.tm_min  * 60; 

          if(tstamp[pos_zone_sep] == '-') zoneTimeOffsetSec = -zoneTimeOffsetSec;
        }

      if(printForTest){
      if ( pos_zone_sep != string::npos ) cout << "char_zone : " << char_zone;
      cout << " Zone time offset in sec = " << zoneTimeOffsetSec << endl;
      }

  // Parse fraction of second      
     if ( pos_nsec_sep==string::npos ) // There is no nsec in the time stamp...
	{
          time_from_tstamp.m_utcNsec = 0;
	}
      else // parese nsec
	{         
	    len_nsec = ( pos_zone_sep==string::npos ) ?     // If there is no zone info
	               len_time_stamp - pos_nsec_sep  :     
           	       pos_zone_sep   - pos_nsec_sep;       // lendth includes the forward '.' (dot)

	    len_nsec = tstamp.copy(char_nsec,len_nsec,pos_nsec_sep); char_nsec[len_nsec] = '\0';
	    double fraction_of_sec = atof(char_nsec);
            time_from_tstamp.m_utcNsec = (time_t)(fraction_of_sec * 1e9);
	}

     if(printForTest){
     if ( pos_nsec_sep != string::npos ) cout << "char_nsec : " << char_nsec;
     cout << " time_from_tstamp.m_utcNsec : " << time_from_tstamp.m_utcNsec << endl;
     }

  // Parse <time>
          if(pos_nsec_sep != string::npos) len_time = pos_nsec_sep - pos_time_sep - 1; // if <second-fraction> is in record
     else if(pos_zone_sep != string::npos) len_time = pos_zone_sep - pos_time_sep - 1; // if <zone> is in record
     else                                  len_time = len_time_stamp - pos_time_sep - 1; 

          if( len_time < 6)                                 return TOO_SHORT_TIME_RECORD;
     else if( len_time ==6){
       len_time = tstamp.copy(char_time,6,pos_time_sep+1); char_time[len_time] = '\0';
       if( strptime(char_time,"%H%M%S",&tm_tstamp) == NULL)   return WRONG_TIME_FORMAT_A;
     }
     else if(len_time == 8){
       len_time = tstamp.copy(char_time,8,pos_time_sep+1); char_time[len_time] = '\0';
       if( strptime(char_time,"%H:%M:%S",&tm_tstamp) == NULL) return WRONG_TIME_FORMAT_B;
     }
     else                                                   return TOO_LONG_TIME_RECORD;

     if(printForTest)
     cout << "char_time : " << char_time << endl;

  // Return sparsed time stamp in seconds
     time_from_tstamp.m_utcSec  = mktime( &tm_tstamp ) - zoneTimeOffsetSec;

     return PARSE_IS_OK;
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



//--------------------
// Friend functions --
//--------------------

// t = d + t1
Time operator+( const Duration& d, const Time& t1 )
{
    return t1 + d;
}

// t = t1 + d
//Time operator+( const Time& t1, const Duration& d )
//{
//    return t1 + d;
//}

ostream & operator <<( ostream& os, const Time& t ) 
{
  if ( t == TimeConstants::s_minusInfinity )
    os << "-Infinity";
  else if ( t == TimeConstants::s_plusInfinity )
    os << "+Infinity";
  else 
    {
      os << t.strTimeStampFreeFormat( "%c", Time::Local ) << " (local time) " 
         << t.getUTCNsec() << " ns";
    }
  return os;
}

//--------------
} // namespace PSTime


