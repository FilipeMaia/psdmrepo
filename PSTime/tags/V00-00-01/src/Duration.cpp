//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Duration...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "PSTime/Duration.h"
#include "PSTime/TimeConstants.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <iomanip>
#include <math.h>

using std::cout;
using std::ostream;
using std::setw;


namespace PSTime {

//----------------
// Constructors --
//----------------
Duration::Duration () : m_sec( 0 ), m_nsec( 0 )
{
}


Duration::Duration( const Duration & d ) : m_sec( d.m_sec ), m_nsec( d.m_nsec )
{
}


Duration::Duration( time_t sec, time_t nsec ) 
{
    if ( nsec >= TimeConstants::s_nsecInASec )
        {
            // carry over nanoseconds into seconds
	    time_t extraSec   = nsec / TimeConstants::s_nsecInASec;
            time_t remainNsec = nsec % TimeConstants::s_nsecInASec;
            sec              += extraSec;
            nsec              = remainNsec;
        }

    m_sec  = sec;
    m_nsec = nsec;
}


Duration::Duration( time_t Years, 
                    time_t Days, 
                    time_t Hours, 
                    time_t Mins, 
	            time_t Secs,
                    time_t Nsecs )
{
  time_t SecsInYDHMS = Secs + 60*(Mins + 60*(Hours + 24*(Days + 364*Years)));
  Duration total( SecsInYDHMS, Nsecs );    
  m_sec  = total.m_sec;  
  m_nsec = total.m_nsec;
}


//-------------
// Operators --
//-------------

// d = |d2 - d1|
Duration Duration::operator - ( const Duration & d1 ) const
{
    // This code forms |d2-d1| without having to use signed intergers.

    time_t  d2Sec;
    time_t  d2Nsec;
    time_t  d1Sec;
    time_t  d1Nsec;
   
    if ( *this > d1 )
        {
            d2Sec  = m_sec;
            d2Nsec = m_nsec;
            d1Sec  = d1.m_sec;
            d1Nsec = d1.m_nsec;
        }
    else
        {
            d2Sec  = d1.m_sec;
            d2Nsec = d1.m_nsec;
            d1Sec  = m_sec;
            d1Nsec = m_nsec;
        }

    if ( d2Nsec < d1Nsec )
        {
            // borrow a second from d2Sec
            d2Nsec += TimeConstants::s_nsecInASec;
            d2Sec--;
        }

    time_t  sec  = d2Sec  - d1Sec;
    time_t  nsec = d2Nsec - d1Nsec;

    Duration diff( sec, nsec );
    return   diff;
}


// d = d2 + d1
Duration Duration::operator + ( const Duration & d1 ) const
{
    time_t totalSec  = m_sec  + d1.m_sec;
    time_t totalNsec = m_nsec + d1.m_nsec;

    if ( totalNsec >= TimeConstants::s_nsecInASec )
        {
            // carry nanoseconds over into seconds
            time_t extraSec   = totalNsec / TimeConstants::s_nsecInASec;
            time_t remainNsec = totalNsec % TimeConstants::s_nsecInASec;
            totalSec         += extraSec;
            totalNsec         = remainNsec;
        }

    Duration sum( totalSec, totalNsec );
    
    return sum;
}


// d2 = d1
Duration & Duration::operator = ( const Duration & d1 ) 
{
    m_sec  = d1.m_sec;
    m_nsec = d1.m_nsec;
    return *this;
}


// d2 += d1
Duration & Duration::operator += ( const Duration & d1 ) 
{
    time_t totalSec  = m_sec  + d1.m_sec;
    time_t totalNsec = m_nsec + d1.m_nsec;

    if ( totalNsec >= TimeConstants::s_nsecInASec )
        {
            // carry nanoseconds over into seconds
            time_t  extraSec   = totalNsec / TimeConstants::s_nsecInASec;
            time_t  remainNsec = totalNsec % TimeConstants::s_nsecInASec;
            totalSec          += extraSec;
            totalNsec          = remainNsec;
        }

    m_sec  = totalSec;
    m_nsec = totalNsec;    
    return *this;
}

//--------------------
//  -- Public methods
//--------------------

void Duration::Print() const
{
    printf ( "Duration:: m_sec = %ld   m_nsec = %ld \n", m_sec, m_nsec);

      time_t Years(0); 
      time_t DaysAfterY(0);
      time_t HoursAfterD(0); 
      time_t MinsAfterH(0); 
      time_t SecsAfterM(0);

    splitDurationSecsForYDHMS( Years, 
                             DaysAfterY, 
                             HoursAfterD, 
                             MinsAfterH, 
                             SecsAfterM );

    printf ( "Duration:  P%ldY%ldDT%dH%dM%dS  ",
	                     Years, 
                             DaysAfterY, 
                             (int)HoursAfterD, 
                             (int)MinsAfterH, 
                             (int)SecsAfterM );
    printf ( " or P%ldY %ldD T%dH %dM %dS  ",
	                     Years, 
                             DaysAfterY, 
                             (int)HoursAfterD, 
                             (int)MinsAfterH, 
                             (int)SecsAfterM );

    printf ( " or %s  \n", strDurationBasic().data() );
}


string Duration::strDurationBasic() const
{
      time_t Years(0); 
      time_t DaysAfterY(0);
      time_t HoursAfterD(0); 
      time_t MinsAfterH(0); 
      time_t SecsAfterM(0);

    splitDurationSecsForYDHMS( Years, 
                               DaysAfterY, 
                               HoursAfterD, 
                               MinsAfterH, 
                               SecsAfterM );
    time_t NsecAfterS = m_nsec;

    string strD = "P";

    if ( Years      != 0 ) { char charY[4];   sprintf( charY,"%ldY",Years );       strD += charY; }
    if ( DaysAfterY != 0 ) { char charD[2];   sprintf( charD,"%ldD",DaysAfterY );  strD += charD; }

           strD += "T";

    if ( HoursAfterD!= 0 ) { char charH[2];   sprintf( charH,"%ldH",HoursAfterD ); strD += charH; }
    if ( MinsAfterH != 0 ) { char charM[2];   sprintf( charM,"%ldM",MinsAfterH );  strD += charM; }
    if ( SecsAfterM != 0 ) { char charS[2];   sprintf( charS,"%ldS",SecsAfterM );  strD += charS; }
    if ( NsecAfterS != 0 ) { char charN[2];   sprintf( charN,"%ldN",NsecAfterS );  strD += charN; }

    return strD;
}

//--------------------
//  -- Static methods
//--------------------

/** The parsification engine,
 *  the string in standard format P[nY][nM][nD][T[nH][nM][n[.f]S]] is parsed in Duration
 */
int Duration::parseStringToDuration( const std::string& str_dur, Duration& d )
{
  d.m_sec  = 0;
  d.m_nsec = 0;

  // Check the duration string size
  size_t len_str_dur = str_dur.size();
  if( len_str_dur < 3 ) {return DURATION_STRING_TOO_SHORT;} // i.e. P2D
  if( len_str_dur > 30) {return DURATION_STRING_TOO_LONG;} // 20 w/o ns P01Y01M01DT01H01M01S

  // Find position of separators, if available
  size_t posP   = str_dur.find('P'); // Period
  size_t posY   = str_dur.find('Y'); // Year
  size_t posD   = str_dur.find('D'); // Day
  size_t posT   = str_dur.find('T'); // Time of the duration in Hours, Minutes, Seconds
  size_t posH   = str_dur.find('H'); // Hours
  size_t posF   = str_dur.find('.'); // Fraction of the second's decimal point
  size_t posS   = str_dur.find('S'); // Seconds
  size_t posM   = str_dur.find_first_of('M'); // Month
  size_t posMin = str_dur.find_last_of('M');  // Minutes

  if ( posP == string::npos ) { return DURATION_STRING_WRONG_FORMAT_MISSING_P; }

  size_t posN1 = posP + 1;
  size_t posNN;

  // sparse duration Y:
  if ( posY != string::npos ) { 
       posNN = posY - 1;
       size_t lenY = posNN - posN1 + 1;
       if( lenY > 7 ) {return DURATION_TOO_LONG_FIELD_FOR_YEARS;} 
       char charY[8];
       size_t lenY_copy=str_dur.copy(charY,lenY,posN1); charY[lenY_copy] = '\0';      
       d.m_sec += atoi(charY)*3600*24*364;
       posN1 = posY + 1;
  }

  // sparse duration M:
  if ( posM != string::npos ) {
       posNN = posM - 1;
       size_t lenM = posNN - posN1 + 1;
       if( lenM > 7 ) {return DURATION_TOO_LONG_FIELD_FOR_MONTHS;} 
       char charM[8];
       size_t lenM_copy=str_dur.copy(charM,lenM,posN1); charM[lenM_copy] = '\0';
       d.m_sec += atoi(charM)*3600*24*30;
       posN1 = posM + 1;
  }

  // sparse duration D:
  if ( posD != string::npos ) {
       posNN = posD - 1;
       size_t lenD = posNN - posN1 + 1;
       if( lenD > 7 ) {return DURATION_TOO_LONG_FIELD_FOR_DAYS;} 
       char charD[8];
       size_t lenD_copy=str_dur.copy(charD,lenD,posN1); charD[lenD_copy] = '\0';
       d.m_sec += atoi(charD)*3600*24;
       posN1 = posD + 2;
  }

  // check duration T:
  if ( posT != string::npos ) { // T info (H, M, S) is missing
    posN1 = posT + 1;
  }
  else
  {
    return PARSE_IS_OK;
  }

  // sparse duration H:
  if ( posH != string::npos ) {
    posNN = posH - 1;
    size_t lenH = posNN - posN1 + 1;
    if( lenH > 7 ) {return DURATION_TOO_LONG_FIELD_FOR_HOURS;} 
    char charH[8];
    size_t lenH_copy=str_dur.copy(charH,lenH,posN1); charH[lenH_copy] = '\0';
    d.m_sec += atoi(charH)*3600;
    posN1 = posH + 1;
  }

  // sparse duration M:
  if ( posMin != string::npos ) {
    posNN = posMin - 1;
    size_t lenM = posNN - posN1 + 1;
    if( lenM > 7 ) {return DURATION_TOO_LONG_FIELD_FOR_MINUTES;} 
    char charM[8];
    size_t lenM_copy=str_dur.copy(charM,lenM,posN1); charM[lenM_copy] = '\0';
    d.m_sec += atoi(charM)*60;
    posN1 = posMin + 1;
  }

  // Case of fractional second:
  // sparse duration S integer part of seconds:
  if ( posF != string::npos && posS != string::npos) {
    posNN = posF - 1;
    size_t lenS = posNN - posN1 + 1;
    if( lenS > 15 ) {return DURATION_TOO_LONG_FIELD_FOR_SECONDS;} 
    char charS[16];
    size_t lenS_copy=str_dur.copy(charS,lenS,posN1); charS[lenS_copy] = '\0';
    d.m_sec += atoi(charS);
    posN1 = posF + 1;

  // sparse duration S fractional part of second:
    posNN = posS - 1;
    size_t lenF = posNN - posN1 + 1;
    if( lenF > 9 ) {return DURATION_TOO_LONG_FIELD_FOR_SECOND_FRACTION;} 
    char charF[16];
    size_t lenF_copy=str_dur.copy(charF,lenF,posN1); charF[lenF_copy] = '\0';
    double frac_of_sec  = (double)atoi(charF);
    d.m_nsec = (size_t)(frac_of_sec * pow(10,9-lenF_copy));
    return PARSE_IS_OK;
  }

  // sparse duration S in integer seconds:
  if ( posS != string::npos) {
    posNN = posS - 1;
    size_t lenS = posNN - posN1 + 1;
    if( lenS > 7 ) {return DURATION_TOO_LONG_FIELD_FOR_SECONDS;} 
    char charS[16];
    size_t lenS_copy=str_dur.copy(charS,lenS,posN1); charS[lenS_copy] = '\0';
    d.m_sec += atoi(charS);
    posN1 = posF + 1;
  }
    return PARSE_IS_OK;
}

//--------------------
//  -- Private methods
//--------------------

void Duration::splitDurationSecsForYDHMS(time_t &Years, 
                                         time_t &DaysAfterY, 
                                         time_t &HoursAfterD, 
                                         time_t &MinsAfterH, 
                                         time_t &SecsAfterM) const
{
  Years             = m_sec / (364*24*3600);  
  time_t SecsAfterY = m_sec % (364*24*3600);
  DaysAfterY        = SecsAfterY / (24*3600); 
  time_t SecsAfterD = SecsAfterY % (24*3600); 
  HoursAfterD       = SecsAfterD / 3600;
  time_t SecsAfterH = SecsAfterD % 3600; 
  MinsAfterH        = SecsAfterH / 60;
  SecsAfterM        = SecsAfterH % 60; 
}


//-----------
// Friends --
//-----------

// overloaded stream-insertion operator <<
ostream & operator << ( ostream & os, const Duration & d )
{
    if ( d.m_nsec == 0 )
        {
            os << d.m_sec << " sec";
            return os;
        }
    else
        {
            cout.fill( '0' );
            os << d.m_sec << "." << setw(9) << d.m_nsec << " sec";
            cout.fill( ' ' );
            return os;
        }
}

} // namespace PSTime
