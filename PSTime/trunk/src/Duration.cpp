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
using std::cout;
using std::ostream;
using std::setw;


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

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
