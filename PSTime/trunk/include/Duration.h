#ifndef PSTIME_DURATION_H
#define PSTIME_DURATION_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Duration.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

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

class Duration  {
public:

/** Default constructor */
  Duration () ;

/** Copy constructor */
  Duration( const Duration & d );

/** Constructs from seconds and nanoseconds */
  Duration( time_t sec, time_t nsec = 0 );

/** Destructor */
  virtual ~Duration () {};

/** Operators */
    Duration   operator  - ( const Duration & d1 ) const;            

    Duration   operator  + ( const Duration & d1 ) const;            

    Duration & operator  = ( const Duration & d1 );            

    Duration & operator += ( const Duration & d1 );            

    bool operator == ( const Duration & d ) const    
        { 
            return ( m_sec  == d.m_sec && m_nsec == d.m_nsec );
        }

    bool operator != ( const Duration & d ) const 
        { 
            return !( *this == d ); 
        }

    bool operator <  ( const Duration & d ) const  
        { 
            return ( m_sec < d.m_sec ) || ( m_sec == d.m_sec && m_nsec < d.m_nsec );
        }

    bool operator > ( const Duration & d ) const
        { 
            return ( m_sec > d.m_sec ) || ( m_sec == d.m_sec && m_nsec > d.m_nsec );
        }

    bool operator <=  ( const Duration & d ) const 
        { 
            return !( *this > d );
        }

    bool operator >= ( const Duration & d ) const 
        { 
            return !( *this <  d ); 
        }

    // Selectors
    time_t  getSec ( ) const { return m_sec;  }
    time_t  getNsec( ) const { return m_nsec; }

    // Friends
    friend std::ostream & operator << ( std::ostream & os, const Duration & d );  

    // Static data member starts with s_.
    //    static const time_t s_nsecInASec;   

private:

    // Data members
    time_t  m_sec;         // number of seconds
    time_t  m_nsec;        // number of nano seconds
};

} // namespace PSTime

#endif // PSTIME_DURATION_H
