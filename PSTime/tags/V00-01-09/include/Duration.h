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
#include <string>

//		---------------------
// 		-- Class Interface --
//		---------------------


namespace PSTime {

/**
 *  @ingroup PSTime
 * 
 *  @brief This class is intended to work with durations in the ISO8601 standard.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Time
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class Duration  {
public:

  enum ParseStatus {
    PARSE_IS_OK,
    DURATION_STRING_TOO_SHORT,
    DURATION_STRING_TOO_LONG,
    DURATION_STRING_WRONG_FORMAT_MISSING_P,
    DURATION_TOO_LONG_FIELD_FOR_YEARS,
    DURATION_TOO_LONG_FIELD_FOR_MONTHS,
    DURATION_TOO_LONG_FIELD_FOR_DAYS,
    DURATION_TOO_LONG_FIELD_FOR_HOURS,
    DURATION_TOO_LONG_FIELD_FOR_MINUTES,
    DURATION_TOO_LONG_FIELD_FOR_SECONDS,
    DURATION_TOO_LONG_FIELD_FOR_SECOND_FRACTION
  };

/** Default constructor */
  Duration () ;

/** Copy constructor */
  Duration( const Duration & d );

/** Constructs from seconds and nanoseconds */
  Duration( time_t sec, time_t nsec = 0 );

/** 
 * Constructs from human numbers 
 * In this implementation I ignore monthes, because they may have ambigous number of days 30 or 31.
 * The same problem exists for years 364 or 365 days, though I hope that so long duration is not
 * very practical in our applications.
 */
  Duration( time_t Years,  // Needs at least in 3 parameters to distinguish from previous constructor
            time_t Days, 
            time_t Hours, 
            time_t Mins  = 0, 
	    time_t Secs  = 0,
            time_t Nsecs = 0 );

/** Destructor */
  virtual ~Duration () {};

    /** Operators */
    Duration   operator  - ( const Duration & d1 ) const;            

    Duration   operator  + ( const Duration & d1 ) const;            

    Duration & operator  = ( const Duration & d1 );            

    Duration & operator += ( const Duration & d1 );            

    bool operator == ( const Duration & d ) const    
        { 
            return ( m_sec == d.m_sec && m_nsec == d.m_nsec );
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
            return !( *this < d ); 
        }

    /** Selectors */
    time_t  getSec ( ) const { return m_sec;  }
    time_t  getNsec( ) const { return m_nsec; }

    /** Public methods */
    void Print() const;

    /**
     * Makes the duration string in the format: PnYnMnDTnHnMnS
     * from the object entity.
     */
    std::string strDurationBasic() const;


    /**
     * Splits the Duration object entity for
     * Years, DaysAfterY, HoursAfterD, MinsAfterH, SecsAfterM
     * Note, the month are not used because of 28,29,30,31 days ambiguity.
     */
    void splitDurationSecsForYDHMS(time_t &Years, 
                                   time_t &DaysAfterY, 
                                   time_t &HoursAfterD, 
                                   time_t &MinsAfterH, 
				   time_t &SecsAfterM) const;

    /** Friends */
    friend std::ostream & operator << ( std::ostream & os, const Duration & d );  

    // Static data member starts with s_.
    //    static const time_t s_nsecInASec;   

    /** 
     * The parsification engine,
     * the string in standard format P[nY][nM][nD][T[nH][nM][n[.f]S]] is parsed in the Duration object.
     * Note, we assume that the monnth M has 30 days..., 
     * so it is better to escape this ambiguity using any number of days D.
     * Also note that seconds S may be fractional: n[.f]S, that is beyond the ISO8601 standard.
     */
    static int parseStringToDuration( const std::string & str_dur, Duration & d );

private:

    /** Data members */
    time_t  m_sec;         // number of seconds
    time_t  m_nsec;        // number of nanoseconds

}; // class Duration
} // namespace PSTime

#endif // PSTIME_DURATION_H
