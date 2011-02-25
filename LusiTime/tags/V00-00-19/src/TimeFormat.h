#ifndef LUSITIME_TIMEFORMAT_H
#define LUSITIME_TIMEFORMAT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeFormat.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "LusiTime/Exceptions.h"
#include "LusiTime/Time.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace LusiTime {

/**
 *  Utility class to deal with the presentation of the times/dates.
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

class TimeFormat  {
public:

  /**
   * Parse the time string and return time
   */
  static Time parse( const std::string& timeStr ) throw (Exception) ;
  
  /**
   * Convert time to string according to format
   */
  static std::string format ( const Time& time, const std::string& afmt ) throw (Exception) ;
  
protected:

  // Default constructor
  TimeFormat () ;

  // Destructor
  ~TimeFormat () ;

private:

  // Data members

  // Copy constructor and assignment are disabled by default
  TimeFormat ( const TimeFormat& ) ;
  TimeFormat& operator = ( const TimeFormat& ) ;

};

} // namespace LusiTime

#endif // LUSITIME_TIMEFORMAT_H
