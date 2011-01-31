#ifndef PSTIME_TIMEFORMAT_H
#define PSTIME_TIMEFORMAT_H

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
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSTime/Time.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSTime {

/**
 *  Utility class to deal with the presentation of the times/dates.
 *
 *  This software was developed for the LUSI project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace TimeFormat  {

  /**
   *  @brief Parse the time string and return time.
   *  
   * Accepts the time stamp in format:
   *     @li <date>, or
   *     @li <date> <time>[.<fraction-of-sec>][<zone>], or
   *     @li <date>T<time>[.<fraction-of-sec>][<zone>]
   *     
   *     where
   *     @li <date> should be in format YYYY-MM-DD, YYYYMMDD, YYYY-MM, YYYYMM, or YYYY, if
   *            month or day is missing they are assumed to be 1;
   *     @li <time> should be in format HH:MM:SS, HHMMSS, HH:MM, HHMM, HH, if minutes or 
   *            seconds are missing they are assumed to be 0;
   *     @li <fraction-of-sec> may have up to 9 digits for nsec; if this field is missing, 
   *                       it is assumed equal to 0, fraction requre seconds to be specified;
   *     @li <zone> should be Z for UTC or in format <sign + or ->HH[:MM] or <sign + or ->HHMM
   *             if this field is missing time is assumed to be in local time zone.
   *
   *  @throw TimeParseException
   */
  Time parseTime( const std::string& timeStr );
  
  /**
   *  @brief Convert time to string according to format.
   *  
   *  Following format codes are understood:
   *  
   *   @li @c  %d     The day of the month as a decimal number (range 01 to 31).
   *   @li @c  %.<N>f Fractional seconds, will print dot followed by <N> digits.
   *   @li @c  %f     Equivalent to %.9f
   *   @li @c  %F     Equivalent to %Y-%m-%d (the ISO 8601 date format).
   *   @li @c  %H     The hour as a decimal number using a 24-hour clock (range 00 to 23).
   *   @li @c  %m     The month as a decimal number (range 01 to 12).
   *   @li @c  %M     The minute as a decimal number (range 00 to 59).
   *   @li @c  %s     The number of seconds since the Epoch, i.e., since 1970-01-01 00:00:00 UTC.
   *   @li @c  %S     The second as a decimal number (range 00 to 60).
   *   @li @c  %T     The time in 24-hour notation (%H:%M:%S).
   *   @li @c  %Y     The year as a decimal number including the century.
   *   @li @c  %z     'Z' if time is printed in UTC zone, or offset from UTC to local time.
   *   @li @c  %%     A literal '%' character.
   *  
   *  Any unrecognized characters not in the above set are copied to output string intact.
   *    
   *  @param[in] time   
   */
  std::string format ( const Time& time, const std::string& afmt, Time::Zone zone );
  
  
  void format ( std::ostream& str, const Time& time, const std::string& afmt, Time::Zone zone );

} // namespace TimeFormat

} // namespace PSTime

#endif // PSTIME_TIMEFORMAT_H
