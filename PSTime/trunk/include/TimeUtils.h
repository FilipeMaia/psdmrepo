#ifndef PSTIME_TIMEUTILS_H
#define PSTIME_TIMEUTILS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeUtils.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>

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
 *  @brief Namespace containing various utility methods for date/time manipulation.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace TimeUtils  {

  /**
   *  @brief Convert broken time to UTC time (in UTC timezone).
   *  
   *  Unlike mktime this method does not normalize any members of struct tm.
   *  The members tm_isdst, tm_wday, and tm_yday are entirely ignored. No
   *  leap seconds accounted.
   */
  time_t timegm(struct tm* timeptr);

}

} // namespace PSTime

#endif // PSTIME_TIMEUTILS_H
