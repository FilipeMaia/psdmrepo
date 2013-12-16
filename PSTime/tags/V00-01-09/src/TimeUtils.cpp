//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeUtils...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSTime/TimeUtils.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

#define USE_TIMEGM 1


#if !USE_TIMEGM

namespace {

  bool is_leap_year(int year) 
  {
    if (year % 400 == 0) return true;
    if (year % 100 == 0) return false;
    if (year % 4 == 0) return true;
    return false;
  }
  
  int mdays[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };

  // Returns the number of days in a month. 
  int days(int year, int month) 
  {
    int days = mdays[month];
    if (is_leap_year(year) and month > 0) ++ days;
    return days;
  }
  
  int run_days[12] = { 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 };

  // Returns the number of days since the beginning of the year 
  // right after the month ends. 
  int running_days(int year, int month) 
  {
    int days = run_days[month];
    if (is_leap_year(year) and month > 0) ++ days;
    return days;
  }
  
  // number of leap years between given year and 1970
  // not including given year
  int leap_years(int year) {
    return (year-1969)/4 - (year-1901)/100 + (year-1601)/400;
  }
  
}

#endif

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSTime {
namespace TimeUtils {

// Convert broken time to UTC time (in UTC timezone)
time_t 
timegm(struct tm* timeptr)
{
#if USE_TIMEGM
  
  return ::timegm(timeptr);

#else 
  
  // normalize, completely ignore leap seconds
  while (timeptr->tm_sec > 59) {
    timeptr->tm_sec -= 60;
    timeptr->tm_min += 1;
  }
  while (timeptr->tm_min > 59) {
    timeptr->tm_min -= 60;
    timeptr->tm_hour += 1;
  }
  while (timeptr->tm_hour > 23) {
    timeptr->tm_hour -= 24;
    timeptr->tm_mday += 1;
  }
  while (timeptr->tm_mday > ::days(timeptr->tm_year, timeptr->tm_mon)) {
    timeptr->tm_mday -= ::days(timeptr->tm_year, timeptr->tm_mon);
    timeptr->tm_mon += 1;
  }
  while (timeptr->tm_mon > 11) {
    timeptr->tm_mon -= 12;
    timeptr->tm_year += 1;
  }

  int year = timeptr->tm_year+1900;
  int days = (year-1970)*365 + ::leap_years(year);
  days += timeptr->tm_mon == 0 ? 0 : ::running_days(year, timeptr->tm_mon-1);
  days += timeptr->tm_mday-1;

  time_t result = days * time_t(24*3600);
  result += timeptr->tm_hour*3600 + timeptr->tm_min*60 + timeptr->tm_sec;
  
  return result;

#endif
}


} // namespace PSTime
} // namespace TimeUtils
