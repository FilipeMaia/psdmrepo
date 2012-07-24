//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the testTimeUtils.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSTime/TimeUtils.h"

using namespace PSTime;
using namespace std;

#define BOOST_TEST_MODULE testTimeUtils
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module testTimeUtils.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */


// Print (negative) time difference between UTC and local time
int printTimeDiff(int year, int month, int day)
{
  struct tm stm;
  stm.tm_year = year-1900;
  stm.tm_mon = month-1;
  stm.tm_mday = day;
  stm.tm_hour = 0;
  stm.tm_min = 0;
  stm.tm_sec = 0;
  stm.tm_isdst = -1;
  
  time_t tutc = TimeUtils::timegm(&stm);
  stm.tm_isdst = -1;
  time_t tloc = mktime(&stm);
  
  double diff = (tloc-tutc)/3600;
  
  cout << "\nTime: " << asctime(&stm);
  cout << "  UTC seconds  : " << tutc << endl;
  cout << "  Local seconds: " << tloc << endl;
  cout << "  Diff hours   : " << diff << endl;
  
  return int(diff);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  cout << "\n=== Non-leap years ===\n";
  
  int diff_1970_1_1 = printTimeDiff(1970, 1, 1);
  int diff_2001_1_1 = printTimeDiff(2001, 1, 1);
  int diff_2011_1_1 = printTimeDiff(2011, 1, 1);
  int diff_1970_6_1 = printTimeDiff(1970, 6, 1);
  int diff_2001_6_1 = printTimeDiff(2001, 6, 1);
  int diff_2011_6_1 = printTimeDiff(2011, 6, 1);

  BOOST_CHECK_EQUAL(diff_1970_1_1, diff_2001_1_1);
  BOOST_CHECK_EQUAL(diff_1970_1_1, diff_2011_1_1);
  BOOST_CHECK_EQUAL(diff_1970_1_1, diff_1970_6_1+1);
  BOOST_CHECK_EQUAL(diff_2001_1_1, diff_2001_6_1+1);
  BOOST_CHECK_EQUAL(diff_2001_1_1, diff_2011_6_1+1);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_2 )
{
  cout << "\n=== Leap years ===\n";
  
  int diff_1972_1_1 = printTimeDiff(1972, 1, 1);
  int diff_1972_6_1 = printTimeDiff(1972, 6, 1);
  int diff_1980_1_1 = printTimeDiff(1980, 1, 1);
  int diff_1980_6_1 = printTimeDiff(1980, 6, 1);

  BOOST_CHECK_EQUAL(diff_1972_1_1, diff_1980_1_1);
  BOOST_CHECK_EQUAL(diff_1972_1_1, diff_1972_6_1+1);
  BOOST_CHECK_EQUAL(diff_1980_1_1, diff_1980_6_1+1);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_3 )
{
  cout << "\n=== Leap (100) years ===\n";
  
  /*int diff_2100_1_1 =*/ printTimeDiff(2100, 1, 1);
  /*int diff_2100_6_1 =*/ printTimeDiff(2100, 6, 1);

  // timezone info does not have DST for 2100 yet, 
  // may be one of the two depending on platform
  //BOOST_CHECK_EQUAL(diff_2100_1_1, diff_2100_6_1+1);
  //BOOST_CHECK_EQUAL(diff_2100_1_1, diff_2100_6_1);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_4 )
{
  cout << "\n=== Leap (400) years ===\n";
  
  int diff_2000_1_1 = printTimeDiff(2000, 1, 1);
  int diff_2000_6_1 = printTimeDiff(2000, 6, 1);

  BOOST_CHECK_EQUAL(diff_2000_1_1, diff_2000_6_1+1);
}

