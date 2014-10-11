//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the LusiTimeTest.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "LusiTime/Time.h"
#include "LusiTime/Exceptions.h"

using namespace LusiTime ;

#define BOOST_TEST_MODULE LusiTimeParseTest
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module LusiTimeTest.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  // test for S<seconds>[.<fractions>] syntax
  
  Time t0 ;
  
  // test illegal syntax
  BOOST_CHECK_THROW ( t0 = Time::parse("S 1"), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse(" S1"), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("S1 "), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("S12345678900"), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("S-1"), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("S1."), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("S0.1234567890"), ParseException ) ;
  
  // test legal syntax, all variations
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("S1234567890") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 1234567890, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("S1234567890.1") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 1234567890, 100000000 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("S1234567890.123456789") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 1234567890, 123456789 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("S0.000000001") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 1 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("S0") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_2 )
{
  // test for legal date-time specifier
  
  Time t0 ;
  
  // epoch time in different formats
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("1970-01-01 00:00:00Z") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("1970-01-01 00:00:00-00") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("1970-01-01 00:00:00+00:00") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("1970-01-01 00:00:00-0000") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700101 000000Z") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700101        000000Z") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700101T000000Z") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700101T010000+01") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700101T010000+0100") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("1970-01-01 01:00:00+01:00") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 0, 0 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_3 )
{
  // test for illegal date-time specifier
  
  Time t0 ;
  BOOST_CHECK_THROW ( t0 = Time::parse("1970-02-03 10:11:12.1234567890123456789"), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("1999-99-03"), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("1999-01-99"), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("1999-01-01 25:00:00"), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("1999-01-01 00:60:00"), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("1999-01-01 00:00:61"), ParseException ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("1999-02-30 00:00:00"), ParseException ) ;
}

BOOST_AUTO_TEST_CASE( test_4 )
{
  // test for timezones
  
  Time t0 ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("197002 Z") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 31*24*3600, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700102 Z") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 24*3600, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("1970-01-01 -08") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 8*3600, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700102T000000Z") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 24*3600, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700102T000000+0100") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 23*3600, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700102T000000+1000") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 14*3600, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700102T000000-1000") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 34*3600, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700102T000000-0140") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 24*3600+100*60, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700102T000000+0140") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 24*3600-100*60, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700102T000000-01:40") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 24*3600+100*60, 0 ) ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("19700102T000000+01:40") ) ;
  BOOST_CHECK_EQUAL ( t0, Time ( 24*3600-100*60, 0 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_5 )
{
  // test for local timezone
  
  Time t0 ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("20010101") ) ;
  BOOST_CHECK_NO_THROW ( t0 = Time::parse("20010101 01:01:01.01") ) ;
  BOOST_CHECK_THROW ( t0 = Time::parse("20010431 01:01:01.01"), ParseException ) ;
}

