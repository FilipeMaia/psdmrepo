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

#define BOOST_TEST_MODULE LusiTimeFormatTest
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module LusiTimeTest.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  // test for %f specifier 
  
  Time t0(0,123456789) ;
  std::string tstr ;
  
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".123456789" ) ;

  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.1f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".1" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.2f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".12" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.3f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".123" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.4f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".1234" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.5f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".12345" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.6f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".123456" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.7f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".1234567" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.8f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".12345678" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.9f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".123456789" ) ;
  
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.0f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".1" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.10f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".123456789" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%.128f") ) ;
  BOOST_CHECK_EQUAL( tstr, ".123456789" ) ;
}

BOOST_AUTO_TEST_CASE( test_2 )
{
  // test for date specifiers, assume this test runs in US, -0800 timezone
  
  Time t0(0,0) ;
  std::string tstr ;
  
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%Y") ) ;
  BOOST_CHECK_EQUAL( tstr, "1969" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%Y-%m") ) ;
  BOOST_CHECK_EQUAL( tstr, "1969-12" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%Y-%m-%d") ) ;
  BOOST_CHECK_EQUAL( tstr, "1969-12-31" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%F") ) ;
  BOOST_CHECK_EQUAL( tstr, "1969-12-31" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%Y-%j") ) ;
  BOOST_CHECK_EQUAL( tstr, "1969-365" ) ;
}

BOOST_AUTO_TEST_CASE( test_3 )
{
  // test for time specifiers, assume this test runs in US, -0800 timezone
  
  Time t0(0,0) ;
  std::string tstr ;
  
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%H:%M:%S") ) ;
  BOOST_CHECK_EQUAL( tstr, "16:00:00" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%T") ) ;
  BOOST_CHECK_EQUAL( tstr, "16:00:00" ) ;
  BOOST_CHECK_NO_THROW ( tstr = t0.toString("%T%z") ) ;
  BOOST_CHECK_EQUAL( tstr, "16:00:00-0800" ) ;
}


BOOST_AUTO_TEST_CASE( test_4 )
{
  // test for default time formatting, assume this test runs in US, -0800 timezone
  
  Time t0(0,0) ;
  std::string tstr ;
  
  BOOST_CHECK_NO_THROW ( tstr = t0.toString() ) ;
  BOOST_CHECK_EQUAL( tstr, "1969-12-31 16:00:00.000000000-0800" ) ;

  Time t1(1234567890,123456789) ;
  
  BOOST_CHECK_NO_THROW ( tstr = t1.toString() ) ;
  BOOST_CHECK_EQUAL( tstr, "2009-02-13 15:31:30.123456789-0800" ) ;
}

