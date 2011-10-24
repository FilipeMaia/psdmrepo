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

#define BOOST_TEST_MODULE LusiTimeTest
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module LusiTimeTest.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  Time t0 ;
  BOOST_CHECK( not t0.isValid() ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_2 )
{
  Time t0( 1234567, -123456 )  ;
  BOOST_CHECK( not t0.isValid() ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_3 )
{
  Time t0( 1234567, 123456 )  ;
  BOOST_CHECK( t0.isValid() ) ;
  BOOST_CHECK_EQUAL( t0.sec(), 1234567 ) ;
  BOOST_CHECK_EQUAL( t0.nsec(), 123456 ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_4 )
{
  Time t0 ;
  BOOST_CHECK_NO_THROW( t0 = Time::now() ) ;
  BOOST_CHECK( t0.isValid() ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_cmp_1 )
{
  Time t0( 1234567, 123456 )  ;
  Time t1( 1234567, 123456 )  ;
  BOOST_CHECK( t0 == t1 ) ;
  BOOST_CHECK( not ( t0 != t1 ) ) ;
  BOOST_CHECK( not ( t0 < t1 ) ) ;
  BOOST_CHECK( t0 <= t1 ) ;
  BOOST_CHECK( not ( t0 > t1 ) ) ;
  BOOST_CHECK( t0 >= t1 ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_2 )
{
  Time t0( 1234567, 0 )  ;
  Time t1( 1234567, 123456 )  ;
  BOOST_CHECK( not ( t0 == t1 ) ) ;
  BOOST_CHECK( t0 != t1 ) ;
  BOOST_CHECK( t0 < t1 ) ;
  BOOST_CHECK( t0 <= t1 ) ;
  BOOST_CHECK( not ( t0 > t1 ) ) ;
  BOOST_CHECK( not ( t0 >= t1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_3 )
{
  Time t0( 123, 123 )  ;
  Time t1( 1234567, 123456 )  ;
  BOOST_CHECK( not ( t0 == t1 ) ) ;
  BOOST_CHECK( t0 != t1 ) ;
  BOOST_CHECK( t0 < t1 ) ;
  BOOST_CHECK( t0 <= t1 ) ;
  BOOST_CHECK( not ( t0 > t1 ) ) ;
  BOOST_CHECK( not ( t0 >= t1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_4 )
{
  Time t0( 123, 123456 )  ;
  Time t1( 1234567, 123456 )  ;
  BOOST_CHECK( not ( t0 == t1 ) ) ;
  BOOST_CHECK( t0 != t1 ) ;
  BOOST_CHECK( t0 < t1 ) ;
  BOOST_CHECK( t0 <= t1 ) ;
  BOOST_CHECK( not ( t0 > t1 ) ) ;
  BOOST_CHECK( not ( t0 >= t1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_5 )
{
  Time t0( 1234567, 123456 )  ;
  Time t1( 123, 123 )  ;
  BOOST_CHECK( not ( t0 == t1 ) ) ;
  BOOST_CHECK( t0 != t1 ) ;
  BOOST_CHECK( not ( t0 < t1 ) ) ;
  BOOST_CHECK( not ( t0 <= t1 ) ) ;
  BOOST_CHECK( t0 > t1 ) ;
  BOOST_CHECK( t0 >= t1 ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_6 )
{
  Time t0 ;
  Time t1( 123, 123 )  ;
  BOOST_CHECK_THROW( t0 == t1, Exception ) ;
  BOOST_CHECK_THROW( t0 != t1, Exception ) ;
  BOOST_CHECK_THROW( t0 < t1, Exception ) ;
  BOOST_CHECK_THROW( t0 <= t1, Exception ) ;
  BOOST_CHECK_THROW( t0 > t1, Exception ) ;
  BOOST_CHECK_THROW( t0 >= t1, Exception ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_7 )
{
  Time t0( 123, 123 ) ;
  Time t1 ;
  BOOST_CHECK_THROW( t0 == t1, Exception ) ;
  BOOST_CHECK_THROW( t0 != t1, Exception ) ;
  BOOST_CHECK_THROW( t0 < t1, Exception ) ;
  BOOST_CHECK_THROW( t0 <= t1, Exception ) ;
  BOOST_CHECK_THROW( t0 > t1, Exception ) ;
  BOOST_CHECK_THROW( t0 >= t1, Exception ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_8 )
{
  Time t0 ;
  Time t1 ;
  BOOST_CHECK_THROW( t0 == t1, Exception ) ;
  BOOST_CHECK_THROW( t0 != t1, Exception ) ;
  BOOST_CHECK_THROW( t0 < t1, Exception ) ;
  BOOST_CHECK_THROW( t0 <= t1, Exception ) ;
  BOOST_CHECK_THROW( t0 > t1, Exception ) ;
  BOOST_CHECK_THROW( t0 >= t1, Exception ) ;
}

