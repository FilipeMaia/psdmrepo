//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: XtcFileNameTest.cpp 5559 2013-03-01 22:07:31Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//	Test suite case for the XtcFileName.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/FiducialsCompare.h"

using namespace XtcInput ;

#define BOOST_TEST_MODULE FiducialsCompare
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module FiducialsCompare.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================


BOOST_AUTO_TEST_CASE( test_cmp_1 )
{
  FiducialsCompare fidCompare;
                                           // secA, fidA, secB, fidB
  BOOST_CHECK(     fidCompare.fiducialsGreater(1060, 10, 1000,  5) );
  BOOST_CHECK( not fidCompare.fiducialsGreater(1000,  5, 1060, 10) );

  BOOST_CHECK(     fidCompare.fiducialsGreater(1000, 10, 1060,  5) );
  BOOST_CHECK( not fidCompare.fiducialsGreater(1060,  5, 1000, 10) );

  // if equal, 
  BOOST_CHECK( not fidCompare.fiducialsGreater(1060, 10,  1000, 10) );
  BOOST_CHECK( not fidCompare.fiducialsGreater(1000, 10,  1060, 10) );

  BOOST_CHECK( fidCompare.fiducialsEqual(1060, 10,  1000, 10) );
  BOOST_CHECK( fidCompare.fiducialsEqual(1000, 10,  1060, 10) );
}

BOOST_AUTO_TEST_CASE( test_cmp_2 )
{
  FiducialsCompare fidCompare;

  // fid=10 > fid=0x1fed6 as the fiducials wrapped around 
  BOOST_CHECK(     fidCompare.fiducialsGreater(1060,         10, 1000, 0x1FFE0-10) );
  BOOST_CHECK( not fidCompare.fiducialsGreater(1000, 0x1FFE0-10, 1060,         10) );
  BOOST_CHECK(     fidCompare.fiducialsGreater(1000,         10, 1060, 0x1FFE0-10) );
  BOOST_CHECK( not fidCompare.fiducialsGreater(1060, 0x1FFE0-10, 1000,         10) );

  // if the second is early enough with fid=10, then it is less than
  BOOST_CHECK( not fidCompare.fiducialsGreater(1000-100,     10, 1000, 0x1FFE0-10) );
  BOOST_CHECK( not fidCompare.fiducialsGreater(1100-300,     10, 1100, 0x1FFE0-10) );
  BOOST_CHECK(     fidCompare.fiducialsGreater(1000, 0x1FFE0-10, 1000-360,     10) );
}

BOOST_AUTO_TEST_CASE( test_cmp_3 )
{
  FiducialsCompare fidCompare;

  // Here we set up two times where the clocks are within 80 seconds, but then the
  // drift based on the fiducials exceeds 80 seconds.
  BOOST_CHECK( std::abs(fidCompare.fidBasedBtoASecondsDiff(1000,         0, 1000-60, 0x1FFE0/2)) > 80);
  // the below times generated a large difference when the difference was calculated with poor
  // precision float arthimetic, but the difference is only 1 or 2 seconds. 
  BOOST_CHECK( std::abs(fidCompare.fidBasedBtoASecondsDiff(1394974655 ,   86835, 1394974656, 86856)) < 3 );
}

