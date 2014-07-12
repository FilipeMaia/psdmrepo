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
  // We expect to have fiducialsGreater emit a warning like:
  //
  // clock drift is -122 exceeds 90 secondsA=1000 fidA=0 secondsB=940 fidB=65520
  //
  // if the code that generated this warning has been disabled (for performance 
  // purposes) then this test should be removed.
  std::stringstream buffer;
  std::streambuf * old = std::cerr.rdbuf(buffer.rdbuf());

  BOOST_CHECK( not fidCompare.fiducialsGreater(1000,         0, 1000-60, 0x1FFE0/2) );
  std::string warning = buffer.str();
  std::cerr.rdbuf(old);
  BOOST_CHECK(warning.find("clock drift") != std::string::npos);
  BOOST_CHECK(warning.find("exceeds") != std::string::npos);
}

BOOST_AUTO_TEST_CASE( test_cmp_4 )
{
  FiducialsCompare fidCompare;
  // the below times generated the drift warning when the drift was calculated with poor
  // precision float arthimetic, but the drift is only 1 or 2 seconds.
  // This test is not needed if the warning code is disabled.
  std::stringstream buffer;
  std::streambuf * old = std::cerr.rdbuf(buffer.rdbuf());
  BOOST_CHECK( not fidCompare.fiducialsGreater(1394974655 ,   86835, 1394974656, 86856) );
  std::string warning = buffer.str();
  std::cerr.rdbuf(old);
  std::cout << warning << std::endl;
  BOOST_CHECK(warning.find("clock drift") == std::string::npos);
  BOOST_CHECK(warning.find("exceeds") == std::string::npos);

}
