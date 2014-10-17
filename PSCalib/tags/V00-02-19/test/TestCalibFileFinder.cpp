//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the TestCalibFileFinder.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSCalib/CalibFileFinder.h"

using namespace PSCalib;
using namespace std;

#define BOOST_TEST_MODULE TestCalibFileFinder
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module TestCalibFileFinder.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  // one interval covering whole range
  vector<string> files;
  files.push_back("0-end.data");

  string res;

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 0));
  BOOST_CHECK_EQUAL(res, "0-end.data");

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 100));
  BOOST_CHECK_EQUAL(res, "0-end.data");

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 1000000));
  BOOST_CHECK_EQUAL(res, "0-end.data");
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_2 )
{
  // two adjacent intervals
  vector<string> files;
  files.push_back("0-10.data");
  files.push_back("11-20.data");

  string res;

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 0));
  BOOST_CHECK_EQUAL(res, "0-10.data");
  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 10));
  BOOST_CHECK_EQUAL(res, "0-10.data");

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 11));
  BOOST_CHECK_EQUAL(res, "11-20.data");
  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 20));
  BOOST_CHECK_EQUAL(res, "11-20.data");

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 21));
  BOOST_CHECK_EQUAL(res, "");
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_3 )
{
  // two overlapping intervals
  vector<string> files;
  files.push_back("0-10.data");
  files.push_back("5-15.data");

  string res;

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 0));
  BOOST_CHECK_EQUAL(res, "0-10.data");
  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 4));
  BOOST_CHECK_EQUAL(res, "0-10.data");

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 5));
  BOOST_CHECK_EQUAL(res, "5-15.data");
  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 15));
  BOOST_CHECK_EQUAL(res, "5-15.data");
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_4 )
{
  // overlapping intervals with common start
  vector<string> files;
  files.push_back("0-10.data");
  files.push_back("0-5.data");
  files.push_back("0-end.data");

  string res;

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 0));
  BOOST_CHECK_EQUAL(res, "0-5.data");
  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 5));
  BOOST_CHECK_EQUAL(res, "0-5.data");

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 6));
  BOOST_CHECK_EQUAL(res, "0-10.data");
  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 10));
  BOOST_CHECK_EQUAL(res, "0-10.data");

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 11));
  BOOST_CHECK_EQUAL(res, "0-end.data");
  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 20));
  BOOST_CHECK_EQUAL(res, "0-end.data");
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_5 )
{
  // one interval completely inside another
  vector<string> files;
  files.push_back("0-end.data");
  files.push_back("5-10.data");

  string res;

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 0));
  BOOST_CHECK_EQUAL(res, "0-end.data");
  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 4));
  BOOST_CHECK_EQUAL(res, "0-end.data");

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 5));
  BOOST_CHECK_EQUAL(res, "5-10.data");
  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 10));
  BOOST_CHECK_EQUAL(res, "5-10.data");

  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 11));
  BOOST_CHECK_EQUAL(res, "0-end.data");
  BOOST_CHECK_NO_THROW(res = CalibFileFinder::selectCalibFile(files, 20));
  BOOST_CHECK_EQUAL(res, "0-end.data");
}
