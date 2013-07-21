//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the AppDataPath.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppDataPath.h"
using namespace AppUtils ;

#define BOOST_TEST_MODULE AppDataPathTest
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module AppDataPathTest.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_exist )
{
  AppDataPath path("AppUtils/file-for-AppDataPath-unit-test");
  BOOST_CHECK(not path.path().empty()) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_not_exist )
{
  AppDataPath path("AppUtils/file-for-AppDataPath-unit-test-does-not-exist");
  BOOST_CHECK(path.path().empty()) ;
}
