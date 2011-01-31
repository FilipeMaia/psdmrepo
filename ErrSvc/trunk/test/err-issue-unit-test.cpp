//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the err-issue-unit-test.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ErrSvc/Issue.h"

using namespace ErrSvc ;

#define BOOST_TEST_MODULE err_issue_unit_test
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module err-issue-unit-test.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */


// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  BOOST_CHECK_NO_THROW( Issue(ERR_LOC, "test") ) ;

  Issue issue(ERR_LOC, "Some problem 1");
  std::cout << issue << std::endl;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_2 )
{
  try {
    throw Issue(ERR_LOC, "Some problem 2");
  } catch (const Issue& ex) {
    std::cout << ex << std::endl;
  }
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_3 )
{
  try {
    throw Issue(ERR_LOC, "Some problem 3");
  } catch (const std::exception& ex) {
    std::cout << ex.what() << std::endl;
  }
}
