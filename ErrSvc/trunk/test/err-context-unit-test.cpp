//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the err-context-unit-test.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ErrSvc/Context.h"

using namespace ErrSvc ;

#define BOOST_TEST_MODULE err_context_unit_test
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module err-context-unit-test.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  BOOST_CHECK_NO_THROW( Context x("", 1, "") ) ;

  Context ctx(ERR_LOC);
  std::cout << ctx << std::endl;
}
