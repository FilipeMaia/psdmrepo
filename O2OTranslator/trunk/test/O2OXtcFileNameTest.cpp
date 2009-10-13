//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the O2OFileNameFactoryTest.
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "O2OTranslator/O2OXtcFileName.h"

using namespace O2OTranslator ;

#define BOOST_TEST_MODULE O2OFileNameFactoryTest
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module O2OXtcFileNameFactory.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  O2OXtcFileName name;
  BOOST_CHECK_NO_THROW( name = O2OXtcFileName("/dir/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e1-r2-s3-c4.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 1U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 2U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 3U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 4U ) ;
}

BOOST_AUTO_TEST_CASE( test_2 )
{
  O2OXtcFileName name;
  BOOST_CHECK_NO_THROW( name = O2OXtcFileName("/dir/e123456-r234567-s345678-c456789.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e123456-r234567-s345678-c456789.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 123456U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 234567U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 345678U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 456789U ) ;
}

BOOST_AUTO_TEST_CASE( test_fail_1 )
{
  O2OXtcFileName name;
  BOOST_CHECK_NO_THROW( name = O2OXtcFileName("/dir/e1r2s3c4.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 0U ) ;
  BOOST_CHECK_NO_THROW( name = O2OXtcFileName("/dir/e1-r2-s3c4.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 0U ) ;
  BOOST_CHECK_NO_THROW( name = O2OXtcFileName("/dir/e-1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 0U ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_1 )
{
  O2OXtcFileName name1;
  O2OXtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = O2OXtcFileName("/dir/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = O2OXtcFileName("/dir/e1-r2-s3-c5.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_2 )
{
  O2OXtcFileName name1;
  O2OXtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = O2OXtcFileName("/dir1/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = O2OXtcFileName("/dir2/e1-r2-s4-c4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_3 )
{
  O2OXtcFileName name1;
  O2OXtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = O2OXtcFileName("/dir2/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = O2OXtcFileName("/dir1/e1-r3-s3-c4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_4 )
{
  O2OXtcFileName name1;
  O2OXtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = O2OXtcFileName("/dir3/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = O2OXtcFileName("/dir345/e2-r2-s3-c4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}
