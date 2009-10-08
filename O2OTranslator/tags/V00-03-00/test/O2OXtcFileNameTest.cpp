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
  BOOST_CHECK_NO_THROW( name = O2OXtcFileName("/dir/e1_r2_s3_ch4.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e1_r2_s3_ch4.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 1U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 2U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 3U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 4U ) ;
}

BOOST_AUTO_TEST_CASE( test_2 )
{
  O2OXtcFileName name;
  BOOST_CHECK_NO_THROW( name = O2OXtcFileName("/dir/e123456_r234567_s345678_ch456789.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e123456_r234567_s345678_ch456789.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 123456U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 234567U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 345678U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 456789U ) ;
}

BOOST_AUTO_TEST_CASE( test_exc_1 )
{
  O2OXtcFileName name;
  BOOST_CHECK_THROW( name = O2OXtcFileName("/dir/e1r2s3ch4.xtc"), std::exception ) ;
  BOOST_CHECK_THROW( name = O2OXtcFileName("/dir/e1_r2_s3ch4.xtc"), std::exception ) ;
  BOOST_CHECK_THROW( name = O2OXtcFileName("/dir/e_1_r2_s3_ch4.xtc"), std::exception ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_1 )
{
  O2OXtcFileName name1;
  O2OXtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = O2OXtcFileName("/dir/e1_r2_s3_ch4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = O2OXtcFileName("/dir/e1_r2_s3_ch5.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_2 )
{
  O2OXtcFileName name1;
  O2OXtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = O2OXtcFileName("/dir1/e1_r2_s3_ch4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = O2OXtcFileName("/dir2/e1_r2_s4_ch4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_3 )
{
  O2OXtcFileName name1;
  O2OXtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = O2OXtcFileName("/dir2/e1_r2_s3_ch4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = O2OXtcFileName("/dir1/e1_r3_s3_ch4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_4 )
{
  O2OXtcFileName name1;
  O2OXtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = O2OXtcFileName("/dir3/e1_r2_s3_ch4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = O2OXtcFileName("/dir345/e2_r2_s3_ch4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}
