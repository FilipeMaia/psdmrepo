//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the XtcFileName.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/XtcFileName.h"

using namespace XtcInput ;

#define BOOST_TEST_MODULE XtcFileName
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module XtcFileName.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  XtcFileName name;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/e1-r2-s3-c4.smd.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e1-r2-s3-c4.smd.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.small(), true ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 1U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 2U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 3U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 4U ) ;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e1-r2-s3-c4.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.small(), false ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 1U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 2U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 3U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 4U ) ;
}

BOOST_AUTO_TEST_CASE( test_2 )
{
  XtcFileName name;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/e123456-r234567-s345678-c456789.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e123456-r234567-s345678-c456789.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.small(), false ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 123456U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 234567U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 345678U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 456789U ) ;

  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/e123456-r234567-s345678-c456789.smd.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e123456-r234567-s345678-c456789.smd.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.small(), true ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 123456U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 234567U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 345678U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 456789U ) ;
}

BOOST_AUTO_TEST_CASE( test_3 )
{
  XtcFileName name;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir", 1, 2, 3, 4, false) ) ;
  BOOST_CHECK_EQUAL ( name.path(), "/dir/e1-r0002-s03-c04.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e1-r0002-s03-c04.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.small(), false ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 1U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 2U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 3U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 4U ) ;

  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir", 1, 2, 3, 4, true) ) ;
  BOOST_CHECK_EQUAL ( name.path(), "/dir/e1-r0002-s03-c04.smd.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e1-r0002-s03-c04.smd.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.small(), true ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 1U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 2U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 3U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 4U ) ;
}

BOOST_AUTO_TEST_CASE( test_4 )
{
  XtcFileName name;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/", 1, 2, 3, 4, false) ) ;
  BOOST_CHECK_EQUAL ( name.small(), false ) ;
  BOOST_CHECK_EQUAL ( name.path(), "/dir/e1-r0002-s03-c04.xtc" ) ;
  BOOST_CHECK_NO_THROW( name = XtcFileName("", 1, 2, 3, 4, false) ) ;
  BOOST_CHECK_EQUAL ( name.path(), "e1-r0002-s03-c04.xtc" ) ;
  BOOST_CHECK_NO_THROW( name = XtcFileName(".", 1, 2, 3, 4, false) ) ;
  BOOST_CHECK_EQUAL ( name.path(), "./e1-r0002-s03-c04.xtc" ) ;

  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/", 1, 2, 3, 4, true) ) ;
  BOOST_CHECK_EQUAL ( name.small(), true ) ;
  BOOST_CHECK_EQUAL ( name.path(), "/dir/e1-r0002-s03-c04.smd.xtc" ) ;
  BOOST_CHECK_NO_THROW( name = XtcFileName("", 1, 2, 3, 4, true) ) ;
  BOOST_CHECK_EQUAL ( name.path(), "e1-r0002-s03-c04.smd.xtc" ) ;
  BOOST_CHECK_NO_THROW( name = XtcFileName(".", 1, 2, 3, 4, true) ) ;
  BOOST_CHECK_EQUAL ( name.path(), "./e1-r0002-s03-c04.smd.xtc" ) ;
}

BOOST_AUTO_TEST_CASE( test_5 )
{
  XtcFileName name;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir", 123456, 234567, 345678, 456789, true) ) ;
  BOOST_CHECK_EQUAL ( name.small(), true ) ;
  BOOST_CHECK_EQUAL ( name.path(), "/dir/e123456-r234567-s345678-c456789.smd.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e123456-r234567-s345678-c456789.smd.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 123456U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 234567U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 345678U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 456789U ) ;

  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir", 123456, 234567, 345678, 456789, false) ) ;
  BOOST_CHECK_EQUAL ( name.small(), false ) ;
  BOOST_CHECK_EQUAL ( name.path(), "/dir/e123456-r234567-s345678-c456789.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.basename(), "e123456-r234567-s345678-c456789.xtc" ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 123456U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 234567U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 345678U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 456789U ) ;
}

BOOST_AUTO_TEST_CASE( test_fail_1 )
{
  XtcFileName name;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/e1r2s3c4.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.small(), false ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 0U ) ;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/e1r2s3c4.smd.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.small(), true ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 0U ) ;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/e1-r2-s3c4.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.small(), false ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 0U ) ;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/e1-r2-s3c4.smd.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.small(), true ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 0U ) ;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/e-1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.small(), false ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 0U ) ;
  BOOST_CHECK_NO_THROW( name = XtcFileName("/dir/e-1-r2-s3-c4.smd.xtc") ) ;
  BOOST_CHECK_EQUAL ( name.small(), true ) ;
  BOOST_CHECK_EQUAL ( name.expNum(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.run(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.stream(), 0U ) ;
  BOOST_CHECK_EQUAL ( name.chunk(), 0U ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_1 )
{
  XtcFileName name1;
  XtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir/e1-r2-s3-c5.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir/e1-r2-s3-c4.smd.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir/e1-r2-s3-c5.smd.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_2 )
{
  XtcFileName name1;
  XtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir1/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir2/e1-r2-s4-c4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir1/e1-r2-s3-c4.smd.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir2/e1-r2-s4-c4.smd.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_3 )
{
  XtcFileName name1;
  XtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir2/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir1/e1-r3-s3-c4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir2/e1-r2-s3-c4.smd.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir1/e1-r3-s3-c4.smd.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_4 )
{
  XtcFileName name1;
  XtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir3/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir345/e2-r2-s3-c4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir3/e1-r2-s3-c4.smd.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir345/e2-r2-s3-c4.smd.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}

BOOST_AUTO_TEST_CASE( test_cmp_5 )
{
  XtcFileName name1;
  XtcFileName name2;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir3/e1-r2-s3-c4.smd.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir345/e2-r2-s3-c4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir345/e2-r2-s3-c4.smd.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir3/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
  BOOST_CHECK_NO_THROW( name1 = XtcFileName("/dir3/e1-r2-s3-c4.smd.xtc") ) ;
  BOOST_CHECK_NO_THROW( name2 = XtcFileName("/dir3/e1-r2-s3-c4.xtc") ) ;
  BOOST_CHECK( name1 < name2 ) ;
  BOOST_CHECK( not ( name1 < name1 ) ) ;
  BOOST_CHECK( not ( name2 < name1 ) ) ;
}


BOOST_AUTO_TEST_CASE( test_ext )
{
  BOOST_CHECK_EQUAL ( XtcFileName("e1-r2-s3-c4.xtc").extension(), ".xtc" ) ;
  BOOST_CHECK_EQUAL ( XtcFileName("/dir3/e1-r2-s3-c4.xtc").extension(), ".xtc" ) ;
  BOOST_CHECK_EQUAL ( XtcFileName("e1-r2-s3-c4.smd.xtc").extension(), ".xtc" ) ;
  BOOST_CHECK_EQUAL ( XtcFileName("/dir3/e1-r2-s3-c4.smd.xtc").extension(), ".xtc" ) ;
  BOOST_CHECK_EQUAL ( XtcFileName("/dir3/e1-r2-s3-c4.xtc.inprogress").extension(), ".inprogress" ) ;
  BOOST_CHECK_EQUAL ( XtcFileName("/dir3/e1-r2-s3-c4").extension(), "" ) ;
  BOOST_CHECK_EQUAL ( XtcFileName("/dir3/e1-r2-s3-c4.smd").extension(), ".smd" ) ;
  BOOST_CHECK_EQUAL ( XtcFileName("/dir3.dirext/e1-r2-s3-c4").extension(), "" ) ;
  BOOST_CHECK_EQUAL ( XtcFileName("/dir3.dirext/.hidden").extension(), ".hidden" ) ;
  BOOST_CHECK_EQUAL ( XtcFileName("/dir3.dirext/").extension(), "" ) ;
}
