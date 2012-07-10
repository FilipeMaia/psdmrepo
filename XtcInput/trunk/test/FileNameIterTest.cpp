//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the FileNameIterTest.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/ChunkFileIterList.h"
#include "XtcInput/RunFileIterList.h"
#include "XtcInput/StreamFileIterList.h"

using namespace XtcInput ;

#define BOOST_TEST_MODULE FileNameIterTest
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module FileNameIterTest.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_chunk )
{
  XtcFileName names[] = {
      XtcFileName("e1-r001-s00-c00.xtc"),
      XtcFileName("e1-r001-s00-c01.xtc"),
      XtcFileName("e1-r001-s00-c02.xtc"),
  };
  const int num_names = sizeof names / sizeof names[0];

  ChunkFileIterList iter(&names[0], &names[num_names]);
  BOOST_CHECK_EQUAL(iter.liveTimeout(), 0U);

  XtcFileName name;
  name = iter.next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r001-s00-c00.xtc");
  name = iter.next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r001-s00-c01.xtc");
  name = iter.next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r001-s00-c02.xtc");
  name = iter.next();
  BOOST_CHECK(name.path().empty());
  
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_stream_1 )
{
  XtcFileName names[] = {
      XtcFileName("e1-r001-s01-c01.xtc"),
      XtcFileName("e1-r001-s00-c01.xtc"),
      XtcFileName("e1-r001-s00-c00.xtc"),
      XtcFileName("e1-r001-s01-c00.xtc"),
  };
  const int num_names = sizeof names / sizeof names[0];

  StreamFileIterList iter(&names[0], &names[num_names], MergeFileName);

  boost::shared_ptr<ChunkFileIterI> chunkIter;
  XtcFileName name;

  chunkIter = iter.next();
  BOOST_REQUIRE(chunkIter);
  BOOST_CHECK_EQUAL(iter.stream(), 0U);
  name = chunkIter->next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r001-s00-c00.xtc");
  name = chunkIter->next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r001-s00-c01.xtc");
  name = chunkIter->next();
  BOOST_CHECK(name.path().empty());

  chunkIter = iter.next();
  BOOST_REQUIRE(chunkIter);
  BOOST_CHECK_EQUAL(iter.stream(), 1U);
  name = chunkIter->next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r001-s01-c00.xtc");
  name = chunkIter->next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r001-s01-c01.xtc");
  name = chunkIter->next();
  BOOST_CHECK(name.path().empty());

  chunkIter = iter.next();
  BOOST_CHECK(not chunkIter);
  
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_stream_2 )
{
  XtcFileName names[] = {
      XtcFileName("e1-r001-s01-c01.xtc"),
      XtcFileName("e1-r001-s00-c01.xtc"),
      XtcFileName("e1-r001-s00-c00.xtc"),
      XtcFileName("e1-r001-s01-c00.xtc"),
  };
  const unsigned num_names = sizeof names / sizeof names[0];

  StreamFileIterList iter(&names[0], &names[num_names], MergeOneStream);

  boost::shared_ptr<ChunkFileIterI> chunkIter;
  XtcFileName name;

  chunkIter = iter.next();
  BOOST_REQUIRE(chunkIter);
  BOOST_CHECK_EQUAL(iter.stream(), 0U);
  
  for (unsigned i = 0; i != num_names; ++ i) {
    name = chunkIter->next();
    BOOST_CHECK_EQUAL(name.path(), names[i].path());
  }

  name = chunkIter->next();
  BOOST_CHECK(name.path().empty());

  chunkIter = iter.next();
  BOOST_CHECK(not chunkIter);
  
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_stream_3 )
{
  XtcFileName names[] = {
      XtcFileName("e1-r001-s01-c01.xtc"),
      XtcFileName("e1-r001-s00-c01.xtc"),
      XtcFileName("e1-r001-s00-c00.xtc"),
      XtcFileName("e1-r001-s01-c00.xtc"),
  };
  const unsigned num_names = sizeof names / sizeof names[0];

  StreamFileIterList iter(&names[0], &names[num_names], MergeNoChunking);

  boost::shared_ptr<ChunkFileIterI> chunkIter;
  XtcFileName name;

  for (unsigned i = 0; i != num_names; ++ i) {
    chunkIter = iter.next();
    BOOST_REQUIRE(chunkIter);
    BOOST_CHECK_EQUAL(iter.stream(), i);
    name = chunkIter->next();
    BOOST_CHECK_EQUAL(name.path(), names[i].path());
    name = chunkIter->next();
    BOOST_CHECK(name.path().empty());
  }

}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_run_1 )
{
  XtcFileName names[] = {
      XtcFileName("e1-r001-s00-c00.xtc"),
      XtcFileName("e1-r001-s01-c00.xtc"),
      XtcFileName("e1-r002-s01-c00.xtc"),
      XtcFileName("e1-r002-s00-c00.xtc"),
      XtcFileName("e1-r003-s00-c00.xtc"),
  };
  const int num_names = sizeof names / sizeof names[0];

  RunFileIterList iter(&names[0], &names[num_names], MergeFileName);

  boost::shared_ptr<StreamFileIterI> streamIter;
  boost::shared_ptr<ChunkFileIterI> chunkIter;
  XtcFileName name;

  streamIter = iter.next();
  BOOST_REQUIRE(streamIter);
  BOOST_CHECK_EQUAL(iter.run(), 1U);
  
  chunkIter = streamIter->next();
  BOOST_CHECK_EQUAL(streamIter->stream(), 0U);
  BOOST_REQUIRE(chunkIter);
  name = chunkIter->next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r001-s00-c00.xtc");
  name = chunkIter->next();
  BOOST_CHECK(name.path().empty());

  chunkIter = streamIter->next();
  BOOST_CHECK_EQUAL(streamIter->stream(), 1U);
  name = chunkIter->next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r001-s01-c00.xtc");
  name = chunkIter->next();
  BOOST_CHECK(name.path().empty());

  chunkIter = streamIter->next();
  BOOST_CHECK(not chunkIter);


  streamIter = iter.next();
  BOOST_REQUIRE(streamIter);
  BOOST_CHECK_EQUAL(iter.run(), 2U);
  
  chunkIter = streamIter->next();
  BOOST_CHECK_EQUAL(streamIter->stream(), 0U);
  BOOST_REQUIRE(chunkIter);
  name = chunkIter->next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r002-s00-c00.xtc");
  name = chunkIter->next();
  BOOST_CHECK(name.path().empty());

  chunkIter = streamIter->next();
  BOOST_CHECK_EQUAL(streamIter->stream(), 1U);
  name = chunkIter->next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r002-s01-c00.xtc");
  name = chunkIter->next();
  BOOST_CHECK(name.path().empty());

  chunkIter = streamIter->next();
  BOOST_CHECK(not chunkIter);


  streamIter = iter.next();
  BOOST_REQUIRE(streamIter);
  BOOST_CHECK_EQUAL(iter.run(), 3U);
  
  chunkIter = streamIter->next();
  BOOST_CHECK_EQUAL(streamIter->stream(), 0U);
  BOOST_REQUIRE(chunkIter);
  name = chunkIter->next();
  BOOST_CHECK_EQUAL(name.path(), "e1-r003-s00-c00.xtc");
  name = chunkIter->next();
  BOOST_CHECK(name.path().empty());

  chunkIter = streamIter->next();
  BOOST_CHECK(not chunkIter);


  streamIter = iter.next();
  BOOST_CHECK(not streamIter);
  
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_run_2 )
{
  XtcFileName names[] = {
      XtcFileName("e1-r001-s00-c00.xtc"),
      XtcFileName("e1-r003-s00-c00.xtc"),
      XtcFileName("e1-r002-s01-c00.xtc"),
      XtcFileName("e1-r001-s01-c00.xtc"),
      XtcFileName("e1-r002-s00-c00.xtc"),
  };
  const unsigned num_names = sizeof names / sizeof names[0];

  RunFileIterList iter(&names[0], &names[num_names], MergeOneStream);

  boost::shared_ptr<StreamFileIterI> streamIter;
  boost::shared_ptr<ChunkFileIterI> chunkIter;
  XtcFileName name;

  streamIter = iter.next();
  BOOST_REQUIRE(streamIter);
  BOOST_CHECK_EQUAL(iter.run(), 0U);
  
  chunkIter = streamIter->next();
  BOOST_CHECK_EQUAL(streamIter->stream(), 0U);
  BOOST_REQUIRE(chunkIter);

  for (unsigned i = 0; i != num_names; ++ i) {
    name = chunkIter->next();
    BOOST_CHECK_EQUAL(name.path(), names[i].path());
  }

  name = chunkIter->next();
  BOOST_CHECK(name.path().empty());

  chunkIter = streamIter->next();
  BOOST_CHECK(not chunkIter);

  streamIter = iter.next();
  BOOST_CHECK(not streamIter);
  
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_run_3 )
{
  XtcFileName names[] = {
      XtcFileName("e1-r001-s00-c00.xtc"),
      XtcFileName("e1-r003-s00-c00.xtc"),
      XtcFileName("e1-r002-s01-c00.xtc"),
      XtcFileName("e1-r001-s01-c00.xtc"),
      XtcFileName("e1-r002-s00-c00.xtc"),
  };
  const unsigned num_names = sizeof names / sizeof names[0];

  RunFileIterList iter(&names[0], &names[num_names], MergeNoChunking);

  boost::shared_ptr<StreamFileIterI> streamIter;
  boost::shared_ptr<ChunkFileIterI> chunkIter;
  XtcFileName name;

  streamIter = iter.next();
  BOOST_REQUIRE(streamIter);
  BOOST_CHECK_EQUAL(iter.run(), 0U);
  
  for (unsigned i = 0; i != num_names; ++ i) {

    chunkIter = streamIter->next();
    BOOST_CHECK_EQUAL(streamIter->stream(), i);
    BOOST_REQUIRE(chunkIter);
    name = chunkIter->next();
    BOOST_CHECK_EQUAL(name.path(), names[i].path());
    name = chunkIter->next();
    BOOST_CHECK(name.path().empty());
  
  }

  chunkIter = streamIter->next();
  BOOST_CHECK(not chunkIter);

  streamIter = iter.next();
  BOOST_CHECK(not streamIter);
  
}

