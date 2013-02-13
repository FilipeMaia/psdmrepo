//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the DatasetTest.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "IData/Dataset.h"
#include "IData/Exceptions.h"

using namespace IData ;

#define BOOST_TEST_MODULE DatasetTest
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module DatasetTest.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

// "default defaults", everything is empty

BOOST_AUTO_TEST_CASE( test_def_def )
{
  Dataset ds;
  BOOST_CHECK(not ds.isFile());
  BOOST_CHECK_EQUAL(ds.expID(), 0U);
  BOOST_CHECK_EQUAL(ds.instrument(), "");
  BOOST_CHECK_EQUAL(ds.experiment(), "");
  BOOST_CHECK(ds.runs().empty());
  BOOST_CHECK(not ds.exists("exp"));
  BOOST_CHECK(not ds.exists("run"));
  BOOST_CHECK(not ds.exists("live"));
  BOOST_CHECK(not ds.exists("anything"));
  BOOST_CHECK_EQUAL(ds.value("exp"), "");
  BOOST_CHECK_EQUAL(ds.value("anything"), "");
}

// ==============================================================

// test some global defaults

BOOST_AUTO_TEST_CASE( test_def_1 )
{
  Dataset::setDefOption("a", "b");
  Dataset::setDefOption("live", "");

  Dataset ds;
  BOOST_CHECK(not ds.isFile());
  BOOST_CHECK_EQUAL(ds.expID(), 0U);
  BOOST_CHECK_EQUAL(ds.instrument(), "");
  BOOST_CHECK_EQUAL(ds.experiment(), "");
  BOOST_CHECK(ds.runs().empty());
  BOOST_CHECK(not ds.exists("exp"));
  BOOST_CHECK(not ds.exists("run"));
  BOOST_CHECK(ds.exists("live"));
  BOOST_CHECK(ds.exists("a"));
  BOOST_CHECK_EQUAL(ds.value("live"), "");
  BOOST_CHECK_EQUAL(ds.value("a"), "b");
}

// ==============================================================

// test for default experiment name in the format "expnumber"

BOOST_AUTO_TEST_CASE( test_def_exp_1 )
{
  Dataset::setAppExpName("10");

  Dataset ds;
  BOOST_CHECK(not ds.isFile());
  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.instrument(), "AMO");
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");
  BOOST_CHECK(ds.exists("exp"));
  BOOST_CHECK_EQUAL(ds.value("exp"), "10");
}

// ==============================================================

// test for default experiment name in the format "expname"

BOOST_AUTO_TEST_CASE( test_def_exp_2 )
{
  Dataset::setAppExpName("amo00209");

  Dataset ds;
  BOOST_CHECK(not ds.isFile());
  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.instrument(), "AMO");
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");
  BOOST_CHECK(ds.exists("exp"));
  BOOST_CHECK_EQUAL(ds.value("exp"), "amo00209");
}

// ==============================================================

// test for default experiment name in the format "INSTR/expname"

BOOST_AUTO_TEST_CASE( test_def_exp_3 )
{
  Dataset::setAppExpName("AMO/amo00209");

  Dataset ds;
  BOOST_CHECK(not ds.isFile());
  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.instrument(), "AMO");
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");
  BOOST_CHECK(ds.exists("exp"));
  BOOST_CHECK_EQUAL(ds.value("exp"), "AMO/amo00209");
}

// ==============================================================

// test few versions of experiment names that should fail

BOOST_AUTO_TEST_CASE( test_def_exp_fail )
{
  BOOST_CHECK_THROW(Dataset::setAppExpName("10a"), ExpNameException);
  BOOST_CHECK_THROW(Dataset::setAppExpName("-10"), ExpNameException);
  BOOST_CHECK_THROW(Dataset::setAppExpName("AMO:amo00209"), ExpNameException);
  BOOST_CHECK_THROW(Dataset::setAppExpName("AMO/sxrdaq10"), ExpNameException);
  BOOST_CHECK_THROW(Dataset::setAppExpName("AMO/10"), ExpNameException);
}

// ==============================================================

// test experiment name in the format "expnumber"

BOOST_AUTO_TEST_CASE( test_exp_1 )
{
  Dataset ds("exp=10");
  BOOST_CHECK(not ds.isFile());
  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.instrument(), "AMO");
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");
  BOOST_CHECK(ds.exists("exp"));
  BOOST_CHECK_EQUAL(ds.value("exp"), "10");
}

// ==============================================================

// test experiment name in the format "expname"

BOOST_AUTO_TEST_CASE( test_exp_2 )
{
  Dataset ds("exp=amo00209");
  BOOST_CHECK(not ds.isFile());
  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.instrument(), "AMO");
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");
  BOOST_CHECK(ds.exists("exp"));
  BOOST_CHECK_EQUAL(ds.value("exp"), "amo00209");
}

// ==============================================================

// test experiment name in the format "INSTR/expname"

BOOST_AUTO_TEST_CASE( test_exp_3 )
{
  Dataset ds("exp=AMO/amo00209");
  BOOST_CHECK(not ds.isFile());
  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.instrument(), "AMO");
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");
  BOOST_CHECK(ds.exists("exp"));
  BOOST_CHECK_EQUAL(ds.value("exp"), "AMO/amo00209");
}

// ==============================================================

// test few versions of experiment names that should fail

BOOST_AUTO_TEST_CASE( test_exp_fail )
{
  BOOST_CHECK_THROW(Dataset("exp=10a"), ExpNameException);
  BOOST_CHECK_THROW(Dataset("exp=-10"), ExpNameException);
  BOOST_CHECK_THROW(Dataset("exp=AMO:amo00209"), ExpNameException);
  BOOST_CHECK_THROW(Dataset("exp=AMO/sxrdaq10"), ExpNameException);
  BOOST_CHECK_THROW(Dataset("exp=AMO/10"), ExpNameException);
}

// ==============================================================

// test run number parsing

BOOST_AUTO_TEST_CASE( test_run_exceptions )
{
  BOOST_CHECK_NO_THROW(Dataset("run=1"));
  BOOST_CHECK_NO_THROW(Dataset("run=1,10"));
  BOOST_CHECK_NO_THROW(Dataset("run=1-100"));
  BOOST_CHECK_NO_THROW(Dataset("run=1,10-100,200,220-300"));
  BOOST_CHECK_THROW(Dataset("run=-1"), RunNumberSpecException);
  BOOST_CHECK_THROW(Dataset("run=x"), RunNumberSpecException);
  BOOST_CHECK_THROW(Dataset("run=2,x"), RunNumberSpecException);
  BOOST_CHECK_THROW(Dataset("run=2-x"), RunNumberSpecException);
  BOOST_CHECK_THROW(Dataset("run=1..5"), RunNumberSpecException);
  BOOST_CHECK_THROW(Dataset("run=1,10-x50"), RunNumberSpecException);
}

// ==============================================================

// test run number parsing

BOOST_AUTO_TEST_CASE( test_run_1 )
{
  Dataset ds("run=1");
  const Dataset::Runs& runs = ds.runs();
  Dataset::Runs::const_iterator it = runs.begin();

  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 1U);
  BOOST_CHECK_EQUAL(it->second, 1U);
  ++ it;
  BOOST_CHECK(it == runs.end());
}

// ==============================================================

// test run number parsing

BOOST_AUTO_TEST_CASE( test_run_2 )
{
  Dataset ds("run=1,10");
  const Dataset::Runs& runs = ds.runs();
  Dataset::Runs::const_iterator it = runs.begin();

  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 1U);
  BOOST_CHECK_EQUAL(it->second, 1U);
  ++ it;
  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 10U);
  BOOST_CHECK_EQUAL(it->second, 10U);
  ++ it;
  BOOST_CHECK(it == runs.end());
}

// ==============================================================

// test run number parsing

BOOST_AUTO_TEST_CASE( test_run_3 )
{
  Dataset ds("run=1-100");
  const Dataset::Runs& runs = ds.runs();
  Dataset::Runs::const_iterator it = runs.begin();

  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 1U);
  BOOST_CHECK_EQUAL(it->second, 100U);
  ++ it;
  BOOST_CHECK(it == runs.end());
}

// ==============================================================

// test run number parsing

BOOST_AUTO_TEST_CASE( test_run_4 )
{
  Dataset ds("run=1,10-100,200,220-300");
  const Dataset::Runs& runs = ds.runs();
  Dataset::Runs::const_iterator it = runs.begin();

  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 1U);
  BOOST_CHECK_EQUAL(it->second, 1U);
  ++ it;
  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 10U);
  BOOST_CHECK_EQUAL(it->second, 100U);
  ++ it;
  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 200U);
  BOOST_CHECK_EQUAL(it->second, 200U);
  ++ it;
  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 220U);
  BOOST_CHECK_EQUAL(it->second, 300U);
  ++ it;
  BOOST_CHECK(it == runs.end());
}

// ==============================================================

// test combined data

BOOST_AUTO_TEST_CASE( test_run_combo_1 )
{
  BOOST_REQUIRE_NO_THROW(Dataset("exp=10:run=1"));
  Dataset ds("exp=10:run=1");

  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.instrument(), "AMO");
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");

  const Dataset::Runs& runs = ds.runs();
  Dataset::Runs::const_iterator it = runs.begin();

  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 1U);
  BOOST_CHECK_EQUAL(it->second, 1U);
  ++ it;
  BOOST_CHECK(it == runs.end());
}

// ==============================================================

// test combined data

BOOST_AUTO_TEST_CASE( test_run_combo_2 )
{
  BOOST_REQUIRE_NO_THROW(Dataset("exp=amo00209:run=1,10-100,200,220-300"));
  Dataset ds("exp=amo00209:run=1,10-100,200,220-300");

  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.instrument(), "AMO");
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");

  const Dataset::Runs& runs = ds.runs();
  Dataset::Runs::const_iterator it = runs.begin();

  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 1U);
  BOOST_CHECK_EQUAL(it->second, 1U);
  ++ it;
  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 10U);
  BOOST_CHECK_EQUAL(it->second, 100U);
  ++ it;
  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 200U);
  BOOST_CHECK_EQUAL(it->second, 200U);
  ++ it;
  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 220U);
  BOOST_CHECK_EQUAL(it->second, 300U);
  ++ it;
  BOOST_CHECK(it == runs.end());
}

// ==============================================================

// test combined data

BOOST_AUTO_TEST_CASE( test_run_combo_3 )
{
  BOOST_REQUIRE_NO_THROW(Dataset(":dir=/u2/data/path:h5:live:exp=amo00209:run=1:"));
  Dataset ds(":dir=/u2/data/path:h5:live:exp=amo00209:run=1:");

  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.instrument(), "AMO");
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");

  const Dataset::Runs& runs = ds.runs();
  Dataset::Runs::const_iterator it = runs.begin();

  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 1U);
  BOOST_CHECK_EQUAL(it->second, 1U);
  ++ it;
  BOOST_CHECK(it == runs.end());

  BOOST_CHECK(ds.exists("dir"));
  BOOST_CHECK_EQUAL(ds.value("dir"), "/u2/data/path");
  BOOST_CHECK(ds.exists("h5"));
  BOOST_CHECK_EQUAL(ds.value("h5"), "");
  BOOST_CHECK(ds.exists("live"));
  BOOST_CHECK_EQUAL(ds.value("live"), "");

}

// ==============================================================

// test file name

BOOST_AUTO_TEST_CASE( test_filename_1 )
{
  BOOST_REQUIRE_NO_THROW(Dataset("e100-r0200-s00-c00.xtc"));
  Dataset ds("e100-r0200-s00-c00.xtc");

  BOOST_CHECK(ds.isFile());
  BOOST_CHECK(ds.exists("xtc"));
  BOOST_CHECK_EQUAL(ds.expID(), 100U);
  BOOST_CHECK_EQUAL(ds.experiment(), "sxr27211");
  BOOST_CHECK_EQUAL(ds.dirName(), "/reg/d/psdm/SXR/sxr27211/xtc");

  const Dataset::Runs& runs = ds.runs();
  Dataset::Runs::const_iterator it = runs.begin();

  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 200U);
  BOOST_CHECK_EQUAL(it->second, 200U);
  ++ it;
  BOOST_CHECK(it == runs.end());

  const Dataset::NameList& files = ds.files();
  Dataset::NameList::const_iterator nit = files.begin();

  BOOST_REQUIRE(nit != files.end());
  BOOST_CHECK_EQUAL(*nit, "e100-r0200-s00-c00.xtc");
  ++ nit;
  BOOST_CHECK(nit == files.end());
}

// ==============================================================

// test file name

BOOST_AUTO_TEST_CASE( test_filename_2 )
{
  BOOST_REQUIRE_NO_THROW(Dataset("amo00209-r0123.h5"));
  Dataset ds("amo00209-r0123.h5");

  BOOST_CHECK(ds.isFile());
  BOOST_CHECK(ds.exists("h5"));
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");
  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.dirName(), "/reg/d/psdm/AMO/amo00209/hdf5");

  const Dataset::Runs& runs = ds.runs();
  Dataset::Runs::const_iterator it = runs.begin();

  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 123U);
  BOOST_CHECK_EQUAL(it->second, 123U);
  ++ it;
  BOOST_CHECK(it == runs.end());

  const Dataset::NameList& files = ds.files();
  Dataset::NameList::const_iterator nit = files.begin();

  BOOST_REQUIRE(nit != files.end());
  BOOST_CHECK_EQUAL(*nit, "amo00209-r0123.h5");
  ++ nit;
  BOOST_CHECK(nit == files.end());
}

// ==============================================================

// test file name

BOOST_AUTO_TEST_CASE( test_filename_3 )
{
  BOOST_REQUIRE_NO_THROW(Dataset("/reg/d/psdm/AMO/amo00209/hdf5/amo00209-r0123.h5"));
  Dataset ds("/reg/d/psdm/AMO/amo00209/hdf5/amo00209-r0123.h5");

  BOOST_CHECK(ds.isFile());
  BOOST_CHECK(ds.exists("h5"));
  BOOST_CHECK_EQUAL(ds.experiment(), "amo00209");
  BOOST_CHECK_EQUAL(ds.expID(), 10U);
  BOOST_CHECK_EQUAL(ds.dirName(), "/reg/d/psdm/AMO/amo00209/hdf5");

  const Dataset::Runs& runs = ds.runs();
  Dataset::Runs::const_iterator it = runs.begin();

  BOOST_REQUIRE(it != runs.end());
  BOOST_CHECK_EQUAL(it->first, 123U);
  BOOST_CHECK_EQUAL(it->second, 123U);
  ++ it;
  BOOST_CHECK(it == runs.end());

  const Dataset::NameList& files = ds.files();
  Dataset::NameList::const_iterator nit = files.begin();

  BOOST_REQUIRE(nit != files.end());
  BOOST_CHECK_EQUAL(*nit, "/reg/d/psdm/AMO/amo00209/hdf5/amo00209-r0123.h5");
  ++ nit;
  BOOST_CHECK(nit == files.end());
}

