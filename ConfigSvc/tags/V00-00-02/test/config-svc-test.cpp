//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the config-svc-test.
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <sstream>
#include <vector>
#include <list>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigSvc/ConfigSvc.h"
#include "ConfigSvc/ConfigSvcImplFile.h"

#define BOOST_TEST_MODULE configSvcTest
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module config-svc-test.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

namespace {
  

// example input
const char* config = "\
# comment\n\
[section1]\n\
paramInt1 = 0\n\
paramInt2 = -1000\n\
paramInt3 = 1000\n\
\n\
paramFloat1 = 0.123456\n\
paramFloat2 = 1e10\n\
paramFloat3 = -1e10\n\
paramFloat4 = 1e-10\n\
\n\
paramBool1 = 0\n\
paramBool2 = 1\n\
paramBool3 = no\n\
paramBool4 = yes\n\
paramBool5 = off\n\
paramBool6 = on\n\
paramBool7 = FALSE\n\
paramBool8 = True\n\
\n\
paramString1 = \n\
paramString2 = OneWord \n\
paramString3 = Two Words \n\
\n\
paramList1 = \n\
paramList2 = 1000 \n\
paramList3 = 2000 3000 \n\
";


}

class TestFixture {
public:
  TestFixture() {
    std::istringstream stream(config);
    // start with reading configuration file
    std::auto_ptr<ConfigSvc::ConfigSvcImplI> cfgImpl ( new ConfigSvc::ConfigSvcImplFile(stream) );
    // initialize config service
    ConfigSvc::ConfigSvc::init(cfgImpl);
  }
};

BOOST_GLOBAL_FIXTURE( TestFixture );

// ==============================================================

BOOST_AUTO_TEST_CASE( test_exceptions )
{
  ConfigSvc::ConfigSvc cfgsvc;
  
  BOOST_CHECK_THROW(cfgsvc.get("unknown_section", "unknown_param"), ConfigSvc::ExceptionMissing);
  BOOST_CHECK_THROW(cfgsvc.get("section1", "unknown_param"), ConfigSvc::ExceptionMissing);
  BOOST_CHECK_THROW(cfgsvc.getList("unknown_section", "unknown_param"), ConfigSvc::ExceptionMissing);
  BOOST_CHECK_THROW(cfgsvc.getList("section1", "unknown_param"), ConfigSvc::ExceptionMissing);

  BOOST_CHECK_NO_THROW(cfgsvc.get("unknown_section", "unknown_param", 0));
  BOOST_CHECK_NO_THROW(cfgsvc.get("section1", "unknown_param", 0));
  BOOST_CHECK_NO_THROW(cfgsvc.getList("unknown_section", "unknown_param", std::vector<int>()));
  BOOST_CHECK_NO_THROW(cfgsvc.getList("section1", "unknown_param", std::list<int>()));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_int )
{
  ConfigSvc::ConfigSvc cfgsvc;

  int val;
  
  val = cfgsvc.get("section1", "paramInt1");
  BOOST_CHECK_EQUAL(val, 0);
  val = cfgsvc.get("section1", "paramInt2");
  BOOST_CHECK_EQUAL(val, -1000);
  val = cfgsvc.get("section1", "paramInt3");
  BOOST_CHECK_EQUAL(val, 1000);
  val = cfgsvc.get("section1", "paramInt3", -1000000);
  BOOST_CHECK_EQUAL(val, 1000);
  val = cfgsvc.get("section1", "unknown_param", -1000000);
  BOOST_CHECK_EQUAL(val, -1000000);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_double )
{
  ConfigSvc::ConfigSvc cfgsvc;

  double val;
  
  val = cfgsvc.get("section1", "paramFloat1");
  BOOST_CHECK_EQUAL(val, 0.123456);
  val = cfgsvc.get("section1", "paramFloat2");
  BOOST_CHECK_EQUAL(val, 1e10);
  val = cfgsvc.get("section1", "paramFloat3");
  BOOST_CHECK_EQUAL(val, -1e10);
  val = cfgsvc.get("section1", "paramFloat4");
  BOOST_CHECK_EQUAL(val, 1e-10);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_bool )
{
  ConfigSvc::ConfigSvc cfgsvc;

  bool val;
  
  val = cfgsvc.get("section1", "paramBool1");
  BOOST_CHECK_EQUAL(val, false);
  val = cfgsvc.get("section1", "paramBool2");
  BOOST_CHECK_EQUAL(val, true);
  val = cfgsvc.get("section1", "paramBool3");
  BOOST_CHECK_EQUAL(val, false);
  val = cfgsvc.get("section1", "paramBool4");
  BOOST_CHECK_EQUAL(val, true);
  val = cfgsvc.get("section1", "paramBool5");
  BOOST_CHECK_EQUAL(val, false);
  val = cfgsvc.get("section1", "paramBool6");
  BOOST_CHECK_EQUAL(val, true);
  val = cfgsvc.get("section1", "paramBool7");
  BOOST_CHECK_EQUAL(val, false);
  val = cfgsvc.get("section1", "paramBool8");
  BOOST_CHECK_EQUAL(val, true);
  val = cfgsvc.get("section1", "paramBool8", false);
  BOOST_CHECK_EQUAL(val, true);
  val = cfgsvc.get("section1", "unknown_param", false);
  BOOST_CHECK_EQUAL(val, false);
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_str )
{
  ConfigSvc::ConfigSvc cfgsvc;

  std::string val;
  val = cfgsvc.getStr("section1", "paramString1");
  BOOST_CHECK_EQUAL(val, "");
  val = cfgsvc.getStr("section1", "paramString2");
  BOOST_CHECK_EQUAL(val, "OneWord");
  val = cfgsvc.getStr("section1", "paramString3");
  BOOST_CHECK_EQUAL(val, "Two Words");
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_charp )
{
  ConfigSvc::ConfigSvc cfgsvc;

  const char* val;
  val = cfgsvc.get("section1", "paramString1");
  BOOST_CHECK_EQUAL(val, std::string(""));
  val = cfgsvc.get("section1", "paramString2");
  BOOST_CHECK_EQUAL(val, std::string("OneWord"));
  val = cfgsvc.get("section1", "paramString3");
  BOOST_CHECK_EQUAL(val, std::string("Two Words"));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_list_str )
{
  ConfigSvc::ConfigSvc cfgsvc;

  std::vector<std::string> val;
  val = cfgsvc.getList("section1", "paramList1");
  BOOST_CHECK(val.empty());
  val = cfgsvc.getList("section1", "paramList2");
  BOOST_CHECK_EQUAL(val.size(), 1U);
  BOOST_CHECK_EQUAL(val[0], "1000");
  val = cfgsvc.getList("section1", "paramList3");
  BOOST_CHECK_EQUAL(val.size(), 2U);
  BOOST_CHECK_EQUAL(val[0], "2000");
  BOOST_CHECK_EQUAL(val[1], "3000");
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_list_int )
{
  ConfigSvc::ConfigSvc cfgsvc;

  std::vector<int> val;
  val = cfgsvc.getList("section1", "paramList1");
  BOOST_CHECK(val.empty());
  val = cfgsvc.getList("section1", "paramList2");
  BOOST_CHECK_EQUAL(val.size(), 1U);
  BOOST_CHECK_EQUAL(val[0], 1000);
  val = cfgsvc.getList("section1", "paramList3");
  BOOST_CHECK_EQUAL(val.size(), 2U);
  BOOST_CHECK_EQUAL(val[0], 2000);
  BOOST_CHECK_EQUAL(val[1], 3000);
}

