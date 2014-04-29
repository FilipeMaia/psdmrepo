//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the psevt-unit-test.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/Event.h"
#include "PSEvt/Exceptions.h"
#include "PSEvt/ProxyDict.h"
#include "pdsdata/xtc/DetInfo.hh"

using namespace PSEvt ;

#define BOOST_TEST_MODULE psevt-unit-test
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module psevt-unit-test.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

class TestFixture {
public:
  TestFixture()
    : amap(boost::make_shared<AliasMap>())
    , dict(boost::make_shared<ProxyDict>(amap))
    , evt(dict)
    , di1(0, Pds::DetInfo::Detector(1), 1, Pds::DetInfo::Device(1), 1)
    , di2(0, Pds::DetInfo::Detector(2), 2, Pds::DetInfo::Device(2), 2)
  {
  }

  boost::shared_ptr<AliasMap> amap;
  boost::shared_ptr<ProxyDict> dict;
  Event evt;
  Pds::DetInfo di1;
  Pds::DetInfo di2;
};

BOOST_FIXTURE_TEST_SUITE( proxyDictTest, TestFixture );

// ==============================================================

BOOST_AUTO_TEST_CASE( test )
{
  boost::shared_ptr<int> p1(new int(1));
  BOOST_CHECK_NO_THROW( evt.put(p1) );

  boost::shared_ptr<int> p2(new int(2));
  BOOST_CHECK_NO_THROW( evt.put(p2, "p2") );

  // Adding the same key again must throw
  BOOST_CHECK_THROW( evt.put(p2, "p2"), ExceptionDuplicateKey );
  
  boost::shared_ptr<int> p3(new int(3));
  BOOST_CHECK_NO_THROW( evt.put(p3, di1) );
  // second time around throws
  BOOST_CHECK_THROW( evt.put(p3, di1), ExceptionDuplicateKey );
  
  boost::shared_ptr<int> p4(new int(4));
  BOOST_CHECK_NO_THROW( evt.put(p4, di2) );

  BOOST_CHECK_NO_THROW( evt.put(p3, di1, "p3") );
  BOOST_CHECK_NO_THROW( evt.put(p4, di2, "p4") );

  boost::shared_ptr<Proxy<int> > pp1( new DataProxy<int>(p1) );
  BOOST_CHECK_NO_THROW( evt.putProxy(pp1, "pp1") );
  
  boost::shared_ptr<Proxy<int> > pp2( new DataProxy<int>(p2) );
  BOOST_CHECK_NO_THROW( evt.putProxy(pp2, "pp2") );
  
  /// check that all proxies are still there
  
  BOOST_CHECK( evt.exists<int>() );
  BOOST_CHECK( evt.exists<int>("p2") );
  BOOST_CHECK( evt.exists<int>(di1) );
  BOOST_CHECK( evt.exists<int>(di2) );
  BOOST_CHECK( evt.exists<int>(di1, "p3") );
  BOOST_CHECK( evt.exists<int>(di2, "p4") );

  BOOST_CHECK( not evt.exists<int>("not") );
  BOOST_CHECK( not evt.exists<int>(di1, "not") );

  BOOST_CHECK( evt.exists<int>("pp1") );
  BOOST_CHECK( evt.exists<int>("pp2") );

  /// get data back
  
  p1 = evt.get();
  BOOST_CHECK( p1.get() );
  BOOST_CHECK_EQUAL( *p1, 1 );
  
  p2 = evt.get("p2");
  BOOST_CHECK( p2.get() );
  BOOST_CHECK_EQUAL( *p2, 2 );
  
  p3 = evt.get(di1);
  BOOST_CHECK( p3.get() );
  BOOST_CHECK_EQUAL( *p3, 3 );
  
  p4 = evt.get(di2);
  BOOST_CHECK( p4.get() );
  BOOST_CHECK_EQUAL( *p4, 4 );
  
  p3 = evt.get(di1, "p3");
  BOOST_CHECK( p3.get() );
  BOOST_CHECK_EQUAL( *p3, 3 );
  
  p4 = evt.get(di2, "p4");
  BOOST_CHECK( p4.get() );
  BOOST_CHECK_EQUAL( *p4, 4 );
  
  p1 = evt.get("pp1");
  BOOST_CHECK( p1.get() );
  BOOST_CHECK_EQUAL( *p1, 1 );
  
  p2 = evt.get("pp2");
  BOOST_CHECK( p2.get() );
  BOOST_CHECK_EQUAL( *p2, 2 );  
  
  /// remove some proxies
  
  BOOST_CHECK( evt.remove<int>() );
  BOOST_CHECK( not evt.exists<int>() );

  BOOST_CHECK( evt.remove<int>("p2") );
  BOOST_CHECK( not evt.exists<int>("p2") );

  BOOST_CHECK( evt.remove<int>(di1) );
  BOOST_CHECK( not evt.exists<int>(di1) );

  BOOST_CHECK( evt.remove<int>(di1, "p3") );
  BOOST_CHECK( not evt.exists<int>(di1, "p3") );

  BOOST_CHECK( not evt.remove<int>(di1, "p333") );
  BOOST_CHECK( not evt.exists<int>(di1, "p333") );
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_keys )
{
  boost::shared_ptr<int> p0(new int(0));
  BOOST_CHECK_NO_THROW( evt.put(p0) );

  boost::shared_ptr<int> p1(new int(1));
  BOOST_CHECK_NO_THROW( evt.put(p1, di1) );

  boost::shared_ptr<int> p2(new int(2));
  BOOST_CHECK_NO_THROW( evt.put(p2, di2) );

  boost::shared_ptr<int> p3(new int(3));
  BOOST_CHECK_NO_THROW( evt.put(p3, di2, "p3") );

  std::list<EventKey> keys = evt.keys();
  BOOST_CHECK_EQUAL( keys.size(), 4U );

  keys = evt.keys(Source("*.0:*.0"));
  BOOST_CHECK_EQUAL( keys.size(), 0U );

  keys = evt.keys(Source("*.1:*.1"));
  BOOST_CHECK_EQUAL( keys.size(), 1U );

  keys = evt.keys(Source("*.2:*.2"));
  BOOST_CHECK_EQUAL( keys.size(), 2U );

}

// ==============================================================

BOOST_AUTO_TEST_SUITE_END()
