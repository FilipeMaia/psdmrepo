//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the O2OFileNameFactoryTest.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "O2OTranslator/O2OFileNameFactory.h"

using namespace O2OTranslator ;

#define BOOST_TEST_MODULE O2OFileNameFactoryTest
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module O2OFileNameFactoryTest.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  O2OFileNameFactory* f = 0 ;
  BOOST_CHECK_NO_THROW( f = new O2OFileNameFactory("{a}/{b}/{c/d}-{seq}-{seq2}-{seq10}.{ext}") ) ;
  BOOST_CHECK_NO_THROW( f->addKeyword("a","A") ) ;
  BOOST_CHECK_NO_THROW( f->addKeyword("b","B") ) ;
  BOOST_CHECK_NO_THROW( f->addKeyword("ext","ext") ) ;

  std::string path ;

  path = f->makePath ( 0 ) ;
  BOOST_CHECK_EQUAL ( path, "A/B/{c/d}-0-00-0000000000.ext" ) ;
  path = f->makePath ( 56 ) ;
  BOOST_CHECK_EQUAL ( path, "A/B/{c/d}-56-56-0000000056.ext" ) ;
  path = f->makePath ( 123456789 ) ;
  BOOST_CHECK_EQUAL ( path, "A/B/{c/d}-123456789-123456789-0123456789.ext" ) ;
  path = f->makePath ( O2OFileNameFactory::Family ) ;
  BOOST_CHECK_EQUAL ( path, "A/B/{c/d}-%d-%02d-%010d.ext" ) ;
  path = f->makePath ( O2OFileNameFactory::FamilyPattern ) ;
  BOOST_CHECK_EQUAL ( path, "A/B/{c/d}-[0-9]+-[0-9]{2}-[0-9]{10}.ext" ) ;
}

