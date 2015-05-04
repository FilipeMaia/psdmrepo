#include "Translator/firstPrimeGreaterOrEqualTo.h"

#define BOOST_TEST_MODULE translator-unit-test
#include <boost/test/included/unit_test.hpp>

/*
 * Simple test suite for module translator-unit-test.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

class TestFixture {
public:
  TestFixture()  {}
};

BOOST_FIXTURE_TEST_SUITE( TranslatorTest, TestFixture );

// ==============================================================

BOOST_AUTO_TEST_CASE( test_prime )
{   
  // not all answers are primes, for input over a threshold, an odd number is returned
  static unsigned testVals[] =    {0,1,2,3,4,32, 359,10000,104728, 104729,104730,43453,unsigned(9999999999)};
  static unsigned testAnswers[] = {2,2,2,3,5,37, 359,10007,104729, 104729,104731,43457,unsigned(1410065407)};
  for (unsigned i = 0; i < sizeof(testVals)/sizeof(unsigned); ++i) {
    unsigned prime = firstPrimeGreaterOrEqualTo(testVals[i]);
    BOOST_CHECK_EQUAL( prime, testAnswers[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
