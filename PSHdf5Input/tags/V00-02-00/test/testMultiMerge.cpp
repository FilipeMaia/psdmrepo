//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the testMultiMerge.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <vector>
#include <iterator>
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSHdf5Input/MultiMerge.h"

using namespace PSHdf5Input ;

#define BOOST_TEST_MODULE testMultiMerge
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module testMultiMerge.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

// ==============================================================

BOOST_AUTO_TEST_CASE( test_1 )
{
  int in1[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int in2[10] = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8};

  MultiMerge<int*> merger;
  merger.add(in1, in1+10);
  merger.add(in2, in2+10);
  
  int counter = 0;
  std::vector<int> res;
  for(; merger.next(std::back_inserter(res)); ++ counter, res.clear()) {
    
    std::cout << counter << ": ";
    std::copy(res.begin(), res.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    
    if (counter == 0) {
      BOOST_CHECK ( res.size() == 1 and res[0] == -1 ) ;        
    } else if (counter == 1) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 0 and res[1] == 0 ) ;        
    } else if (counter == 2) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 1 and res[1] == 1 ) ;        
    } else if (counter == 3) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 2 and res[1] == 2 ) ;        
    } else if (counter == 4) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 3 and res[1] == 3 ) ;        
    } else if (counter == 5) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 4 and res[1] == 4 ) ;        
    } else if (counter == 6) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 5 and res[1] == 5 ) ;        
    } else if (counter == 7) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 6 and res[1] == 6 ) ;        
    } else if (counter == 8) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 7 and res[1] == 7 ) ;        
    } else if (counter == 9) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 8 and res[1] == 8 ) ;        
    } else if (counter == 10) {
      BOOST_CHECK ( res.size() == 1 and res[0] == 9 ) ;        
    }
  }  
  BOOST_CHECK_EQUAL ( counter, 11 ) ;        
  
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_2 )
{
  int in1[5] = {0, 2, 4, 6, 8};
  int in2[5] = {1, 3, 5, 7, 9};
  int in3[5] = {2, 3, 6, 7, 8};
  int in4[5] = {1, 4, 5, 7, 9};

  MultiMerge<int*> merger;
  merger.add(in1, in1+5);
  merger.add(in2, in2+5);
  merger.add(in3, in3+5);
  merger.add(in4, in4+5);
  
  int counter = 0;
  std::vector<int> res;
  for(; merger.next(std::back_inserter(res)); ++ counter, res.clear()) {
    
    std::cout << counter << ": ";
    std::copy(res.begin(), res.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    
    if (counter == 0) {
      BOOST_CHECK ( res.size() == 1 and res[0] == 0 ) ;
    } else if (counter == 1) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 1 and res[1] == 1 ) ;
    } else if (counter == 2) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 2 and res[1] == 2 ) ;
    } else if (counter == 3) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 3 and res[1] == 3 ) ;
    } else if (counter == 4) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 4 and res[1] == 4 ) ;
    } else if (counter == 5) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 5 and res[1] == 5 ) ;
    } else if (counter == 6) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 6 and res[1] == 6 ) ;
    } else if (counter == 7) {
      BOOST_CHECK ( res.size() == 3 and res[0] == 7 and res[1] == 7 and res[2] == 7 ) ;
    } else if (counter == 8) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 8 and res[1] == 8 ) ;
    } else if (counter == 9) {
      BOOST_CHECK ( res.size() == 2 and res[0] == 9 and res[1] == 9 ) ;
    }
  }  
  BOOST_CHECK_EQUAL ( counter, 10 ) ;
  
}
