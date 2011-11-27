//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the ndarray-unit-test.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"

#define BOOST_TEST_MODULE ndarray_unit_test
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module ndarray-unit-test.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */


const int DataSize = 24;
int gdata[DataSize] = {
    0,1,2,3,4,5,6,7,8,9,
    10,11,12,13,14,15,16,17,18,19,
    20,21,22,23
};

// ==============================================================

BOOST_AUTO_TEST_CASE( test_notown )
{
  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr(gdata, dims);

  BOOST_CHECK_EQUAL ( arr.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr.shape()[2], 4U ) ;

  BOOST_CHECK_EQUAL ( arr.strides()[0], 12U ) ;
  BOOST_CHECK_EQUAL ( arr.strides()[1], 4U ) ;
  BOOST_CHECK_EQUAL ( arr.strides()[2], 1U ) ;

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[1][2][3], gdata[23] ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_own_copy )
{
  unsigned dims[3] = {2,3,4};
  int* data = new int[DataSize];
  std::copy(gdata, gdata+DataSize, data);
  ndarray<int,3> arr(data, dims, true);

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[1][2][3], gdata[23] ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_strides_def )
{
  unsigned dims[3] = {2,3,4};
  unsigned strides[3] = {12,4,1};
  ndarray<int,3> arr(gdata, dims, strides);

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[0][0][1], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr[0][1][0], gdata[4] ) ;
  BOOST_CHECK_EQUAL ( arr[1][0][0], gdata[12] ) ;
  BOOST_CHECK_EQUAL ( arr[1][2][3], gdata[23] ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_strides )
{
  unsigned dims[3] = {2,3,4};
  unsigned strides[3] = {12,4,1};
  ndarray<int,3> arr(gdata, dims, strides);

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[0][0][1], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr[0][1][0], gdata[4] ) ;
  BOOST_CHECK_EQUAL ( arr[1][0][0], gdata[12] ) ;
  BOOST_CHECK_EQUAL ( arr[1][2][3], gdata[23] ) ;
}
