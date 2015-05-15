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

// special destroyer for owned array data
template <typename ElemType>
struct _array_delete {
  void operator()(ElemType* ptr) const { delete [] ptr; }
};

const unsigned DataSize = 24;
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

  BOOST_CHECK_EQUAL ( arr.strides()[0], 12 ) ;
  BOOST_CHECK_EQUAL ( arr.strides()[1], 4 ) ;
  BOOST_CHECK_EQUAL ( arr.strides()[2], 1 ) ;

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[1][2][3], gdata[23] ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_strides_def )
{
  // default strides
  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr(gdata, dims);
  int strides[3] = {12,4,1};
  arr.strides(strides);

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[0][0][1], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr[0][1][0], gdata[4] ) ;
  BOOST_CHECK_EQUAL ( arr[1][0][0], gdata[12] ) ;
  BOOST_CHECK_EQUAL ( arr[1][2][3], gdata[23] ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_strides )
{
  // Fortran-like memory layout
  unsigned dims[3] = {4,3,2};
  ndarray<int,3> arr(gdata, dims, ndns::Fortran);

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[1][0][0], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr[0][1][0], gdata[4] ) ;
  BOOST_CHECK_EQUAL ( arr[0][0][1], gdata[12] ) ;
  BOOST_CHECK_EQUAL ( arr[3][2][1], gdata[23] ) ;

  int strides[3] = {1, 4, 12};
  arr.strides(strides);

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[1][0][0], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr[0][1][0], gdata[4] ) ;
  BOOST_CHECK_EQUAL ( arr[0][0][1], gdata[12] ) ;
  BOOST_CHECK_EQUAL ( arr[3][2][1], gdata[23] ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_swap )
{
  ndarray<int,3> arr0;
  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr1(gdata, dims);

  BOOST_CHECK_EQUAL ( arr0.shape()[0], 0U ) ;
  BOOST_CHECK_EQUAL ( arr0.shape()[1], 0U ) ;
  BOOST_CHECK_EQUAL ( arr0.shape()[2], 0U ) ;

  BOOST_CHECK_EQUAL ( arr0.strides()[0], 0 ) ;
  BOOST_CHECK_EQUAL ( arr0.strides()[1], 0 ) ;
  BOOST_CHECK_EQUAL ( arr0.strides()[2], 0 ) ;

  BOOST_CHECK_EQUAL ( arr1.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr1.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr1.shape()[2], 4U ) ;

  BOOST_CHECK_EQUAL ( arr1.strides()[0], 12 ) ;
  BOOST_CHECK_EQUAL ( arr1.strides()[1], 4 ) ;
  BOOST_CHECK_EQUAL ( arr1.strides()[2], 1 ) ;

  BOOST_CHECK_EQUAL ( arr1[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr1[1][2][3], gdata[23] ) ;

  arr0.swap(arr1);

  BOOST_CHECK_EQUAL ( arr1.shape()[0], 0U ) ;
  BOOST_CHECK_EQUAL ( arr1.shape()[1], 0U ) ;
  BOOST_CHECK_EQUAL ( arr1.shape()[2], 0U ) ;

  BOOST_CHECK_EQUAL ( arr1.strides()[0], 0 ) ;
  BOOST_CHECK_EQUAL ( arr1.strides()[1], 0 ) ;
  BOOST_CHECK_EQUAL ( arr1.strides()[2], 0 ) ;

  BOOST_CHECK_EQUAL ( arr0.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr0.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr0.shape()[2], 4U ) ;

  BOOST_CHECK_EQUAL ( arr0.strides()[0], 12 ) ;
  BOOST_CHECK_EQUAL ( arr0.strides()[1], 4 ) ;
  BOOST_CHECK_EQUAL ( arr0.strides()[2], 1 ) ;

  BOOST_CHECK_EQUAL ( arr0[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr0[1][2][3], gdata[23] ) ;

}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_new_array )
{
  // Test internal memory allocation
  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr1(dims);

  BOOST_CHECK ( not arr1.empty() ) ;
  BOOST_CHECK_EQUAL ( arr1.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr1.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr1.shape()[2], 4U ) ;
  BOOST_CHECK_EQUAL ( arr1.size(), DataSize ) ;

  ndarray<int,3> arr2 = make_ndarray<int>(2, 3, 4);

  BOOST_CHECK ( not arr2.empty() ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[2], 4U ) ;
  BOOST_CHECK_EQUAL ( arr2.size(), DataSize ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_shared_ptr )
{
  boost::shared_ptr<int> shptr(new int[DataSize], _array_delete<int>());

  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr1(shptr, dims);

  BOOST_CHECK ( not arr1.empty() ) ;
  BOOST_CHECK_EQUAL ( arr1.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr1.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr1.shape()[2], 4U ) ;
  BOOST_CHECK_EQUAL ( arr1.size(), DataSize ) ;

  ndarray<int,3> arr2 = make_ndarray<int>(shptr, 2, 3, 4);

  BOOST_CHECK ( not arr2.empty() ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[2], 4U ) ;
  BOOST_CHECK_EQUAL ( arr2.size(), DataSize ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_sub )
{
  // Test sub-arrays
  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr1(dims);

  BOOST_CHECK ( not arr1.empty() ) ;
  BOOST_CHECK_EQUAL ( arr1.size(), DataSize ) ;

  // copy data over
  std::copy(gdata, gdata+DataSize, arr1.begin());

  BOOST_CHECK_EQUAL ( arr1[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr1[0][0][1], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr1[0][1][0], gdata[4] ) ;
  BOOST_CHECK_EQUAL ( arr1[1][0][0], gdata[12] ) ;
  BOOST_CHECK_EQUAL ( arr1[1][2][3], gdata[23] ) ;

  ndarray<int,2> arr2 = arr1[0];

  BOOST_CHECK_EQUAL ( arr2.shape()[0], 3U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[1], 4U ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[0], 4 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[1], 1 ) ;

  BOOST_CHECK_EQUAL ( arr2[0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr2[0][1], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr2[1][0], gdata[4] ) ;
  BOOST_CHECK_EQUAL ( arr2[2][3], gdata[11] ) ;

  arr2 = arr1[1];

  BOOST_CHECK_EQUAL ( arr2.shape()[0], 3U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[1], 4U ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[0], 4 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[1], 1 ) ;

  BOOST_CHECK_EQUAL ( arr2[0][0], gdata[12+0] ) ;
  BOOST_CHECK_EQUAL ( arr2[0][1], gdata[12+1] ) ;
  BOOST_CHECK_EQUAL ( arr2[1][0], gdata[12+4] ) ;
  BOOST_CHECK_EQUAL ( arr2[2][3], gdata[12+11] ) ;

  ndarray<int,1> arr3 = arr1[0][0];

  BOOST_CHECK_EQUAL ( arr3.shape()[0], 4U ) ;
  BOOST_CHECK_EQUAL ( arr3.strides()[0], 1 ) ;

  BOOST_CHECK_EQUAL ( arr3[0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr3[1], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr3[3], gdata[3] ) ;

  arr3 = arr1[1][2];

  BOOST_CHECK_EQUAL ( arr3.shape()[0], 4U ) ;
  BOOST_CHECK_EQUAL ( arr3.strides()[0], 1 ) ;

  BOOST_CHECK_EQUAL ( arr3[0], gdata[20] ) ;
  BOOST_CHECK_EQUAL ( arr3[1], gdata[21] ) ;
  BOOST_CHECK_EQUAL ( arr3[3], gdata[23] ) ;

}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_copy )
{
  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr1(gdata, dims);

  ndarray<int,3> arr2(arr1);

  BOOST_CHECK_EQUAL ( arr2.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[2], 4U ) ;

  BOOST_CHECK_EQUAL ( arr2.strides()[0], 12 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[1], 4 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[2], 1 ) ;

  BOOST_CHECK_EQUAL ( arr2[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr2[1][2][3], gdata[23] ) ;

  arr2 = arr1;

  BOOST_CHECK_EQUAL ( arr2.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[2], 4U ) ;

  BOOST_CHECK_EQUAL ( arr2.strides()[0], 12 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[1], 4 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[2], 1 ) ;

  BOOST_CHECK_EQUAL ( arr2[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr2[1][2][3], gdata[23] ) ;

}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_copy2 )
{
  unsigned dims[3] = {2,3,4};
  ndarray<const int,3> arr1(gdata, dims);

  ndarray<const int,3> arr2(arr1);

  BOOST_CHECK_EQUAL ( arr2.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[2], 4U ) ;

  BOOST_CHECK_EQUAL ( arr2.strides()[0], 12 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[1], 4 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[2], 1 ) ;

  BOOST_CHECK_EQUAL ( arr2[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr2[1][2][3], gdata[23] ) ;

  arr2 = arr1;

  BOOST_CHECK_EQUAL ( arr2.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[2], 4U ) ;

  BOOST_CHECK_EQUAL ( arr2.strides()[0], 12 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[1], 4 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[2], 1 ) ;

  BOOST_CHECK_EQUAL ( arr2[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr2[1][2][3], gdata[23] ) ;

}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_const )
{
  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr1(gdata, dims);

  ndarray<const int,3> arr2(arr1);

  BOOST_CHECK_EQUAL ( arr2.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[2], 4U ) ;

  BOOST_CHECK_EQUAL ( arr2.strides()[0], 12 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[1], 4 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[2], 1 ) ;

  BOOST_CHECK_EQUAL ( arr2[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr2[1][2][3], gdata[23] ) ;

  arr2 = arr1;

  BOOST_CHECK_EQUAL ( arr2.shape()[0], 2U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[1], 3U ) ;
  BOOST_CHECK_EQUAL ( arr2.shape()[2], 4U ) ;

  BOOST_CHECK_EQUAL ( arr2.strides()[0], 12 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[1], 4 ) ;
  BOOST_CHECK_EQUAL ( arr2.strides()[2], 1 ) ;

  BOOST_CHECK_EQUAL ( arr2[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr2[1][2][3], gdata[23] ) ;

}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_assign )
{
  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr(gdata, dims);

  ndarray<int,3> arr1 = arr.copy();

  arr1[0][0][0] = gdata[0] + 100;
  arr1[0][0][1] = gdata[1] + 100;
  arr1[0][0][2] = gdata[2] + 100;

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[0][0][1], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr[0][0][2], gdata[2] ) ;

  BOOST_CHECK_EQUAL ( arr1[0][0][0], gdata[0]+100 ) ;
  BOOST_CHECK_EQUAL ( arr1[0][0][1], gdata[1]+100 ) ;
  BOOST_CHECK_EQUAL ( arr1[0][0][2], gdata[2]+100 ) ;
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_reshape )
{
  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr(gdata, dims);

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[0][0][1], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr[0][1][0], gdata[4] ) ;
  BOOST_CHECK_EQUAL ( arr[1][0][0], gdata[12] ) ;
  BOOST_CHECK_EQUAL ( arr[1][2][3], gdata[23] ) ;

  unsigned dims2[3] = {6,2,2};
  arr.reshape(dims2);

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[0][0][1], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr[0][1][0], gdata[2] ) ;
  BOOST_CHECK_EQUAL ( arr[1][0][0], gdata[4] ) ;
  BOOST_CHECK_EQUAL ( arr[5][1][1], gdata[23] ) ;

  arr.reshape(dims, ndns::Fortran);

  BOOST_CHECK_EQUAL ( arr[0][0][0], gdata[0] ) ;
  BOOST_CHECK_EQUAL ( arr[1][0][0], gdata[1] ) ;
  BOOST_CHECK_EQUAL ( arr[0][1][0], gdata[2] ) ;
  BOOST_CHECK_EQUAL ( arr[0][0][1], gdata[6] ) ;
  BOOST_CHECK_EQUAL ( arr[1][2][3], gdata[23] ) ;
}
