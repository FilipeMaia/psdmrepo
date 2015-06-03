//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the NDArrayUtils
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <string>
//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_hdf2psana/NDArrayConverter.h"
#include "Translator/NDArrayUtil.h"
#include "ndarray/ndarray.h"

using namespace std ;
using namespace Translator ;

#define BOOST_TEST_MODULE ndarray-utils-unit-test
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module psevt-unit-test.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

class TestFixture {
public:
  TestFixture()
  {
  }
};

BOOST_FIXTURE_TEST_SUITE( NDArrayUtilsTest, TestFixture );

// ==============================================================

BOOST_AUTO_TEST_CASE( test )
{
  typedef pair<const type_info *, string> TypeName;
  static vector< TypeName > typesNamesList;
  typesNamesList.push_back( TypeName(& typeid( ndarray<int8_t, 1>) , string("ndarray_int8_1")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<int16_t, 2>) , string("ndarray_int16_2")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<int32_t, 3>) , string("ndarray_int32_3")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<int64_t, 4>) , string("ndarray_int64_4")) );

  typesNamesList.push_back( TypeName(& typeid( ndarray<const int8_t, 1>) , string("ndarray_const_int8_1")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const short, 2>) , string("ndarray_const_int16_2")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const int32_t, 3>) , string("ndarray_const_int32_3")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const long, 4>) , string("ndarray_const_int64_4")) );

  typesNamesList.push_back( TypeName(& typeid( ndarray<unsigned char, 1>) , string("ndarray_uint8_1")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<unsigned short, 2>) , string("ndarray_uint16_2")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<uint32_t, 3>) , string("ndarray_uint32_3")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<unsigned long, 4>) , string("ndarray_uint64_4")) );

  typesNamesList.push_back( TypeName(& typeid( ndarray<const unsigned int8_t, 1>) , string("ndarray_const_uint8_1")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const uint16_t, 2>) , string("ndarray_const_uint16_2")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const unsigned, 3>) , string("ndarray_const_uint32_3")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const uint64_t, 4>) , string("ndarray_const_uint64_4")) );
  
  typesNamesList.push_back( TypeName(& typeid( ndarray<long double, 1>) , string("ndarray_float128_1")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<double , 2>) , string("ndarray_float64_2")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<float, 3>) , string("ndarray_float32_3")) );
  
  typesNamesList.push_back( TypeName(& typeid( ndarray<const long double, 1>) , string("ndarray_const_float128_1")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const double , 2>) , string("ndarray_const_float64_2")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const float, 3>) , string("ndarray_const_float32_3")) );

  for (unsigned idx = 0; idx < typesNamesList.size(); ++idx) {
    TypeName & typeName = typesNamesList[idx];
    const type_info *typePtr = typeName.first;
    const string & expectedGroupName = typeName.second;
    BOOST_CHECK_EQUAL( ndarrayGroupName(typePtr, psddl_hdf2psana::NDArrayParameters::SlowDimNotVlen), expectedGroupName );
  }

  typesNamesList.clear();

  typesNamesList.push_back( TypeName(& typeid( ndarray<int8_t, 1>) , string("ndarray_int8_1_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<int16_t, 2>) , string("ndarray_int16_2_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<int32_t, 3>) , string("ndarray_int32_3_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<int64_t, 4>) , string("ndarray_int64_4_vlen")) );

  typesNamesList.push_back( TypeName(& typeid( ndarray<const int8_t, 1>) , string("ndarray_const_int8_1_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const short, 2>) , string("ndarray_const_int16_2_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const int32_t, 3>) , string("ndarray_const_int32_3_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const long, 4>) , string("ndarray_const_int64_4_vlen")) );

  typesNamesList.push_back( TypeName(& typeid( ndarray<unsigned char, 1>) , string("ndarray_uint8_1_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<unsigned short, 2>) , string("ndarray_uint16_2_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<uint32_t, 3>) , string("ndarray_uint32_3_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<unsigned long, 4>) , string("ndarray_uint64_4_vlen")) );

  typesNamesList.push_back( TypeName(& typeid( ndarray<const unsigned int8_t, 1>) , string("ndarray_const_uint8_1_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const uint16_t, 2>) , string("ndarray_const_uint16_2_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const unsigned, 3>) , string("ndarray_const_uint32_3_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const uint64_t, 4>) , string("ndarray_const_uint64_4_vlen")) );
  
  typesNamesList.push_back( TypeName(& typeid( ndarray<long double, 1>) , string("ndarray_float128_1_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<double , 2>) , string("ndarray_float64_2_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<float, 3>) , string("ndarray_float32_3_vlen")) );
  
  typesNamesList.push_back( TypeName(& typeid( ndarray<const long double, 1>) , string("ndarray_const_float128_1_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const double , 2>) , string("ndarray_const_float64_2_vlen")) );
  typesNamesList.push_back( TypeName(& typeid( ndarray<const float, 3>) , string("ndarray_const_float32_3_vlen")) );

  for (unsigned idx = 0; idx < typesNamesList.size(); ++idx) {
    TypeName & typeName = typesNamesList[idx];
    const type_info *typePtr = typeName.first;
    const string & expectedGroupName = typeName.second;
    BOOST_CHECK_EQUAL( ndarrayGroupName(typePtr, psddl_hdf2psana::NDArrayParameters::SlowDimIsVlen), expectedGroupName );
  }
}

BOOST_AUTO_TEST_SUITE_END()
