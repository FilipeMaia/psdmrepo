//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the rdb-mysql-type-traits-test.
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "RdbMySQL/TypeTraits.h"

using namespace RdbMySQL ;
using namespace std ;

#define BOOST_TEST_MODULE rdb-mysql-type-traits-test
#include <boost/test/included/unit_test.hpp>

/**
 * Simple test suite for module rdb-mysql-type-traits-test.
 * See http://www.boost.org/doc/libs/1_36_0/libs/test/doc/html/index.html
 */

namespace {

  template <typename T> struct TypeName {} ;
  template <> struct TypeName<signed char> { static const char* typeName() { return "signed char" ; } } ;
  template <> struct TypeName<unsigned char> { static const char* typeName() { return "unsigned char" ; } } ;
  template <> struct TypeName<short> { static const char* typeName() { return "short" ; } } ;
  template <> struct TypeName<unsigned short> { static const char* typeName() { return "unsigned short" ; } } ;
  template <> struct TypeName<int> { static const char* typeName() { return "int" ; } } ;
  template <> struct TypeName<unsigned int> { static const char* typeName() { return "unsigned int" ; } } ;
  template <> struct TypeName<long> { static const char* typeName() { return "long" ; } } ;
  template <> struct TypeName<unsigned long> { static const char* typeName() { return "unsigned long" ; } } ;
  template <> struct TypeName<float> { static const char* typeName() { return "float" ; } } ;
  template <> struct TypeName<double> { static const char* typeName() { return "double" ; } } ;
  template <> struct TypeName<long double> { static const char* typeName() { return "long double" ; } } ;


  template <typename T>
  bool fromStr ( const char* valueStr )
  {
    cout << "Converting string \"" << valueStr << "\" to type " << TypeName<T>::typeName() << endl ;
    T val ;
    bool res = TypeTraits<T>::str2val ( valueStr, strlen(valueStr), val ) ;
    if ( res ) {
      cout << "Conversion successful, result = " << val << endl ;
    } else {
      cout << "Conversion failed" << endl ;
    }
    return res;
  }

  template <typename T>
  bool fromStrFail ( const char* valueStr )
  {
    cout << "Converting string \"" << valueStr << "\" to type " << TypeName<T>::typeName() << " (and expect failure)" << endl ;
    T val ;
    bool res = TypeTraits<T>::str2val ( valueStr, strlen(valueStr), val ) ;
    if ( res ) {
      cout << "Conversion successful, result = " << val << endl ;
    } else {
      cout << "Conversion failed" << endl ;
    }
    return not res;
  }

}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_schar )
{
  BOOST_CHECK(fromStr<signed char>("64"));
  BOOST_CHECK(fromStrFail<signed char> ( "128" ));
  BOOST_CHECK(fromStrFail<signed char> ( "-129" ));
  BOOST_CHECK(fromStrFail<signed char> ( "bad" ));
  BOOST_CHECK(fromStrFail<signed char> ( "0xBAD" ));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_uchar )
{
  BOOST_CHECK(fromStr<unsigned char> ( "64" ));
  BOOST_CHECK(fromStrFail<unsigned char> ( "256" ));
  BOOST_CHECK(fromStrFail<unsigned char> ( "-1" ));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_short )
{
  BOOST_CHECK(fromStr<short> ( "32767" ));
  BOOST_CHECK(fromStr<short> ( "-32768" ));
  BOOST_CHECK(fromStrFail<short> ( "32768" ));
  BOOST_CHECK(fromStrFail<short> ( "-32769" ));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_ushort )
{
  BOOST_CHECK(fromStr<unsigned short> ( "40000" ));
  BOOST_CHECK(fromStr<unsigned short> ( "65535" ));
  BOOST_CHECK(fromStrFail<unsigned short> ( "65536" ));
  BOOST_CHECK(fromStrFail<unsigned short> ( "-1" ));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_int32 )
{
  BOOST_CHECK(fromStr<int> ( "2147483647" ));
  BOOST_CHECK(fromStr<int> ( "-2147483648" ));
  BOOST_CHECK(fromStrFail<int> ( "2147483648" ));
  BOOST_CHECK(fromStrFail<int> ( "-2147483649" ));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_uint32 )
{
  BOOST_CHECK(fromStr<unsigned int> ( "2147483648" ));
  BOOST_CHECK(fromStr<unsigned int> ( "4294967295" ));
  BOOST_CHECK(fromStrFail<unsigned int> ( "4294967296" ));
  BOOST_CHECK(fromStrFail<unsigned int> ( "-1" ));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_int64 )
{
  BOOST_CHECK(fromStr<long> ( "9223372036854775807" ));
  BOOST_CHECK(fromStr<long> ( "-9223372036854775808" ));
  BOOST_CHECK(fromStrFail<long> ( "9223372036854775808" ));
  BOOST_CHECK(fromStrFail<long> ( "-9223372036854775809" ));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_uint64 )
{
  BOOST_CHECK(fromStr<unsigned long> ( "4294967296" ));
  BOOST_CHECK(fromStr<unsigned long> ( "18446744073709551615" ));
  BOOST_CHECK(fromStrFail<unsigned long> ( "18446744073709551616" ));
  BOOST_CHECK(fromStrFail<unsigned long> ( "18446744073709551617" ));
  // below test is platform-dependent, disable
  //BOOST_CHECK(fromStrFail<unsigned long> ( "-1" ));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_float )
{
  BOOST_CHECK(fromStr<float> ( "1.0" ));
  BOOST_CHECK(fromStr<float> ( "1.0e10" ));
  BOOST_CHECK(fromStr<float> ( "0.125" ));
  BOOST_CHECK(fromStr<float> ( "3.4e+38" ));
  // below test is platform-dependent, disable
  //BOOST_CHECK(fromStrFail<float> ( "1.0e-310" ));
  BOOST_CHECK(fromStrFail<float> ( "1e+39" ));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_double )
{
  BOOST_CHECK(fromStr<double> ( "1.0" ));
  BOOST_CHECK(fromStr<double> ( "1e+39" ));
  BOOST_CHECK(fromStr<double> ( "1.0e-310" ));
  BOOST_CHECK(fromStrFail<double> ( "1e+310" ));
}

// ==============================================================

BOOST_AUTO_TEST_CASE( test_ldouble )
{
  BOOST_CHECK(fromStr<long double> ( "1.0" ));
  BOOST_CHECK(fromStr<long double> ( "1.0e-310" ));
  BOOST_CHECK(fromStr<long double> ( "1e+4932" ));
  BOOST_CHECK(fromStrFail<long double> ( "1e+5000" ));
}

// ==============================================================

