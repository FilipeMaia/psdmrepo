#ifndef RDBMYSQL_TYPETRAITS_HH
#define RDBMYSQL_TYPETRAITS_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TypeTraits.
//
// Environment:
//      This software was developed for the BaBar collaboration.  If you
//      use all or part of it, please give an appropriate acknowledgement.
//
// Author List:
//      Andy Salnikov
//
// Copyright Information:
//      Copyright (C) 2005 SLAC
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <sstream>
#include <iomanip>
#include <string>
#include <limits>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "RdbMySQL/Buffer.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RdbMySQL {

/**
 *  This class is an iterator for the rows in the result set.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see Result
 *  @see Row
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

/// unspecialized one does not have anything in it
template <typename T>
struct TypeTraits {
};

// default implementation of methods
template <typename T>
struct TypeTraitsDefStr2Val {
  static bool str2val ( const char* str, unsigned int size, T& val ) {
    std::istringstream is ( std::string( str, size ) ) ;
    is >> val ;
    // what should happen is that when input is OK then we should have reached 
    // the end of stream, but fail flag should still be clear
    return ! is.fail() && is.eof() ;
  }
};

template <typename T, typename LongType>
struct TypeTraitsDefStr2ValViaLong {
  static bool str2val ( const char* str, unsigned int size, T& val ) {
    LongType v ;
    if ( ! TypeTraitsDefStr2Val<LongType>::str2val( str, size, v ) ) return false ;
    if ( v < std::numeric_limits<T>::min() || v > std::numeric_limits<T>::max() ) return false ;
    val = T(v) ;
    return true ;
  }
};

template <typename T>
struct TypeTraitsDefVal2Str {
  static std::string val2str ( const T& val ) {
    std::ostringstream os ;
    os << val ;
    return os.str() ;
  }
};

template <typename T>
struct TypeTraitsDefVal2StrFloat {
  static std::string val2str ( const T& val ) {
    std::ostringstream str ;
    str << std::setprecision(std::numeric_limits<T>::digits10+1) << val ;
    return str.str() ;
  }
};

template <typename T, typename LongType>
struct TypeTraitsDefVal2StrViaLong {
  static std::string val2str ( const T& val ) {
    return TypeTraits<LongType>::val2str(LongType(val)) ;
  }
};


// ---------------------- specialization for specific types --------------------------

/// specialization for string type
template <>
struct TypeTraits<std::string> {

  typedef std::string value_type ;

  static bool str2val ( const char* str, unsigned long size, value_type& val ) {
    val = std::string ( str, size ) ;
    return true ;
  }

  static std::string val2str ( const value_type& val ) {
    return val ;
  }

};

/// specialization for "buffer" type
template <>
struct TypeTraits< Buffer > {

  typedef Buffer value_type ;

  static bool str2val ( const char* str, unsigned long size, value_type& val ) {
    val = value_type( str, size ) ;
    return true ;
  }

  static std::string val2str ( const value_type& val ) {
    return val.data() ? std::string( val.data(), val.size() ) : std::string() ;
  }

};

/// specialization for bool type
template <>
struct TypeTraits<bool> {

  typedef bool value_type ;

  static bool str2val ( const char* str, unsigned int size, value_type& val ) {
    unsigned long v ;
    if ( ! TypeTraitsDefStr2Val<unsigned long>::str2val( str, size, v ) ) return false ;
    val = value_type(v) ;
    return true ;
  }

  static std::string val2str ( const value_type& val ) {
    /// old versions of MySQL do not have TRUE/FALSE support
    return std::string ( val ? "1" : "0" ) ;
  }

};

/// specialization for long type
template <>
struct TypeTraits<long> : public TypeTraitsDefVal2Str<long>,
				  public TypeTraitsDefStr2Val<long> {
};

/// specialization for unsigned long type
template <>
struct TypeTraits<unsigned long> : public TypeTraitsDefVal2Str<unsigned long>,
					   public TypeTraitsDefStr2Val<unsigned long> {
};

/// specialization for signed char type
template <>
struct TypeTraits<signed char> : TypeTraitsDefStr2ValViaLong<signed char, long>,
					 TypeTraitsDefVal2StrViaLong<signed char, long> {
};

/// specialization for unsigned char type
template <>
struct TypeTraits<unsigned char> : TypeTraitsDefStr2ValViaLong<unsigned char, unsigned long>,
					   TypeTraitsDefVal2StrViaLong<unsigned char, unsigned long> {
};

/// specialization for signed short type
template <>
struct TypeTraits<short> : TypeTraitsDefStr2ValViaLong<short, long>,
				   TypeTraitsDefVal2StrViaLong<short, long> {
};

/// specialization for unsigned short type
template <>
struct TypeTraits<unsigned short> : TypeTraitsDefStr2ValViaLong<unsigned short, unsigned long>,
					    TypeTraitsDefVal2StrViaLong<unsigned short, unsigned long> {
};

/// specialization for int type
template <>
struct TypeTraits<int> : TypeTraitsDefStr2ValViaLong< int, long>,
				 TypeTraitsDefVal2StrViaLong< int, long> {
};

/// specialization for unsigned int type
template <>
struct TypeTraits<unsigned int> : TypeTraitsDefStr2ValViaLong<unsigned int, unsigned long>,
					  TypeTraitsDefVal2StrViaLong<unsigned int, unsigned long> {
};

/// specialization for float type
template <>
struct TypeTraits<float> : TypeTraitsDefStr2Val<float>,
				   TypeTraitsDefVal2StrFloat<float> {
};

/// specialization for double type
template <>
struct TypeTraits<double> : TypeTraitsDefStr2Val<double>,
				    TypeTraitsDefVal2StrFloat<double> {
};

/// specialization for double type
template <>
struct TypeTraits<long double> : TypeTraitsDefStr2Val<long double>,
				         TypeTraitsDefVal2StrFloat<long double> {
};

} // namespace RdbMySQL

#endif // RDBMYSQL_TYPETRAITS_HH
