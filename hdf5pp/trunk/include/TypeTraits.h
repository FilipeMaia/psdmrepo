#ifndef HDF5PP_TYPETRAITS_H
#define HDF5PP_TYPETRAITS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TypeTraits.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5/hdf5.h"
#include "hdf5pp/Type.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Type traits library
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace hdf5pp {

struct TypeTraitsHelper {
  static Type string_h5type() ;
};

template <typename T>
struct TypeTraits  {
  static Type stored_type() { return T::stored_type() ; }
  static Type native_type() { return T::native_type() ; }
  static const void* address( const T& value ) { return static_cast<const void*>(&value) ; }
};

#define TYPE_TRAITS_SIMPLE(CPP_TYPE,H5_TYPE) \
  template <> struct TypeTraits<CPP_TYPE> { \
    static Type stored_type() { return Type::LockedType(H5_TYPE); } \
    static Type native_type() { return Type::LockedType(H5_TYPE); } \
    static const void* address( const CPP_TYPE& value ) { return static_cast<const void*>(&value) ; } \
  }

TYPE_TRAITS_SIMPLE(float,H5T_NATIVE_FLOAT);
TYPE_TRAITS_SIMPLE(double,H5T_NATIVE_DOUBLE);
TYPE_TRAITS_SIMPLE(int8_t,H5T_NATIVE_INT8);
TYPE_TRAITS_SIMPLE(uint8_t,H5T_NATIVE_UINT8);
TYPE_TRAITS_SIMPLE(int16_t,H5T_NATIVE_INT16);
TYPE_TRAITS_SIMPLE(uint16_t,H5T_NATIVE_UINT16);
TYPE_TRAITS_SIMPLE(int32_t,H5T_NATIVE_INT32);
TYPE_TRAITS_SIMPLE(uint32_t,H5T_NATIVE_UINT32);
TYPE_TRAITS_SIMPLE(int64_t,H5T_NATIVE_INT64);
TYPE_TRAITS_SIMPLE(uint64_t,H5T_NATIVE_UINT64);

#undef TYPE_TRAITS_SIMPLE

template <>
struct TypeTraits<const char*> {
  typedef const char* vtype ;
  static Type stored_type() { return TypeTraitsHelper::string_h5type() ; }
  static Type native_type() { return TypeTraitsHelper::string_h5type() ; }
  static const void* address( const vtype& value ) { return static_cast<const void*>(&value) ; }
};


} // namespace hdf5pp

#endif // HDF5PP_TYPETRAITS_H
