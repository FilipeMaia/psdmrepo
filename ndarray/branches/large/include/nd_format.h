#ifndef NDARRAY_FORMAT_H
#define NDARRAY_FORMAT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class format.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iosfwd>
#include <tr1/type_traits>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------


namespace ndarray_details {

// we want to print characters as numbers
template <typename T>
inline
T printable(T value) {
  return value;
}
inline
int printable(char value) {
  return int(value);
}
inline
int printable(signed char value) {
  return int(value);
}
inline
int printable(unsigned char value) {
  return int(value);
}

// pretty printing of the type name, only limited subset of type names supported
template <typename T>
struct TypeName {
  static const char* typeName() { return "T"; }
};
template <>
struct TypeName<char> {
  static const char* typeName() { return "char"; }
};
template <>
struct TypeName<signed char> {
  static const char* typeName() { return "signed char"; }
};
template <>
struct TypeName<unsigned char> {
  static const char* typeName() { return "unsigned char"; }
};
template <>
struct TypeName<short> {
  static const char* typeName() { return "short"; }
};
template <>
struct TypeName<unsigned short> {
  static const char* typeName() { return "unsigned short"; }
};
template <>
struct TypeName<int> {
  static const char* typeName() { return "int"; }
};
template <>
struct TypeName<unsigned int> {
  static const char* typeName() { return "unsigned int"; }
};
template <>
struct TypeName<long> {
  static const char* typeName() { return "long"; }
};
template <>
struct TypeName<unsigned long> {
  static const char* typeName() { return "unsigned long"; }
};
template <>
struct TypeName<float> {
  static const char* typeName() { return "float"; }
};
template <>
struct TypeName<double> {
  static const char* typeName() { return "double"; }
};


template <typename ElemType, unsigned NDim>
struct dump_ndarray_data {
  static void dump(std::ostream& str, const ndarray<ElemType, NDim>& array, unsigned offset=0)
  {
    char sep[] = ",\n                 ";
    if (offset+3 < sizeof sep) sep[offset+3] = '\0';
    str << '[';
    unsigned size = array.shape()[0];
    if (size > 7) {
      for (unsigned i = 0; i != 3; ++ i) {
        if (i) str << sep;
        dump_ndarray_data<ElemType, NDim-1>::dump(str, array[i], offset+1);
      }
      str << sep << "...";
      for (unsigned i = size-3; i != size; ++ i) {
        str << sep;
        dump_ndarray_data<ElemType, NDim-1>::dump(str, array[i], offset+1);
      }
    } else {
      for (unsigned i = 0; i != size; ++ i) {
        if (i) str << sep;
        dump_ndarray_data<ElemType, NDim-1>::dump(str, array[i], offset+1);
      }
    }
    str << ']';
  }
};

template <typename ElemType>
struct dump_ndarray_data<ElemType, 1> {
  static void dump(std::ostream& str, const ndarray<ElemType, 1>& array, unsigned offset=0)
  {
    str << '[';
    unsigned size = array.size();
    if (size > 9) {
      for (unsigned i = 0; i != 4; ++ i) str << printable(array[i]) << ", ";
      str << "...";
      for (unsigned i = size-4; i != size; ++ i) str << ", " << printable(array[i]);
    } else {
      for (unsigned i = 0; i != size; ++ i) {
        if (i) str << ", ";
        str << printable(array[i]);
      }
    }
    str << ']';
  }
};

} // namespace ndarray_details

/// @addtogroup ndarray

/**
 *  @ingroup ndarray
 *
 *  @brief Methods for formatting ndarray objects.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename ElemType, unsigned NDim>
std::ostream&
operator<<(std::ostream& str, const ndarray<ElemType, NDim>& array)
{
  str << "ndarray<";
  if (std::tr1::is_const<ElemType>::value) str << "const ";
  str << ndarray_details::TypeName<typename std::tr1::remove_const<ElemType>::type>::typeName();
  str << ">(shape=(";
  for (unsigned i = 0; i != NDim; ++ i) {
    if (i) str << ',';
    str << array.shape()[i];
  }
  str << "), data=";
  ndarray_details::dump_ndarray_data<ElemType, NDim>::dump(str, array);
  str << ")";
  return str;
}


#endif // NDARRAY_FORMAT_H
